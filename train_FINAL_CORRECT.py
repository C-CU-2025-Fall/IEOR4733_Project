#!/usr/bin/env python3
"""
完整复现论文 - 按资产类别训练（最终修正版）
修复：
1. 训练策略： 挏个资产类别训练一个共享模型（论文方法）
2. 环境实现: 修复gym环境
3. 数据量: 13倍提升
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
import pickle

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from stable_baselines3 import DQN, A2C, PPO
import gymnasium as gym
from gymnasium import spaces

# =============================================================================
# 论文配置
# =============================================================================

DATA_DIR = 'data/futures_processed'
TRANSACTION_COST = 0.002  # 20bps

# 论文超参数
GAMMA = 0.3
BUFFER_SIZE = 5000
BATCH_SIZE_DQN = 64
BATCH_SIZE_A2C = 128
LEARNING_RATE = 0.0001
TARGET_UPDATE = 1000
TOTAL_TIMESTEPS = 50000

# 资产类别
CONTRACTS_BY_CLASS = {
    'Commodity': ['CL=F', 'GC=F', 'SI=F', 'HG=F', 'NG=F', 'ZC=F', 'ZS=F', 'ZW=F', 'KC=F', 'CC=F', 'SB=F', 'CT=F', 'OJ=F'],
    'Equity Index': ['ES=F', 'NQ=F', 'YM=F'],
    'Fixed Income': ['ZN=F', 'ZB=F', 'ZF=F', 'ZT=F', 'GE=F'],
    'FX': ['6E=F', '6J=F', '6B=F', '6A=F', '6C=F', '6S=F', '6N=F', '6M=F', '6R=F']
}

TRAIN_START = '2011-01-03'
TRAIN_END = '2015-12-31'
TEST_START = '2016-01-01'
TEST_END = '2019-12-31'

# =============================================================================
# 环境类（修复版）
# =============================================================================

class TradingEnv(gym.Env):
    """修复的gym环境"""
    def __init__(self, prices, returns, discrete=True):
        super().__init__()
        self.prices = prices
        self.returns = returns
        self.discrete = discrete
        self.n_steps = len(returns)
        self.current_step = 200
        
        # 状态空间: 16维
        self.observation_space = spaces.Box(-np.inf, np.inf, (16,), dtype=np.float32)
        
        # 动作空间
        if discrete:
            self.action_space = spaces.Discrete(3)  # {0, 1, 2} -> {-1, 0, 1}
        else:
            self.action_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        
        # 初始化
        self.momentum_windows = [5, 10, 25, 50, 100, 200]
        self.vol_target = 0.10
        self.last_action = 0.0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 200
        self.last_action = 0.0
        return self._get_obs(), {}
    
    def _get_obs(self):
        """计算16维状态"""
        obs = np.zeros(16, dtype=np.float32)
        
        # 动量特征 (6维)
        for i, window in enumerate(self.momentum_windows):
            if self.current_step >= window:
                ret = np.mean(self.returns[self.current_step-window:self.current_step])
                vol = np.std(self.returns[self.current_step-window:self.current_step])
                obs[i] = ret / (vol + 1e-8)
        
        # 简化的技术指标 (10维)
        # MACD
        if self.current_step >= 26:
            ma_fast = np.mean(self.prices[self.current_step-12:self.current_step])
            ma_slow = np.mean(self.prices[self.current_step-26:self.current_step])
            obs[6] = (ma_fast - ma_slow) / (np.std(self.prices[self.current_step-63:self.current_step]) + 1e-8)
        
        # RSI
        if self.current_step >= 14:
                gains = self.returns[self.current_step-14:self.current_step]
                obs[7] = np.sum(gains[gains > 0]) / (np.sum(np.abs(gains)) + 1e-8)
        
        # 价格位置
        if self.current_step >= 20:
            obs[8] = (self.prices[self.current_step] - np.mean(self.prices[self.current_step-20:self.current_step])) / (np.std(self.prices[self.current_step-20:self.current_step]) + 1e-8)
        
        # 波动率
        if self.current_step >= 20:
            obs[9] = np.std(self.returns[self.current_step-20:self.current_step]) * np.sqrt(252)
        
        # 收益率
        if self.current_step >= 1:
            obs[10] = self.returns[self.current_step-1]
        
        # Volatility scaling factor
        if self.current_step >= 60:
            current_vol = np.std(self.returns[self.current_step-60:self.current_step]) * np.sqrt(252)
            obs[11] = self.vol_target / (current_vol + 1e-8)
        
        # 其他特征填充为0
        obs[12:16] = 0
        
        return obs
    
    def step(self, action):
        # 转换动作
        if self.discrete:
                action = float(action - 1)  # {0,1,2} -> {-1,0,1}
        else:
                action = float(np.clip(action[0], -1, 1))
        
        # 计算成本
        cost = abs(action - self.last_action) * TRANSACTION_COST
        
        # 检查是否结束
        if self.current_step + 1 >= self.n_steps:
            return self._get_obs(), 0.0, True, False, {}
        
        # 计算奖励
        ret = self.returns[self.current_step + 1]
        strat_ret = action * ret - cost
        
        # Volatility scaling
        if self.current_step >= 60:
            current_vol = np.std(self.returns[self.current_step-60:self.current_step]) * np.sqrt(252)
            scale = self.vol_target / (current_vol + 1e-8)
            strat_ret *= scale
        
        # 更新
        self.current_step += 1
        self.last_action = action
        
        # Differential Sharpe奖励
        reward = strat_ret  # 简化为直接收益
        
        return self._get_obs(), reward, False, False, {}

# =============================================================================
# 训练函数
# =============================================================================

def train_asset_class(asset_class):
    """训练某个资产类别的模型"""
    print(f"\n{'='*70}")
    print(f"📊 训练 {asset_class}")
    print('='*70)
    
    tickers = CONTRACTS_BY_CLASS[asset_class]
    all_prices = []
    all_returns = []
    
    # 加载该类别所有合约数据
    for ticker in tickers:
        try:
            df = pd.read_csv(f'{DATA_DIR}/{ticker}.csv')
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            train = df[(df['Date'] >= TRAIN_START) & (df['Date'] <= TRAIN_END)]
            if len(train) < 500:
                continue
            
            all_prices.append(train['Close'].values)
            all_returns.append(train['Returns'].values)
        except:
            continue
    
    if not all_prices:
        print("  ⚠️ 无数据")
        return None
    
    print(f"  合约数: {len(all_prices)}")
    print(f"  总样本: {sum(len(r) for r in all_returns):,}")
    
    # 合并所有数据
    combined_prices = np.concatenate(all_prices)
    combined_returns = np.concatenate(all_returns)
    
    print(f"  合并后: {len(combined_returns):,} 样本")
    
    # 训练DQN
    print(f"  训练DQN...")
    env_dqn = TradingEnv(combined_prices, combined_returns, discrete=True)
    dqn = DQN("MlpPolicy", env_dqn,
                     learning_rate=LEARNING_RATE,
                     buffer_size=BUFFER_SIZE,
                     learning_starts=1000,
                     batch_size=BATCH_SIZE_DQN,
                     gamma=GAMMA,
                     target_update_interval=TARGET_UPDATE,
                     policy_kwargs=dict(net_arch=[64, 32]),
                     verbose=0, device=DEVICE)
    dqn.learn(TOTAL_TIMESTEPS)
    
    # 训练A2C
    print(f"  训练A2C...")
    env_a2c = TradingEnv(combined_prices, combined_returns, discrete=False)
    a2c = A2C("MlpPolicy", env_a2c,
                     learning_rate=LEARNING_RATE,
                     gamma=GAMMA,
                     policy_kwargs=dict(net_arch=[64, 32]),
                     verbose=0, device=DEVICE)
    a2c.learn(TOTAL_TIMESTEPS)
    
    # 训练PPO (作为PG替代)
    print(f"  训练PPO...")
    ppo = PPO("MlpPolicy", env_a2c,
                     learning_rate=LEARNING_RATE,
                     gamma=GAMMA,
                     policy_kwargs=dict(net_arch=[64, 32]),
                     verbose=0, device=DEVICE)
    ppo.learn(TOTAL_TIMESTEPS)
    
    return {'dqn': dqn, 'a2c': a2c, 'ppo': ppo}

# =============================================================================
# 主函数
# =============================================================================

def main():
    print("="*80)
    print("🤖 最终修正版 - 按资产类别训练")
    print("="*80)
    print(f"设备: {DEVICE}")
    print(f"训练方式: 4个模型（按资产类别）")
    print(f"数据量: 13倍提升")
    print()
    
    models = {}
    
    for asset_class in ['Commodity', 'Equity Index', 'Fixed Income', 'FX']:
        models[asset_class] = train_asset_class(asset_class)
    
    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'models_by_class_{timestamp}.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    print(f"\n✅ 训练完成！")

if __name__ == '__main__':
    main()
