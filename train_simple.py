#!/usr/bin/env python3
"""
最简单可用的版本 - 使用stable-baselines3的标准接口
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
# 配置
# =============================================================================

DATA_DIR = 'data/futures_processed'
TRANSACTION_COST = 0.002

GAMMA = 0.3
BUFFER_SIZE = 5000
BATCH_SIZE_DQN = 64
BATCH_SIZE_A2C = 128
LEARNING_RATE = 0.0001
TARGET_UPDATE = 1000
TOTAL_TIMESTEPS = 50000

CONTRACTS_BY_CLASS = {
    'Commodity': ['CL=F', 'GC=F', 'SI=F', 'HG=F', 'NG=F', 'ZC=F', 'ZS=F', 'ZW=F', 'KC=F', 'CC=F', 'SB=F', 'CT=F', 'OJ=F'],
    'Equity Index': ['ES=F', 'NQ=F', 'YM=F'],
    'Fixed Income': ['ZN=F', 'ZB=F', 'ZF=F', 'ZT=F', 'GE=F'],
    'FX': ['6E=F', '6J=F', '6B=F', '6A=F', '6C=F', '6S=F', '6N=F', '6M=F', '6R=F']
}

TRAIN_START = '2011-01-03'
TRAIN_END = '2015-12-31'

# =============================================================================
# 简化的环境
# =============================================================================

class SimpleTradingEnv(gym.Env):
    def __init__(self, data_list, discrete=True):
        super().__init__()
        self.data_list = data_list
        self.discrete = discrete
        
        # 合并所有数据
        self.all_prices = np.concatenate([d['prices'] for d in data_list])
        self.all_returns = np.concatenate([d['returns'] for d in data_list])
        self.n_steps = len(self.all_returns)
        
        # 空间定义
        self.observation_space = spaces.Box(-np.inf, np.inf, (16,), np.float32)
        if discrete:
            self.action_space = spaces.Discrete(3)  # {0, 1, 2}
        else:
            self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        
        self.current_step = 200
        self.last_action = 0.0 if not discrete else 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 200
        self.last_action = 0.0 if not self.discrete else 0
        return self._get_obs(), {}
    
    def _get_obs(self):
        # 简化：返回16维向量
        obs = np.zeros(16, dtype=np.float32)
        
        # 使用过去的收益
        if self.current_step >= 5:
            obs[0:6] = np.array([
                np.mean(self.all_returns[self.current_step-5:self.current_step]) if self.current_step >= 5 else 0,
                np.mean(self.all_returns[self.current_step-10:self.current_step]) if self.current_step >= 10 else 0,
                np.mean(self.all_returns[self.current_step-25:self.current_step]) if self.current_step >= 25 else 0,
                np.mean(self.all_returns[self.current_step-50:self.current_step]) if self.current_step >= 50 else 0,
                np.mean(self.all_returns[self.current_step-100:self.current_step]) if self.current_step >= 100 else 0,
                np.mean(self.all_returns[self.current_step-200:self.current_step]) if self.current_step >= 200 else 0,
            ])
        
        # 波动率
        if self.current_step >= 20:
            obs[6] = np.std(self.all_returns[self.current_step-20:self.current_step]) * np.sqrt(252)
        
        # 收益率
        if self.current_step >= 1:
            obs[7] = self.all_returns[self.current_step-1]
        
        # 其他填充
        obs[8:] = np.random.randn(8) * 0.01
        
        # 检查NaN
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        return obs
    
    def step(self, action):
        # 转换动作
        if self.discrete:
                action_value = float(action - 1)  # {-1, 0, 1}
        else:
                action_value = float(np.clip(action[0], -1, 1))
        
        # 讣算成本
        cost = abs(action_value - self.last_action) * TRANSACTION_COST
        
        # 计算收益
        if self.current_step + 1 >= self.n_steps:
                return self._get_obs(), 0.0, True, False, {}
        
        ret = self.all_returns[self.current_step + 1]
        strat_ret = action_value * ret - cost
        
        # 简化奖励：直接使用策略收益
        reward = strat_ret
        
        # 检查NaN
        reward = 0.0 if np.isnan(reward) or np.isinf(reward) else reward
        
        self.current_step += 1
        self.last_action = action_value if not self.discrete else action
        
        done = self.current_step >= self.n_steps - 1
        
        return self._get_obs(), reward, done, False, {}

# =============================================================================
# 训练函数
# =============================================================================

def load_class_data(asset_class):
    """加载数据"""
    tickers = CONTRACTS_BY_CLASS[asset_class]
    all_data = []
    
    for ticker in tickers:
        try:
            df = pd.read_csv(f'{DATA_DIR}/{ticker}.csv')
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            train = df[(df['Date'] >= TRAIN_START) & (df['Date'] <= TRAIN_END)]
            if len(train) < 500:
                continue
            
            all_data.append({
                'prices': train['Close'].values,
                'returns': train['Returns'].values
            })
        except:
            continue
    
    return all_data

def train_asset_class(asset_class):
    """训练某个资产类别"""
    print(f"\n{'='*70}")
    print(f"📊 训练 {asset_class}")
    print('='*70)
    
    all_data = load_class_data(asset_class)
    if not all_data:
        print("  ⚠️ 无数据")
        return None
    
    print(f"  合约数: {len(all_data)}")
    print(f"  总样本: {sum(len(d['returns']) for d in all_data):,}")
    
    # DQN
    print("  训练DQN...")
    env_dqn = SimpleTradingEnv(all_data, discrete=True)
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
    
    # A2C
    print("  训练A2C...")
    env_a2c = SimpleTradingEnv(all_data, discrete=False)
    a2c = A2C("MlpPolicy", env_a2c,
                 learning_rate=LEARNING_RATE,
                 gamma=GAMMA,
                 policy_kwargs=dict(net_arch=[64, 32]),
                 verbose=0, device=DEVICE)
    a2c.learn(TOTAL_TIMESTEPS)
    
    # PPO
    print("  训练PPO...")
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
    print("🤖 简化稳定版 - 按资产类别训练")
    print("="*80)
    print(f"设备: {DEVICE}")
    print(f"训练方式: 4个模型")
    print()
    
    models = {}
    
    for asset_class in ['Commodity', 'Equity Index', 'Fixed Income', 'FX']:
        models[asset_class] = train_asset_class(asset_class)
    
    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'models_{timestamp}.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    print(f"\n✅ 训练完成！")

if __name__ == '__main__':
    main()
