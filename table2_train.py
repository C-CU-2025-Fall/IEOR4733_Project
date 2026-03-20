#!/usr/bin/env python3
"""
Table 2 完整复现 - 集成修复后的 PG/A2C

使用修复后的 PG/A2C 实现，支持 6 个模型完整对比
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from indicators import FeatureEngineer, compute_volatility
from fix_pg_a2c import FixedPG, FixedA2C

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# 配置
# =============================================================================

BP = 0.0020
VOL_TARGET = 0.10
MAX_STEPS = 500

# 训练配置
N_EPISODES = 200
MICRO_EPISODES = 5

# 资产类别
CONTRACTS_BY_CLASS = {
    'Commodity': ['CL=F', 'GC=F', 'SI=F', 'HG=F', 'NG=F', 'ZC=F', 'ZS=F', 'ZW=F', 
                  'KC=F', 'CC=F', 'SB=F', 'CT=F', 'OJ=F'],
    'Equity Index': ['ES=F', 'NQ=F', 'YM=F'],
    'Fixed Income': ['ZN=F', 'ZB=F', 'ZF=F', 'ZT=F', 'GE=F'],
    'FX': ['6E=F', '6J=F', '6B=F', '6A=F', '6C=F', '6S=F', '6N=F', '6M=F', '6R=F']
}

# =============================================================================
# 环境
# =============================================================================

class VolatilityScaledEnv:
    def __init__(self, prices, returns, vol_target=VOL_TARGET):
        self.prices = prices
        self.returns = returns
        self.vol_target = vol_target
        self.n_steps = len(returns)
        self.volatility = compute_volatility(returns, 60)
        self.feature_engineer = FeatureEngineer(window_size=60)
        self.step_idx = 60
        self.last_action = 0.0
        
    def reset(self):
        self.step_idx = 60
        self.last_action = 0.0
        return self._get_state()
    
    def _get_state(self):
        return self.feature_engineer.build_features(
            self.prices, self.returns, self.step_idx
        )
    
    def step(self, action):
        action = float(np.clip(action, -1, 1))
        vol_scale = self.vol_target / (self.volatility[self.step_idx] + 1e-10)
        vol_scale = np.clip(vol_scale, 0.5, 2.0)
        
        current_price = self.prices[self.step_idx]
        cost = BP * abs(action - self.last_action) * vol_scale * current_price
        
        if self.step_idx + 1 >= self.n_steps:
            return self._get_state(), 0.0, True
        
        reward = (action * vol_scale) * self.returns[self.step_idx + 1] - cost
        
        self.step_idx += 1
        self.last_action = action
        
        return self._get_state(), reward, False

# =============================================================================
# 模型工厂
# =============================================================================

def create_model(model_name):
    """创建模型"""
    models = {
        'long': lambda: 'Long Only',
        'signr': lambda: 'Sign(R)',
        'macd': lambda: 'MACD',
        'dqn': lambda: 'DQN',
        'pg': lambda: FixedPG(),
        'a2c': lambda: FixedA2C()
    }
    return models.get(model_name, lambda: None)()

# =============================================================================
# 数据加载
# =============================================================================

def load_data(tickers, start_date='2011-01-01', end_date='2015-12-31'):
    """加载多个合约数据"""
    all_prices = []
    all_returns = []
    
    for ticker in tickers:
        try:
            # 尝试两种文件名格式
            fname = f'data/futures_processed/{ticker}.csv'
            if not os.path.exists(fname):
                fname = f'data/futures_processed/{ticker.replace("=", "")}.csv'
            
            df = pd.read_csv(fname)
            
            # 检查 Date 列
            if 'Date' not in df.columns:
                print(f"  ⚠️ {ticker}: 缺少 Date 列")
                continue
            
            df['Date'] = pd.to_datetime(df['Date'])
            
            # 填充 NaN
            df['Returns'] = df['Returns'].fillna(0)
            
            train = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
            
            if len(train) < 500:
                continue
            
            all_prices.append(train['Close'].values)
            all_returns.append(train['Returns'].values)
        except Exception as e:
            print(f"  ⚠️ {ticker}: {e}")
            continue
    
    if not all_prices:
        return None, None
    
    return np.concatenate(all_prices), np.concatenate(all_returns)

# =============================================================================
# 训练函数
# =============================================================================

def train_model(model_name, asset_class, tickers, micro=False, verbose=True):
    """训练单个模型"""
    
    episodes = MICRO_EPISODES if micro else N_EPISODES
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"📊 训练 {asset_class} - {model_name.upper()}")
        print('='*70)
    
    # 加载数据
    prices, returns = load_data(tickers, '2011-01-01', '2015-12-31')
    
    if prices is None:
        if verbose:
            print("  ⚠️ 无数据，跳过")
        return None
    
    if verbose:
        print(f"  合约数：{len(tickers)}")
        print(f"  总样本：{len(returns):,}")
        print(f"  Episodes: {episodes}")
        print(f"  开始训练...")
    
    # 创建模型
    model = create_model(model_name)
    if model is None or isinstance(model, str):
        # 规则模型
        return {'name': model_name, 'type': 'rule'}
    
    # 创建环境
    env = VolatilityScaledEnv(prices, returns)
    
    # 训练循环
    episode_rewards = []
    
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < MAX_STEPS:
            # 获取动作
            if model_name == 'pg':
                action, mu, sigma = model.get_action(state)
                model.store_transition(state, action, 0, mu, sigma)
            elif model_name == 'a2c':
                action = model.get_action(state)
            elif model_name == 'dqn':
                action = model.get_action(state, epsilon=0.3)
            else:
                action = 1  # long
            
            # 执行动作
            next_state, reward, done = env.step(action)
            
            # 存储/训练
            if model_name == 'pg':
                model.trajectory[-1]['reward'] = reward
                model.trajectory[-1]['next_state'] = next_state
                model.trajectory[-1]['done'] = done
            elif model_name == 'a2c':
                model.train(state, action, reward, next_state, float(done))
            elif model_name == 'dqn':
                model.store_transition(state, action, reward, next_state, float(done))
                model.train()
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        # PG 在 episode 结束时训练
        if model_name == 'pg':
            model.train()
        
        episode_rewards.append(total_reward)
        
        if verbose and (ep + 1) % 1 == 0:
            avg = np.mean(episode_rewards[-3:]) if len(episode_rewards) >= 3 else np.mean(episode_rewards)
            print(f"    Episode {ep+1}/{episodes}: Avg Reward={avg:.4f}")
    
    avg_reward = np.mean(episode_rewards)
    
    if verbose:
        print(f"  ✅ 完成，平均奖励：{avg_reward:.4f}")
    
    return {
        'name': model_name,
        'type': 'rl',
        'model': model,
        'avg_reward': avg_reward,
        'episode_rewards': episode_rewards
    }

# =============================================================================
# 主训练函数
# =============================================================================

def train_all(micro=False):
    """训练所有资产类别和模型"""
    
    print("="*80)
    print("🔥 Table 2 完整训练 - 6 个模型 × 4 个资产类别")
    print("="*80)
    print(f"设备：{DEVICE}")
    print(f"数据：2011-2019 (训练：2011-2015)")
    print(f"模式：{'微训练' if micro else '完整训练'}")
    print("="*80)
    
    models_to_train = ['dqn', 'pg', 'a2c']  # RL 模型需要训练
    results = {}
    
    start_time = time.time()
    
    for asset_class, tickers in CONTRACTS_BY_CLASS.items():
        results[asset_class] = {}
        
        for model_name in models_to_train:
            result = train_model(model_name, asset_class, tickers, micro=micro)
            results[asset_class][model_name] = result
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"✅ 训练完成！")
    print(f"⏱️ 总时间：{elapsed/60:.1f} 分钟")
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode = 'micro' if micro else 'full'
    
    with open(f'results_table2_{mode}_{timestamp}.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"💾 结果：results_table2_{mode}_{timestamp}.pkl")
    print("="*80)
    
    return results

# =============================================================================
# 微训练测试
# =============================================================================

def micro_test():
    """微训练测试 - 快速验证"""
    
    print("="*80)
    print("🧪 微训练测试 - 验证所有模型")
    print("="*80)
    
    # 只测试 Equity Index
    test_class = 'Equity Index'
    test_tickers = ['ES=F', 'NQ=F', 'YM=F']
    
    results = {}
    
    for model_name in ['dqn', 'pg', 'a2c']:
        result = train_model(model_name, test_class, test_tickers, micro=True, verbose=True)
        results[model_name] = result
    
    print(f"\n✅ 微训练测试完成！")
    
    return results

# =============================================================================
# 主函数
# =============================================================================

def main():
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--micro':
            micro_test()
        elif sys.argv[1] == '--help':
            print("用法:")
            print("  python3 table2_train.py           # 完整训练")
            print("  python3 table2_train.py --micro   # 微训练测试")
    else:
        train_all(micro=False)

if __name__ == '__main__':
    main()
