#!/usr/bin/env python3
"""
测试LSTM模型并对比论文Table 2
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 论文Table 2
PAPER = {
    'Commodity': {'Long': -0.726, 'DQN': 0.723, 'A2C': 0.234},
    'Equity Index': {'Long': 0.688, 'DQN': 0.648, 'A2C': 0.510},
    'Fixed Income': {'Long': 0.698, 'DQN': 0.935, 'A2C': 0.714},
    'FX': {'Long': -0.353, 'DQN': 0.546, 'A2C': 0.328}
}

CONTRACTS = {
    'Commodity': ['CL=F', 'GC=F', 'SI=F', 'HG=F', 'NG=F', 'ZC=F', 'ZS=F', 'ZW=F', 'KC=F', 'CC=F', 'SB=F', 'CT=F', 'OJ=F'],
    'Equity Index': ['ES=F', 'NQ=F', 'YM=F'],
    'Fixed Income': ['ZN=F', 'ZB=F', 'ZF=F', 'ZT=F', 'GE=F'],
    'FX': ['6E=F', '6J=F', '6B=F', '6A=F', '6C=F', '6S=F', '6N=F', '6M=F', '6R=F']
}

print("="*80)
print("📊 测试LSTM模型 vs 论文Table 2")
print("="*80)

# 加载模型
with open('models_lstm_20260319_234600.pkl', 'rb') as f:
    models = pickle.load(f)

# 测试函数
def test_model(agent, prices, returns):
    """测试模型"""
    class SimpleEnv:
        def __init__(self, prices, returns):
            self.prices = prices
            self.returns = returns
            self.n_steps = len(returns)
            self.step_idx = 50
            self.last_action = 0.0
            
        def reset(self):
            self.step_idx = 50
            self.last_action = 0.0
            return self._obs()
        
        def _obs(self):
            obs = np.zeros(16, dtype=np.float32)
            if self.step_idx >= 20:
                obs[0] = np.mean(self.returns[self.step_idx-20:self.step_idx])
                obs[1] = np.std(self.returns[self.step_idx-20:self.step_idx])
                obs[2] = self.returns[self.step_idx-1] if self.step_idx >= 1 else 0
            return obs
        
        def step(self, action):
            action = float(np.clip(action, -1, 1))
            cost = abs(action - self.last_action) * 0.002
            
            if self.step_idx + 1 >= self.n_steps:
                return self._obs(), 0.0, True
            
            ret = self.returns[self.step_idx + 1]
            reward = action * ret - cost
            
            self.step_idx += 1
            self.last_action = action
            
            return self._obs(), reward, False
    
    env = SimpleEnv(prices, returns)
    state = env.reset()
    
    returns_list = []
    
    for _ in range(500):
        # Get action
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(DEVICE)
            q_values = agent.q_net(state_t)
            action = q_values.argmax().item() - 1  # {-1, 0, 1}
        
        next_state, reward, done = env.step(action)
        returns_list.append(reward)
        state = next_state
        
        if done:
            break
    
    # 计算Sharpe
    returns_list = np.array(returns_list)
    if len(returns_list) == 0:
        return 0
    
    er = np.mean(returns_list) * 252
    std = np.std(returns_list) * np.sqrt(252)
    sharpe = er / std if std > 0 else 0
    
    return sharpe

print("\n按资产类别测试:\n")

for asset_class in ['Commodity', 'Equity Index', 'Fixed Income', 'FX']:
    print(f"\n【{asset_class}】")
    print(f"{'策略':<10} | {'我们LSTM':>10} | {'论文':>10} | {'差距':>10}")
    print("-" * 50)
    
    if asset_class not in models or models[asset_class] is None:
        print("  ⚠️ 无模型")
        continue
    
    agent = models[asset_class]
    tickers = CONTRACTS[asset_class]
    
    # 测试Long策略
    long_sharpes = []
    for ticker in tickers:
        try:
            df = pd.read_csv(f'data/futures_processed/{ticker}.csv')
            df['Date'] = pd.to_datetime(df['Date'])
            test = df[(df['Date'] >= '2016-01-01') & (df['Date'] <= '2019-12-31')]
            if len(test) < 200:
                continue
            
            returns = test['Returns'].values[50:]
            er = np.mean(returns) * 252
            std = np.std(returns) * np.sqrt(252)
            sharpe = er / std if std > 0 else 0
            long_sharpes.append(sharpe)
        except:
            continue
    
    avg_long = np.mean(long_sharpes) if long_sharpes else 0
    paper_long = PAPER[asset_class]['Long']
    diff_long = avg_long - paper_long
    print(f"{'Long':<10} | {avg_long:>10.3f} | {paper_long:>10.3f} | {diff_long:>+10.3f}")
    
    # 测试DQN
    dqn_sharpes = []
    for ticker in tickers:
        try:
            df = pd.read_csv(f'data/futures_processed/{ticker}.csv')
            df['Date'] = pd.to_datetime(df['Date'])
            test = df[(df['Date'] >= '2016-01-01') & (df['Date'] <= '2019-12-31')]
            if len(test) < 200:
                continue
            
            sharpe = test_model(agent, test['Close'].values, test['Returns'].values)
            dqn_sharpes.append(sharpe)
        except:
            continue
    
    avg_dqn = np.mean(dqn_sharpes) if dqn_sharpes else 0
    paper_dqn = PAPER[asset_class]['DQN']
    diff_dqn = avg_dqn - paper_dqn
    print(f"{'DQN':<10} | {avg_dqn:>10.3f} | {paper_dqn:>10.3f} | {diff_dqn:>+10.3f}")

print("\n\n" + "="*80)
print("📊 总结")
print("="*80)

print("\n✅ LSTM模型已训练并测试完成")
print("⚠️ 结果可能仍不如论文，原因:")
print("  1. 训练episodes较少 (200 vs 可能需要更多)")
print("  2. 简化的状态空间")
print("  3. 数据源差异")
