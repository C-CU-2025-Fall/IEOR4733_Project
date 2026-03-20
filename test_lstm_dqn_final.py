#!/usr/bin/env python3
"""
测试LSTM DQN模型并对比论文Table 2
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 复现训练代码中的类定义
class LSTMNetwork(nn.Module):
    def __init__(self, input_dim, hidden_sizes=[64, 32], output_dim=3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_sizes[0], batch_first=True)
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.fc = nn.Linear(hidden_sizes[1], output_dim)
        self.leaky_relu = nn.LeakyReLU(0.01)
        
    def forward(self, x):
        out1, _ = self.lstm1(x)
        out1 = self.leaky_relu(out1)
        out2, _ = self.lstm2(out1)
        out2 = self.leaky_relu(out2)
        return self.fc(out2[:, -1, :])

class SimpleDQN:
    def __init__(self, state_dim=16, n_actions=3):
        self.q_net = LSTMNetwork(state_dim, [64, 32], n_actions).to(DEVICE)
        
    def get_action(self, state, epsilon=0.0):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(DEVICE)
            q_values = self.q_net(state_t)
            return q_values.argmax().item()

# 论文Table 2
PAPER_TABLE2 = {
    'Commodity': {'Long': -0.726, 'DQN': 0.723},
    'Equity Index': {'Long': 0.688, 'DQN': 0.648},
    'Fixed Income': {'Long': 0.698, 'DQN': 0.935},
    'FX': {'Long': -0.353, 'DQN': 0.546}
}

CONTRACTS = {
    'Commodity': ['CL=F', 'GC=F', 'SI=F', 'HG=F', 'NG=F', 'ZC=F', 'ZS=F', 'ZW=F', 'KC=F', 'CC=F', 'SB=F', 'CT=F', 'OJ=F'],
    'Equity Index': ['ES=F', 'NQ=F', 'YM=F'],
    'Fixed Income': ['ZN=F', 'ZB=F', 'ZF=F', 'ZT=F', 'GE=F'],
    'FX': ['6E=F', '6J=F', '6B=F', '6A=F', '6C=F', '6S=F', '6N=F', '6M=F', '6R=F']
}

# 简单环境
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

def calc_sharpe(returns):
    if len(returns) == 0:
        return 0
    er = np.mean(returns) * 252
    std = np.std(returns) * np.sqrt(252)
    return er / std if std > 0 else 0

def test_dqn(agent, prices, returns):
    env = SimpleEnv(prices, returns)
    state = env.reset()
    returns_list = []
    
    for _ in range(len(returns) - 100):
        action_idx = agent.get_action(state, epsilon=0.0)
        action = float(action_idx - 1)
        next_state, reward, done = env.step(action)
        returns_list.append(reward)
        state = next_state
        if done:
            break
    
    return calc_sharpe(returns_list)

print("="*80)
print("📊 测试LSTM DQN模型 vs 论文Table 2")
print("="*80)
print()

# 重新训练（因为pickle加载会失败）
print("重新训练模型（避免pickle加载问题）...")
import sys
sys.path.append('.')
from train_lstm_verified import train_asset_class

models = {}
for asset_class, tickers in CONTRACTS.items():
    print(f"\n训练 {asset_class}...")
    models[asset_class] = train_asset_class(asset_class, tickers)

print("\n" + "="*80)
print("📊 测试结果 vs 论文Table 2")
print("="*80)

all_results = []

for asset_class in ['Commodity', 'Equity Index', 'Fixed Income', 'FX']:
    print(f"\n{'='*70}")
    print(f"【{asset_class}】")
    print('='*70)
    
    if asset_class not in models or models[asset_class] is None:
        print("  ⚠️ 无模型")
        continue
    
    agent = models[asset_class]
    tickers = CONTRACTS[asset_class]
    
    # 测试Long
    long_sharpes = []
    for ticker in tickers:
        try:
            df = pd.read_csv(f'data/futures_processed/{ticker}.csv')
            df['Date'] = pd.to_datetime(df['Date'])
            test = df[(df['Date'] >= '2016-01-01') & (df['Date'] <= '2019-12-31')]
            if len(test) < 200:
                continue
            
            returns = test['Returns'].values[50:]
            sharpe = calc_sharpe(returns)
            long_sharpes.append(sharpe)
        except:
            continue
    
    avg_long = np.mean(long_sharpes) if long_sharpes else 0
    paper_long = PAPER_TABLE2[asset_class]['Long']
    diff_long = avg_long - paper_long
    
    print(f"Long: {avg_long:.3f} vs 论文 {paper_long:.3f} ({diff_long:+.3f})")
    
    # 测试DQN
    dqn_sharpes = []
    for ticker in tickers:
        try:
            df = pd.read_csv(f'data/futures_processed/{ticker}.csv')
            df['Date'] = pd.to_datetime(df['Date'])
            test = df[(df['Date'] >= '2016-01-01') & (df['Date'] <= '2019-12-31')]
            if len(test) < 200:
                continue
            
            sharpe = test_dqn(agent, test['Close'].values, test['Returns'].values)
            dqn_sharpes.append(sharpe)
            print(f"  {ticker}: DQN Sharpe = {sharpe:.3f}")
        except Exception as e:
            print(f"  {ticker}: ⚠️ {e}")
            continue
    
    avg_dqn = np.mean(dqn_sharpes) if dqn_sharpes else 0
    paper_dqn = PAPER_TABLE2[asset_class]['DQN']
    diff_dqn = avg_dqn - paper_dqn
    
    print(f"\nDQN: {avg_dqn:.3f} vs 论文 {paper_dqn:.3f} ({diff_dqn:+.3f})")
    
    all_results.append({
        'Asset Class': asset_class,
        'Long': avg_long,
        'Long_Paper': paper_long,
        'DQN': avg_dqn,
        'DQN_Paper': paper_dqn
    })

# 保存
df_results = pd.DataFrame(all_results)
df_results.to_csv('lstm_dqn_test_results.csv', index=False)

print("\n\n" + "="*80)
print("📊 总结")
print("="*80)
print(f"\n💾 结果已保存: lstm_dqn_test_results.csv")

print("\n最接近论文的:")
for r in all_results:
    if abs(r['DQN'] - r['DQN_Paper']) < 1.0:
        print(f"  {r['Asset Class']}: DQN {r['DQN']:.3f} vs {r['DQN_Paper']:.3f}")
