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
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# 复现训练代码中的类定义（为了加载pickle）
# =============================================================================

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

# =============================================================================
# 论文Table 2基准
# =============================================================================

PAPER_TABLE2 = {
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

# =============================================================================
# 测试环境
# =============================================================================

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

# =============================================================================
# 测试函数
# =============================================================================

def calc_sharpe(returns):
    """计算Sharpe ratio"""
    if len(returns) == 0:
        return 0
    er = np.mean(returns) * 252
    std = np.std(returns) * np.sqrt(252)
    return er / std if std > 0 else 0

def test_dqn(agent, prices, returns):
    """测试DQN模型"""
    env = SimpleEnv(prices, returns)
    state = env.reset()
    
    returns_list = []
    
    for _ in range(len(returns) - 100):
        action_idx = agent.get_action(state, epsilon=0.0)
        action = float(action_idx - 1)  # {-1, 0, 1}
        
        next_state, reward, done = env.step(action)
        returns_list.append(reward)
        state = next_state
        
        if done:
            break
    
    return calc_sharpe(returns_list)

# =============================================================================
# 主测试
# =============================================================================

print("="*80)
print("📊 测试LSTM DQN模型 vs 论文Table 2")
print("="*80)
print()

# 手动加载模型（避免pickle问题）
import pickle

print("加载训练好的模型...")
try:
    # 尝试直接加载
    with open('models_lstm_20260319_235044.pkl', 'rb') as f:
        models = pickle.load(f)
    print("✅ 模型加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    print("\n将重新训练模型进行测试...")
    
    # 如果加载失败，重新训练
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
    print(f"{'策略':<10} | {'我们':>10} | {'论文':>10} | {'差距':>10} | {'状态':>10}")
    print("-" * 60)
    
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
            sharpe = calc_sharpe(returns)
            long_sharpes.append(sharpe)
        except:
            continue
    
    avg_long = np.mean(long_sharpes) if long_sharpes else 0
    paper_long = PAPER_TABLE2[asset_class]['Long']
    diff_long = avg_long - paper_long
    status_long = '✅' if abs(diff_long) < 0.5 else ('⚠️' if abs(diff_long) < 1.0 else '❌')
    
    print(f"{'Long':<10} | {avg_long:>10.3f} | {paper_long:>10.3f} | {diff_long:>+10.3f} | {status_long:>10}")
    
    all_results.append({
        'Asset Class': asset_class,
        'Strategy': 'Long',
        'Ours': avg_long,
        'Paper': paper_long,
        'Diff': diff_long
    })
    
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
            print(f"  {ticker}: {sharpe:.3f}")
        except Exception as e:
            print(f"  {ticker}: ⚠️ {e}")
            continue
    
    avg_dqn = np.mean(dqn_sharpes) if dqn_sharpes else 0
    paper_dqn = PAPER_TABLE2[asset_class]['DQN']
    diff_dqn = avg_dqn - paper_dqn
    status_dqn = '✅' if abs(diff_dqn) < 0.5 else ('⚠️' if abs(diff_dqn) < 1.0 else '❌')
    
    print(f"\n{'DQN':<10} | {avg_dqn:>10.3f} | {paper_dqn:>10.3f} | {diff_dqn:>+10.3f} | {status_dqn:>10}")
    
    all_results.append({
        'Asset Class': asset_class,
        'Strategy': 'DQN',
        'Ours': avg_dqn,
        'Paper': paper_dqn,
        'Diff': diff_dqn
    })

# 保存结果
df_results = pd.DataFrame(all_results)
df_results.to_csv('lstm_test_results.csv', index=False)

print("\n\n" + "="*80)
print("📊 总结")
print("="*80)

print("\n✅ LSTM DQN测试完成")
print(f"💾 结果已保存: lstm_test_results.csv")

print("\n最佳匹配:")
for r in all_results:
    if abs(r['Diff']) < 0.5:
        print(f"  {r['Asset Class']} {r['Strategy']}: {r['Ours']:.3f} vs {r['Paper']:.3f} (差{r['Diff']:+.3f}) ⚠️")

print("\n⚠️ 结果可能仍不如论文，原因:")
print("  1. 训练episodes较少 (200 vs 论文可能更多)")
print("  2. 简化的状态空间 (16维 vs 论文可能更多)")
print("  3. 数据源差异 (Yahoo Finance vs Pinnacle)")
print("  4. 测试期不同")
