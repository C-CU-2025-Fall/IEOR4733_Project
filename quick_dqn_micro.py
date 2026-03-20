#!/usr/bin/env python3
"""超简单 DQN 微训练测试"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from indicators import FeatureEngineer, compute_volatility

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 环境
class Env:
    def __init__(self, prices, returns):
        self.prices, self.returns = prices, returns
        self.fe = FeatureEngineer(60)
        self.vol = compute_volatility(returns, 60)
        self.idx = 60
        self.last_a = 0.0
        
    def reset(self):
        self.idx, self.last_a = 60, 0.0
        return self.fe.build_features(self.prices, self.returns, self.idx)
    
    def step(self, action):
        action = np.clip(action, -1, 1)
        vol_scale = np.clip(0.10 / (self.vol[self.idx] + 1e-10), 0.5, 2.0)
        cost = 0.002 * abs(action - self.last_a) * vol_scale * self.prices[self.idx]
        
        if self.idx + 1 >= len(self.returns):
            return self.fe.build_features(self.prices, self.returns, self.idx), 0.0, True
        
        reward = (action * vol_scale) * self.returns[self.idx + 1] - cost
        self.idx += 1
        self.last_a = action
        return self.fe.build_features(self.prices, self.returns, self.idx), reward, False

# LSTM 网络
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(8, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.fc = nn.Linear(32, 3)
        
    def forward(self, x):
        o1, _ = self.lstm1(x)
        o1 = F.leaky_relu(o1, 0.01)
        o2, _ = self.lstm2(o1)
        o2 = F.leaky_relu(o2, 0.01)
        return self.fc(o2[:, -1, :])

# DQN
class DQN:
    def __init__(self):
        self.net = LSTM().to(DEVICE)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.0001)
        self.mem = []
        
    def get_action(self, state, eps=0.3):
        if np.random.random() < eps:
            return np.random.randint(0, 3)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            return self.net(s).argmax().item()
    
    def store(self, s, a, r, s_, d):
        self.mem.append((s, a, r, s_, d))
        if len(self.mem) > 5000:
            self.mem.pop(0)
    
    def train(self):
        if len(self.mem) < 64:
            return 0
        b = self.mem[-500:]
        s = torch.FloatTensor([x[0] for x in b]).to(DEVICE)
        a = torch.LongTensor([x[1] for x in b]).to(DEVICE)
        r = torch.FloatTensor([x[2] for x in b]).to(DEVICE)
        s_ = torch.FloatTensor([x[3] for x in b]).to(DEVICE)
        d = torch.FloatTensor([x[4] for x in b]).to(DEVICE)
        
        cq = self.net(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            nq = self.net(s_).max(1)[0]
            tq = r + (1 - d) * 0.3 * nq
        
        loss = F.mse_loss(cq, tq)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

# 主测试
print("="*70)
print("🚀 DQN 超简单微训练测试")
print("="*70)

# 加载数据
dfs = []
for t in ['ES=F', 'NQ=F', 'YM=F']:
    try:
        f = f'data/futures_processed/{t}.csv'
        if not __import__('os').path.exists(f):
            f = f'data/futures_processed/{t.replace("=", "")}.csv'
        df = pd.read_csv(f)
        df['Returns'] = df['Returns'].fillna(0)
        dfs.append(df)
        print(f"✅ {t}: {len(df)}行")
    except Exception as e:
        print(f"⚠️ {t}: {e}")

if not dfs:
    print("❌ 无数据")
    exit(1)

df_all = pd.concat(dfs, ignore_index=True)
prices = df_all['Close'].values
returns = df_all['Returns'].values

print(f"\n总样本：{len(returns):,}")
print(f"开始训练 5 episodes...\n")

env = Env(prices, returns)
dqn = DQN()

rewards = []
for ep in range(5):
    state = env.reset()
    total = 0
    for _ in range(500):
        a = dqn.get_action(state) - 1  # -1, 0, 1
        s_, r, done = env.step(a)
        dqn.store(state, a + 1, r, s_, float(done))
        dqn.train()
        total += r
        state = s_
        if done:
            break
    rewards.append(total)
    avg = np.mean(rewards[-3:]) if len(rewards) >= 3 else np.mean(rewards)
    print(f"Episode {ep+1}/5: Reward={total:.4f}, Avg={avg:.4f}")

print(f"\n✅ DQN 微训练通过！平均奖励：{np.mean(rewards):.4f}")
print("="*70)
