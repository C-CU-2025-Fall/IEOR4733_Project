#!/usr/bin/env python3
"""
论文对齐 DQN 训练 - 完整实现论文所有稳定机制

✅ Fixed Q-targets (Target Network)
✅ Double DQN
✅ Gradient Clipping
✅ Learning Rate Decay
✅ LSTM [64, 32] + Leaky-ReLU
✅ Table 1 所有超参数
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..indicators import FeatureEngineer, compute_volatility
from datetime import datetime
import pickle
import time
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# 论文 Table 1 超参数
# =============================================================================

LR = 0.0001          # 论文：0.0001
GAMMA = 0.3          # 论文：0.3
BATCH_SIZE = 64      # 论文：64
MEMORY_SIZE = 5000   # 论文：5000
TAU = 1000           # 论文：target update 每 1000 步
BP = 0.0020          # 论文：20 bps
VOL_TARGET = 0.10    # 10% 年化波动率目标

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

class Env:
    def __init__(self, prices, returns):
        self.prices = prices
        self.returns = returns
        self.n = len(returns)
        self.t = 0
        self.position = 0
        self.wealth = 1.0
        self.history = []
        self.feature_eng = FeatureEngineer()
        
    def reset(self):
        self.t = max(100, len(self.returns) // 10)
        self.position = 0
        self.wealth = 1.0
        self.history = []
        return self._get_state()
    
    def _get_state(self):
        if self.t < 100:
            return np.zeros(8, dtype=np.float32)
        ret_window = self.returns[self.t-100:self.t]
        features = self.feature_eng.compute_features(ret_window)
        return features[:8]
    
    def step(self, action):
        # action: -1 (short), 0 (hold), +1 (long)
        if self.t >= self.n - 1:
            return self._get_state(), 0, True
        
        pnl = self.position * self.returns[self.t+1] - BP * abs(action - self.position)
        self.wealth *= (1 + pnl)
        self.position = action
        self.t += 1
        reward = pnl
        
        done = self.t >= self.n - 1
        return self._get_state(), reward, done

# =============================================================================
# LSTM 网络
# =============================================================================

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        
        layers = []
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.LeakyReLU(0.01))
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.mlp(lstm_out[:, -1, :])

# =============================================================================
# 经验回放
# =============================================================================

class ReplayBuffer:
    def __init__(self, capacity=MEMORY_SIZE):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, s, a, r, s_, d):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (s, a, r, s_, d)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size=BATCH_SIZE):
        import random
        batch = random.sample(self.buffer, batch_size)
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# =============================================================================
# ✅ 论文对齐 DQN (Fixed Q-targets + Double DQN)
# =============================================================================

class DQN:
    def __init__(self):
        # 主网络
        self.q_net = LSTM(8, [64, 32], 3).to(DEVICE)
        
        # ⭐ 目标网络 (论文：Fixed Q-targets)
        self.target_net = LSTM(8, [64, 32], 3).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=LR)
        
        # ⭐ 学习率衰减
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.9)
        
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.steps = 0
        
        # 记录训练历史
        self.rewards = []
        self.losses = []
        
    def get_action(self, state, eps=0.3):
        if np.random.random() < eps:
            return np.random.randint(0, 3)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            return self.q_net(s).argmax().item()
    
    def store(self, s, a, r, s_, d):
        self.memory.push(s, a, r, s_, d)
    
    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return 0
        
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).to(DEVICE)
        
        # ⭐ Double DQN (论文要求)
        # 主网络选动作，目标网络算 Q 值
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1)  # 主网络选
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()  # 目标网络算
            target_q = rewards + (1 - dones) * GAMMA * next_q
        
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # ⭐ 梯度裁剪 (防止爆炸)
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.5)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # ⭐ 更新目标网络 (论文：τ=1000)
        self.steps += 1
        if self.steps % TAU == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.losses.append(loss.item())
        return loss.item()

# =============================================================================
# 数据加载
# =============================================================================

def load_data(tickers):
    prices, returns = [], []
    for t in tickers:
        try:
            # 尝试从 config/TEMP 加载数据
            f = f'config/TEMP/{t}_CLC.ASC'
            if not os.path.exists(f):
                f = f'data/CLC/{t}_CLC.csv'
            df = pd.read_csv(f) if f.endswith('.csv') else pd.read_csv(f, sep='\t')
            
            if 'Close' not in df.columns:
                df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            
            train = df[(df['Date'] >= '2011-01-01') & (df['Date'] <= '2015-12-31')]
            if len(train) > 500:
                prices.append(train['Close'].values)
                returns.append(np.diff(np.log(train['Close'].values)))
        except:
            continue
    
    if not prices:
        return None, None
    return np.concatenate(prices), np.concatenate(returns)

# =============================================================================
# 训练函数
# =============================================================================

def train_class(name, tickers, episodes=200):
    print(f"\n{'='*70}")
    print(f"📊 训练 {name} - DQN (论文对齐版)")
    print('='*70)
    
    prices, returns = load_data(tickers)
    if prices is None:
        print("  ⚠️ 无数据")
        return None
    
    print(f"  合约数：{len(tickers)}")
    print(f"  总样本：{len(returns):,}")
    print(f"  Episodes: {episodes}")
    print(f"  开始训练...")
    
    env = Env(prices, returns)
    dqn = DQN()
    rewards = []
    
    for ep in range(episodes):
        state = env.reset()
        total = 0
        for _ in range(500):
            a = dqn.get_action(state) - 1
            s_, r, done = env.step(a)
            dqn.store(state, a+1, r, s_, float(done))
            dqn.train()
            total += r
            state = s_
            if done:
                break
        rewards.append(total)
        
        if (ep+1) % 50 == 0:
            avg = np.mean(rewards[-50:])
            print(f"    Episode {ep+1}/{episodes}: Avg Reward={avg:.4f}")
    
    dqn.rewards = rewards
    print(f"  ✅ 完成，平均奖励：{np.mean(rewards):.4f}")
    return dqn

# =============================================================================
# 主函数
# =============================================================================

def main():
    print("="*80)
    print("🔥 论文对齐 DQN 训练 - Fixed Q-targets + Double DQN")
    print("="*80)
    print(f"设备：{DEVICE}")
    print(f"数据：2011-2015")
    print(f"超参数：lr={LR}, γ={GAMMA}, batch={BATCH_SIZE}, τ={TAU}")
    print("="*80)
    
    start = time.time()
    models = {}
    
    for name, tickers in CONTRACTS_BY_CLASS.items():
        models[name] = train_class(name, tickers)
    
    elapsed = (time.time() - start) / 60
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with open(f'models_dqn_paper_{ts}.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    print(f"\n{'='*80}")
    print(f"✅ 训练完成！")
    print(f"⏱️ 总时间：{elapsed:.1f} 分钟")
    print(f"💾 模型：models_dqn_paper_{ts}.pkl")
    print("="*80)

if __name__ == "__main__":
    main()
