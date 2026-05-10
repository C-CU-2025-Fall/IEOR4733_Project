#!/usr/bin/env python3
"""
Paper-aligned DQN training - full implementation of all paper stability mechanisms

✅ Fixed Q-targets (Target Network)
✅ Double DQN
✅ Gradient Clipping
✅ Learning Rate Decay
✅ LSTM [64, 32] + Leaky-ReLU
✅ All Table 1 hyperparameters
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from indicators import FeatureEngineer, compute_volatility
from datetime import datetime
import pickle
import time
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# Paper Table 1 hyperparameters
# =============================================================================

LR = 0.0001          # Paper: 0.0001
GAMMA = 0.3          # Paper: 0.3
BATCH_SIZE = 64      # Paper: 64
MEMORY_SIZE = 5000   # Paper: 5000
TAU = 1000           # Paper: target update every 1000 steps
BP = 0.0020          # Paper: 20 bps
VOL_TARGET = 0.10    # 10% annualized volatility target

CONTRACTS_BY_CLASS = {
    'Commodity': ['CL=F', 'GC=F', 'SI=F', 'HG=F', 'NG=F', 'ZC=F', 'ZS=F', 'ZW=F', 
                  'KC=F', 'CC=F', 'SB=F', 'CT=F', 'OJ=F'],
    'Equity Index': ['ES=F', 'NQ=F', 'YM=F'],
    'Fixed Income': ['ZN=F', 'ZB=F', 'ZF=F', 'ZT=F', 'GE=F'],
    'FX': ['6E=F', '6J=F', '6B=F', '6A=F', '6C=F', '6S=F', '6N=F', '6M=F', '6R=F']
}

# =============================================================================
# Environment
# =============================================================================

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
        cost = BP * abs(action - self.last_a) * vol_scale * self.prices[self.idx]
        
        if self.idx + 1 >= len(self.returns):
            return self.fe.build_features(self.prices, self.returns, self.idx), 0.0, True
        
        reward = (action * vol_scale) * self.returns[self.idx + 1] - cost
        self.idx += 1
        self.last_a = action
        return self.fe.build_features(self.prices, self.returns, self.idx), reward, False

# =============================================================================
# LSTM network (Paper: LSTM [64, 32] + Leaky-ReLU)
# =============================================================================

class LSTM(nn.Module):
    def __init__(self, input_dim=8, hidden_sizes=[64, 32], output_dim=3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_sizes[0], batch_first=True)
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.fc = nn.Linear(hidden_sizes[1], output_dim)
        
        # Weight initialization (orthogonal init, suitable for LSTM)
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.orthogonal_(param, gain=nn.init.calculate_gain('tanh'))
            elif 'weight' in name:
                nn.init.orthogonal_(param, gain=0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
    def forward(self, x):
        o1, _ = self.lstm1(x)
        o1 = F.leaky_relu(o1, 0.01)  # Paper: Leaky-ReLU
        o2, _ = self.lstm2(o1)
        o2 = F.leaky_relu(o2, 0.01)
        return self.fc(o2[:, -1, :])

# =============================================================================
# Experience replay
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
# ✅ Paper-aligned DQN (Fixed Q-targets + Double DQN)
# =============================================================================

class DQN:
    def __init__(self):
        # Main network
        self.q_net = LSTM(8, [64, 32], 3).to(DEVICE)
        
        # ⭐ Target network (Paper: Fixed Q-targets)
        self.target_net = LSTM(8, [64, 32], 3).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=LR)
        
        # ⭐ Learning rate decay
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.9)
        
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.steps = 0
        
        # Record training history
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
        
        # ⭐ Double DQN (paper requirement)
        # Main network selects action, target network computes Q value
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1)  # Main network selects
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()  # Target network computes
            target_q = rewards + (1 - dones) * GAMMA * next_q
        
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # ⭐ Gradient clipping (prevent explosion)
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.5)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # ⭐ Update target network (Paper: τ=1000)
        self.steps += 1
        if self.steps % TAU == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.losses.append(loss.item())
        return loss.item()

# =============================================================================
# Data loading
# =============================================================================

def load_data(tickers):
    prices, returns = [], []
    for t in tickers:
        try:
            f = f'data/futures_processed/{t}.csv'
            if not os.path.exists(f):
                f = f'data/futures_processed/{t.replace("=", "")}.csv'
            df = pd.read_csv(f)
            df['Returns'] = df['Returns'].fillna(0)
            train = df[(df['Date'] >= '2011-01-01') & (df['Date'] <= '2015-12-31')]
            if len(train) > 500:
                prices.append(train['Close'].values)
                returns.append(train['Returns'].values)
        except:
            continue
    return np.concatenate(prices), np.concatenate(returns) if prices else (None, None)

# =============================================================================
# Training function
# =============================================================================

def train_class(name, tickers, episodes=200):
    print(f"\n{'='*70}")
    print(f"📊 Training {name} - DQN (Paper-aligned)")
    print('='*70)
    
    prices, returns = load_data(tickers)
    if prices is None:
        print("  ⚠️ No data")
        return None
    
    print(f"  Contracts: {len(tickers)}")
    print(f"  Total samples: {len(returns):,}")
    print(f"  Episodes: {episodes}")
    print(f"  Starting training...")
    
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
    print(f"  ✅ Done, avg reward: {np.mean(rewards):.4f}")
    return dqn

# =============================================================================
# Main function
# =============================================================================

def main():
    print("="*80)
    print("🔥 Paper-aligned DQN training - Fixed Q-targets + Double DQN")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Data: 2011-2015")
    print(f"Hyperparameters: lr={LR}, γ={GAMMA}, batch={BATCH_SIZE}, τ={TAU}")
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
    print(f"✅ Training complete!")
    print(f"⏱️ Total time: {elapsed:.1f} minutes")
    print(f"💾 Model: models_dqn_paper_{ts}.pkl")
    print("="*80)

if __name__ == '__main__':
    main()
