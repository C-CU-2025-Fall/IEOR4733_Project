#!/usr/bin/env python3
"""
快速收敛性测试 - 10-20 分钟判断训练是否可行

策略:
1. 只训练 30 episodes (~10-15 分钟)
2. 监控关键指标:
   - Q 值是否 NaN
   - 奖励趋势
   - 梯度范数
   - Loss 趋势
3. 提前判断收敛性
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from indicators import FeatureEngineer, compute_volatility
import os
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 超参数
LR = 0.0001
GAMMA = 0.3
BATCH_SIZE = 64
MEMORY_SIZE = 5000
TAU = 1000
BP = 0.0020

# 快速测试配置
TEST_EPISODES = 30  # 30 episodes ≈ 10-15 分钟
CONVERGENCE_THRESHOLD = 0.7  # 70% 的 episodes 需要改善

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

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(8, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.fc = nn.Linear(32, 3)
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
        o1, _ = self.lstm1(x); o1 = F.leaky_relu(o1, 0.01)
        o2, _ = self.lstm2(o1); o2 = F.leaky_relu(o2, 0.01)
        return self.fc(o2[:, -1, :])

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
        return (np.array([x[0] for x in batch]), np.array([x[1] for x in batch]),
                np.array([x[2] for x in batch]), np.array([x[3] for x in batch]),
                np.array([x[4] for x in batch]))
    
    def __len__(self):
        return len(self.buffer)

class DQN:
    def __init__(self):
        self.q_net = LSTM().to(DEVICE)
        self.target_net = LSTM().to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.steps = 0
        
        # 监控指标
        self.q_values = []
        self.grad_norms = []
        self.losses = []
        
    def get_action(self, state, eps=0.3):
        if np.random.random() < eps:
            return np.random.randint(0, 3)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q = self.q_net(s)
            self.q_values.append(q.cpu().numpy()[0])
            return q.argmax().item()
    
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
        
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + (1 - dones) * GAMMA * next_q
        
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度范数监控
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.5)
        self.grad_norms.append(grad_norm.item())
        self.losses.append(loss.item())
        
        self.optimizer.step()
        
        self.steps += 1
        if self.steps % TAU == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        return loss.item()

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

def check_convergence(rewards, q_values, grad_norms, losses):
    """判断是否收敛"""
    checks = {}
    
    # 1. Q 值检查
    nan_q = sum(1 for q in q_values if np.any(np.isnan(q)))
    checks['Q 值正常'] = nan_q < len(q_values) * 0.1  # <10% NaN
    
    # 2. 梯度检查
    avg_grad = np.mean(grad_norms[-100:]) if len(grad_norms) > 100 else np.mean(grad_norms)
    checks['梯度正常'] = avg_grad < 1.0  # 梯度范数 < 1
    
    # 3. Loss 检查
    if len(losses) > 50:
        first_loss = np.mean(losses[:50])
        last_loss = np.mean(losses[-50:])
        checks['Loss 下降'] = last_loss < first_loss * 1.5  # 没有显著上升
    else:
        checks['Loss 下降'] = True
    
    # 4. 奖励趋势
    if len(rewards) > 20:
        first_rewards = np.mean(rewards[:10])
        last_rewards = np.mean(rewards[-10:])
        checks['奖励稳定'] = last_rewards > first_rewards - 100  # 没有显著下降
    else:
        checks['奖励稳定'] = True
    
    # 综合判断
    passed = sum(checks.values())
    total = len(checks)
    
    return checks, passed / total >= 0.75  # 75% 检查通过

def quick_test(name, tickers):
    """快速收敛性测试"""
    print(f"\n{'='*70}")
    print(f"🧪 快速收敛测试 - {name}")
    print('='*70)
    
    prices, returns = load_data(tickers)
    if prices is None:
        print("  ⚠️ 无数据")
        return None
    
    print(f"  合约数：{len(tickers)}")
    print(f"  总样本：{len(returns):,}")
    print(f"  Episodes: {TEST_EPISODES}")
    print(f"  预计时间：~{TEST_EPISODES * 0.4:.0f} 分钟")
    print()
    
    start = time.time()
    
    env = Env(prices, returns)
    dqn = DQN()
    rewards = []
    
    for ep in range(TEST_EPISODES):
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
        
        if (ep+1) % 10 == 0:
            avg = np.mean(rewards[-10:])
            elapsed = (time.time() - start) / 60
            print(f"    Episode {ep+1}/{TEST_EPISODES}: Avg={avg:.4f}, 耗时={elapsed:.1f}分钟")
    
    elapsed = (time.time() - start) / 60
    
    # 收敛性检查
    checks, converged = check_convergence(rewards, dqn.q_values, dqn.grad_norms, dqn.losses)
    
    print(f"\n{'='*70}")
    print(f"📊 收敛性分析 ({elapsed:.1f} 分钟)")
    print('='*70)
    
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")
    
    print()
    if converged:
        print(f"✅ {name}: 收敛性良好，可以完整训练！")
        return {'converged': True, 'time': elapsed, 'avg_reward': np.mean(rewards[-10:])}
    else:
        print(f"❌ {name}: 收敛性问题，需要调整超参数！")
        return {'converged': False, 'time': elapsed, 'issues': [k for k,v in checks.items() if not v]}

def main():
    print("="*80)
    print("🚀 快速收敛性测试 - 10-20 分钟判断")
    print("="*80)
    print(f"设备：{DEVICE}")
    print(f"配置：{TEST_EPISODES} episodes/类别")
    print("="*80)
    
    # 只测试一个类别作为示例
    test_class = 'Equity Index'
    test_tickers = ['ES=F', 'NQ=F', 'YM=F']
    
    result = quick_test(test_class, test_tickers)
    
    if result and result['converged']:
        print("\n" + "="*80)
        print("✅ 测试通过！可以开始完整训练")
        print("="*80)
        print(f"\n预计完整训练时间 (200 episodes): {result['time'] * (200/TEST_EPISODES):.0f} 分钟")
        print(f"预计 4 个类别总时间：{result['time'] * (200/TEST_EPISODES) * 4 / 60:.1f} 小时")
    else:
        print("\n" + "="*80)
        print("❌ 测试失败，需要调整")
        print("="*80)

if __name__ == '__main__':
    main()
