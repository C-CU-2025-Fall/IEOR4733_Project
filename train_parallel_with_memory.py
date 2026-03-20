#!/usr/bin/env python3
"""
并行 DQN 训练 + 内存估计

特性:
1. 4 个资产类别并行训练
2. 内存使用实时监控
3. 自动估计峰值内存
4. 预计完成时间
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from indicators import FeatureEngineer, compute_volatility
from datetime import datetime
import pickle
import time
import os
import psutil
import gc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 超参数
LR = 0.0001
GAMMA = 0.3
BATCH_SIZE = 64
MEMORY_SIZE = 5000
TAU = 1000
BP = 0.0020

# 训练配置
EPISODES = 50  # 快速测试用 50，完整用 200
PARALLEL_CLASSES = True  # 是否并行训练

CONTRACTS_BY_CLASS = {
    'Commodity': ['CL=F', 'GC=F', 'SI=F', 'HG=F', 'NG=F', 'ZC=F', 'ZS=F', 'ZW=F', 
                  'KC=F', 'CC=F', 'SB=F', 'CT=F', 'OJ=F'],
    'Equity Index': ['ES=F', 'NQ=F', 'YM=F'],
    'Fixed Income': ['ZN=F', 'ZB=F', 'ZF=F', 'ZT=F', 'GE=F'],
    'FX': ['6E=F', '6J=F', '6B=F', '6A=F', '6C=F', '6S=F', '6N=F', '6M=F', '6R=F']
}

# =============================================================================
# 内存监控
# =============================================================================

class MemoryMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.peak_memory = 0
        self.samples = []
        
    def sample(self):
        mem_mb = self.process.memory_info().rss / 1024 / 1024
        self.samples.append(mem_mb)
        self.peak_memory = max(self.peak_memory, mem_mb)
        return mem_mb
    
    def report(self):
        if not self.samples:
            return {}
        return {
            'current': self.samples[-1],
            'peak': self.peak_memory,
            'avg': np.mean(self.samples),
            'min': np.min(self.samples)
        }

# =============================================================================
# 环境、网络、DQN (与之前相同)
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
        self.rewards = []
        
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
        
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + (1 - dones) * GAMMA * next_q
        
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.5)
        self.optimizer.step()
        
        self.steps += 1
        if self.steps % TAU == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        return loss.item()

# =============================================================================
# 数据加载
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
# 单个类别训练 (用于并行)
# =============================================================================

def train_single_class(args):
    """训练单个资产类别"""
    name, tickers, episodes = args
    
    prices, returns = load_data(tickers)
    if prices is None:
        return name, None
    
    env = Env(prices, returns)
    dqn = DQN()
    rewards = []
    
    start = time.time()
    
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
    
    elapsed = time.time() - start
    
    return name, {
        'model': dqn,
        'rewards': rewards,
        'time': elapsed,
        'avg_reward': np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
    }

# =============================================================================
# 主函数
# =============================================================================

def main():
    print("="*80)
    print("🚀 并行 DQN 训练 + 内存估计")
    print("="*80)
    print(f"设备：{DEVICE}")
    print(f"Episodes: {EPISODES}")
    print(f"并行：{'是' if PARALLEL_CLASSES else '否'}")
    print("="*80)
    
    monitor = MemoryMonitor()
    start_total = time.time()
    
    # 内存基线
    mem_baseline = monitor.sample()
    print(f"\n📊 内存基线：{mem_baseline:.1f} MB")
    
    if PARALLEL_CLASSES and DEVICE == 'cpu':
        # CPU 并行
        print(f"\n🔥 开始并行训练 (4 进程)...")
        
        with mp.Pool(processes=4) as pool:
            args_list = [(name, tickers, EPISODES) for name, tickers in CONTRACTS_BY_CLASS.items()]
            results = pool.map(train_single_class, args_list)
        
    else:
        # 串行训练 (GPU 或 不并行)
        print(f"\n🔥 开始串行训练...")
        results = []
        for name, tickers in CONTRACTS_BY_CLASS.items():
            mem_before = monitor.sample()
            result = train_single_class((name, tickers, EPISODES))
            mem_after = monitor.sample()
            print(f"  {name}: 内存 {mem_before:.0f} → {mem_after:.0f} MB")
            results.append(result)
    
    # 结果汇总
    models = {}
    for name, data in results:
        if data is not None:
            models[name] = data['model']
            print(f"\n✅ {name}: {data['time']:.1f}分钟, 平均奖励={data['avg_reward']:.4f}")
    
    elapsed_total = (time.time() - start_total) / 60
    mem_report = monitor.report()
    
    # 内存分析
    print(f"\n{'='*80}")
    print(f"📊 内存使用报告")
    print(f"{'='*80}")
    print(f"  基线：{mem_baseline:.1f} MB")
    print(f"  峰值：{mem_report['peak']:.1f} MB")
    print(f"  平均：{mem_report['avg']:.1f} MB")
    print(f"  增量：{mem_report['peak'] - mem_baseline:.1f} MB")
    
    # 推算完整训练 (200 episodes)
    scale = 200 / EPISODES
    print(f"\n{'='*80}")
    print(f"📈 完整训练估计 (200 episodes)")
    print(f"{'='*80}")
    print(f"  预计时间：{elapsed_total * scale:.1f} 分钟 ({elapsed_total * scale / 60:.1f} 小时)")
    print(f"  预计峰值内存：{mem_report['peak']:.0f} MB (不变，因为模型大小固定)")
    
    if PARALLEL_CLASSES and DEVICE == 'cpu':
        print(f"\n  ⚡ 并行加速：4 进程 → 预计 {elapsed_total * scale / 4:.1f} 分钟")
    
    # 保存模型
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'models_dqn_parallel_{ts}.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    print(f"\n💾 模型：models_dqn_parallel_{ts}.pkl")
    print(f"{'='*80}")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
