#!/usr/bin/env python3
"""
安全并行训练 - 基于内存测试结果

使用方式:
1. 先运行 test_memory_single.py 获取内存数据
2. 根据建议运行此脚本

python3 train_parallel_safe.py --parallel 4  # 根据内存测试结果设置
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
import argparse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 超参数
LR = 0.0001
GAMMA = 0.3
BATCH_SIZE = 64
MEMORY_SIZE = 5000
TAU = 1000
BP = 0.0020

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
        
    def sample(self):
        mem_mb = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, mem_mb)
        return mem_mb
    
    def report(self):
        return {
            'current': self.sample(),
            'peak': self.peak_memory
        }

# =============================================================================
# 环境、网络、DQN (简化版)
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
# 单个类别训练
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
# 分批训练 (安全并行)
# =============================================================================

def train_in_batches(classes_to_train, episodes, batch_size, monitor):
    """分批训练，每批后清理内存"""
    
    all_results = {}
    total_start = time.time()
    
    for i in range(0, len(classes_to_train), batch_size):
        batch = classes_to_train[i:i+batch_size]
        
        print(f"\n{'='*80}")
        print(f"📦 第 {i//batch_size + 1} 批：{[name for name, _ in batch]}")
        print(f"{'='*80}")
        
        mem_before = monitor.sample()
        print(f"  批次前内存：{mem_before:.0f} MB")
        
        # 本批并行
        if len(batch) > 1 and DEVICE == 'cpu':
            with mp.Pool(processes=len(batch)) as pool:
                args_list = [(name, tickers, episodes) for name, tickers in batch]
                results = pool.map(train_single_class, args_list)
        else:
            results = [train_single_class((name, tickers, episodes)) for name, tickers in batch]
        
        # 保存结果
        for name, data in results:
            if data:
                all_results[name] = data
                print(f"\n  ✅ {name}: {data['time']:.1f}分钟, 平均奖励={data['avg_reward']:.4f}")
        
        # 批次后清理
        gc.collect()
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        
        mem_after = monitor.sample()
        print(f"  批次后内存：{mem_after:.0f} MB")
        
        # 如果不是最后一批，等待一下
        if i + batch_size < len(classes_to_train):
            print(f"  ⏳ 等待 10 秒让内存稳定...")
            time.sleep(10)
    
    total_elapsed = (time.time() - total_start) / 60
    return all_results, total_elapsed

# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=50, help='Episodes per class')
    parser.add_argument('--parallel', type=int, default=None, help='Number of parallel processes')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size for safe training')
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 安全并行训练")
    print("="*80)
    print(f"设备：{DEVICE}")
    print(f"Episodes: {args.episodes}")
    
    # 获取系统内存
    system = psutil.virtual_memory()
    available_gb = system.available / 1024 / 1024 / 1024
    
    # 确定并行策略
    if args.parallel:
        parallel = args.parallel
        batch_size = args.batch_size or parallel
    else:
        # 自动判断
        if available_gb > 8:
            parallel = 4
            batch_size = 4
        elif available_gb > 4:
            parallel = 2
            batch_size = 2
        else:
            parallel = 1
            batch_size = 1
    
    print(f"并行进程数：{parallel}")
    print(f"批次大小：{batch_size}")
    print(f"系统可用内存：{available_gb:.1f} GB")
    print("="*80)
    
    monitor = MemoryMonitor()
    
    # 准备训练任务
    classes_to_train = list(CONTRACTS_BY_CLASS.items())
    
    # 分批训练
    models_dict, total_time = train_in_batches(
        classes_to_train, 
        args.episodes, 
        batch_size,
        monitor
    )
    
    # 汇总报告
    print(f"\n{'='*80}")
    print(f"📊 训练完成报告")
    print(f"{'='*80}")
    
    for name, data in models_dict.items():
        print(f"  ✅ {name}: {data['time']:.1f}分钟")
    
    mem_report = monitor.report()
    print(f"\n📈 内存使用:")
    print(f"  峰值：{mem_report['peak']:.0f} MB ({mem_report['peak']/1024:.1f} GB)")
    print(f"\n⏱️  总时间：{total_time:.1f} 分钟")
    
    # 保存模型
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'models_dqn_safe_{ts}.pkl', 'wb') as f:
        pickle.dump(models_dict, f)
    
    print(f"\n💾 模型：models_dqn_safe_{ts}.pkl")
    print(f"{'='*80}")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
