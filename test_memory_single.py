#!/usr/bin/env python3
"""
单类别内存测试 - 安全评估后再并行

步骤:
1. 测试单个类别的内存占用
2. 计算系统可用内存
3. 安全决定并行进程数
4. 给出并行建议
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from indicators import FeatureEngineer, compute_volatility
import psutil
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

# 测试配置
TEST_EPISODES = 10  # 10 episodes 足够估算内存

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
        self.baseline = self.sample()
        
    def sample(self):
        mem_mb = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, mem_mb)
        return mem_mb
    
    def get_increase(self):
        return self.peak_memory - self.baseline
    
    def get_system_info(self):
        mem = psutil.virtual_memory()
        return {
            'total': mem.total / 1024 / 1024 / 1024,  # GB
            'available': mem.available / 1024 / 1024 / 1024,
            'used': mem.used / 1024 / 1024 / 1024,
            'percent': mem.percent
        }

# =============================================================================
# 环境、网络、DQN
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
# 单类别内存测试
# =============================================================================

def test_single_class_memory(name, tickers, monitor):
    """测试单个类别的内存占用"""
    print(f"\n{'='*70}")
    print(f"📊 测试 {name}")
    print('='*70)
    
    # 加载数据前内存
    mem_before_load = monitor.sample()
    print(f"  加载前内存：{mem_before_load:.1f} MB")
    
    prices, returns = load_data(tickers)
    if prices is None:
        print("  ⚠️ 无数据")
        return None
    
    # 加载数据后内存
    mem_after_load = monitor.sample()
    print(f"  数据加载后：{mem_after_load:.1f} MB (+{mem_after_load - mem_before_load:.1f} MB)")
    
    # 创建环境
    env = Env(prices, returns)
    mem_after_env = monitor.sample()
    print(f"  环境创建后：{mem_after_env:.1f} MB (+{mem_after_env - mem_after_load:.1f} MB)")
    
    # 创建模型
    dqn = DQN()
    mem_after_model = monitor.sample()
    print(f"  模型创建后：{mem_after_model:.1f} MB (+{mem_after_model - mem_after_env:.1f} MB)")
    
    # 训练几个 episodes
    print(f"  开始训练 {TEST_EPISODES} episodes...")
    for ep in range(TEST_EPISODES):
        state = env.reset()
        for _ in range(500):
            a = dqn.get_action(state) - 1
            s_, r, done = env.step(a)
            dqn.store(state, a+1, r, s_, float(done))
            dqn.train()
            state = s_
            if done:
                break
    
    mem_after_train = monitor.sample()
    print(f"  训练后内存：{mem_after_train:.1f} MB")
    
    # 清理
    del dqn, env
    import gc
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    
    mem_after_cleanup = monitor.sample()
    print(f"  清理后内存：{mem_after_cleanup:.1f} MB")
    
    # 计算增量
    memory_increase = mem_after_train - mem_before_load
    
    return {
        'name': name,
        'baseline': mem_before_load,
        'after_load': mem_after_load,
        'after_model': mem_after_model,
        'peak': mem_after_train,
        'after_cleanup': mem_after_cleanup,
        'increase': memory_increase,
        'contracts': len(tickers),
        'samples': len(returns)
    }

# =============================================================================
# 并行建议
# =============================================================================

def recommend_parallel(results, monitor):
    """根据测试结果给出并行建议"""
    
    system = monitor.get_system_info()
    available_gb = system['available']
    
    # 找到最大内存占用
    if not results:
        return None
    
    max_increase = max(r['increase'] for r in results)
    avg_increase = np.mean([r['increase'] for r in results])
    
    # 计算安全并行数
    # 保留 2GB 给系统和其他进程
    safe_available = (available_gb * 1024) - 2000  # MB
    
    if max_increase > 0:
        safe_parallel = int(safe_available / max_increase)
    else:
        safe_parallel = 4
    
    # 不超过 4 个类别
    safe_parallel = min(safe_parallel, 4)
    safe_parallel = max(safe_parallel, 1)  # 至少 1 个
    
    return {
        'system_total_gb': system['total'],
        'system_available_gb': available_gb,
        'single_class_peak_mb': max_increase,
        'single_class_avg_mb': avg_increase,
        'safe_parallel': safe_parallel,
        'estimated_total_mb': max_increase * safe_parallel,
        'memory_headroom_mb': safe_available - (max_increase * safe_parallel)
    }

# =============================================================================
# 主函数
# =============================================================================

def main():
    print("="*80)
    print("🔍 单类别内存测试 + 并行建议")
    print("="*80)
    
    monitor = MemoryMonitor()
    system = monitor.get_system_info()
    
    print(f"\n📊 系统内存信息:")
    print(f"  总内存：{system['total']:.1f} GB")
    print(f"  可用：{system['available']:.1f} GB ({100-system['percent']:.0f}%)")
    print(f"  已用：{system['used']:.1f} GB ({system['percent']}%)")
    
    print(f"\n📈 进程基线内存：{monitor.baseline:.1f} MB")
    print(f"  测试配置：{TEST_EPISODES} episodes/类别")
    print()
    
    # 测试每个类别
    results = []
    for name, tickers in CONTRACTS_BY_CLASS.items():
        result = test_single_class_memory(name, tickers, monitor)
        if result:
            results.append(result)
    
    # 汇总报告
    print(f"\n{'='*80}")
    print(f"📊 内存测试汇总")
    print(f"{'='*80}")
    
    for r in results:
        print(f"\n  {r['name']}:")
        print(f"    合约数：{r['contracts']}, 样本数：{r['samples']:,}")
        print(f"    内存峰值：{r['peak']:.0f} MB (增量：+{r['increase']:.0f} MB)")
    
    # 并行建议
    rec = recommend_parallel(results, monitor)
    
    if rec:
        print(f"\n{'='*80}")
        print(f"💡 并行训练建议")
        print(f"{'='*80}")
        print(f"\n  系统可用内存：{rec['system_available_gb']:.1f} GB")
        print(f"  单类别峰值：{rec['single_class_peak_mb']:.0f} MB")
        print(f"  单类别平均：{rec['single_class_avg_mb']:.0f} MB")
        print()
        print(f"  ✅ 安全并行数：{rec['safe_parallel']} 进程")
        print(f"  预计总内存：{rec['estimated_total_mb']:.0f} MB ({rec['estimated_total_mb']/1024:.1f} GB)")
        print(f"  内存余量：{rec['memory_headroom_mb']:.0f} MB ({rec['memory_headroom_mb']/1024:.1f} GB)")
        print()
        
        if rec['safe_parallel'] >= 4:
            print(f"  🎉 可以安全并行训练所有 4 个类别！")
        elif rec['safe_parallel'] >= 2:
            print(f"  ⚡ 可以并行训练 {rec['safe_parallel']} 个类别")
            print(f"     建议分批：第一批 {rec['safe_parallel']} 个，第二批 {4-rec['safe_parallel']} 个")
        else:
            print(f"  ⚠️  内存紧张，建议串行训练")
            print(f"     预计每类别内存：{rec['single_class_peak_mb']:.0f} MB")
        
        print(f"\n{'='*80}")
        print(f"🚀 下一步:")
        print(f"{'='*80}")
        if rec['safe_parallel'] >= 4:
            print(f"  python3 train_parallel_with_memory.py  # 4 进程并行")
        else:
            print(f"  python3 train_dqn_paper_aligned.py  # 串行训练")
        print(f"{'='*80}")

if __name__ == '__main__':
    main()
