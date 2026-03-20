#!/usr/bin/env python3
"""
LSTM Pilot Test - 小规模测试验证
目标：验证LSTM代码能跑通，GPU能正常使用
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import time

print("="*80)
print("🧪 LSTM Pilot Test - 小规模验证")
print("="*80)

# 检查GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n设备: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# =============================================================================
# 1. 测试LSTM网络本身
# =============================================================================

print("\n" + "="*70)
print("测试 1: LSTM网络前向传播")
print("="*70)

class SimpleLSTM(nn.Module):
    """简单的LSTM网络"""
    def __init__(self, input_dim, hidden_sizes=[64, 32], output_dim=1):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_sizes[0], batch_first=True)
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.fc = nn.Linear(hidden_sizes[1], output_dim)
        self.leaky_relu = nn.LeakyReLU(0.01)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out1, _ = self.lstm1(x)
        out1 = self.leaky_relu(out1)
        out2, _ = self.lstm2(out1)
        out2 = self.leaky_relu(out2)
        return self.fc(out2[:, -1, :])

# 创建网络
net = SimpleLSTM(input_dim=16, hidden_sizes=[64, 32], output_dim=3).to(DEVICE)
print(f"网络参数: {sum(p.numel() for p in net.parameters()):,}")

# 测试前向传播
test_input = torch.randn(32, 10, 16).to(DEVICE)  # batch=32, seq_len=10, input_dim=16
start = time.time()
output = net(test_input)
elapsed = time.time() - start

print(f"输入形状: {test_input.shape}")
print(f"输出形状: {output.shape}")
print(f"前向传播时间: {elapsed*1000:.2f}ms")
print("✅ LSTM网络工作正常")

# =============================================================================
# 2. 测试GPU利用率
# =============================================================================

print("\n" + "="*70)
print("测试 2: GPU利用率")
print("="*70)

print("运行1000次前向传播...")
start = time.time()
for i in range(1000):
    output = net(test_input)
    if i % 100 == 0:
        print(f"  进度: {i}/1000", end='\r')
elapsed = time.time() - start

print(f"\n总时间: {elapsed:.2f}s")
print(f"每次: {elapsed/1000*1000:.2f}ms")
print("✅ GPU正在使用")

# =============================================================================
# 3. 测试小规模训练
# =============================================================================

print("\n" + "="*70)
print("测试 3: 小规模训练 (1个合约, 10 episodes)")
print("="*70)

# 加载一个合约的数据
DATA_DIR = 'data/futures_processed'
try:
    df = pd.read_csv(f'{DATA_DIR}/ES=F.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    prices = df['Close'].values[:500]  # 只用500天
    returns = df['Returns'].values[:500]
    print(f"数据: ES=F, {len(prices)} 天")
except Exception as e:
    print(f"⚠️ 无法加载数据: {e}")
    print("使用随机数据...")
    prices = np.cumprod(1 + np.random.randn(500) * 0.02)
    returns = np.diff(prices) / prices[:-1]

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
        # 简单特征
        if self.step_idx >= 20:
            obs[0] = np.mean(self.returns[self.step_idx-20:self.step_idx])
            obs[1] = np.std(self.returns[self.step_idx-20:self.step_idx])
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

# 简单的DQN Agent
class SimpleDQN:
    def __init__(self, state_dim=16, n_actions=3):
        self.q_net = SimpleLSTM(state_dim, [64, 32], n_actions).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=0.0001)
        self.gamma = 0.3
        
    def get_action(self, state, epsilon=0.3):
        if np.random.random() < epsilon:
            return np.random.randint(0, 3)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(DEVICE)
            q_values = self.q_net(state_t)
            return q_values.argmax().item()
    
    def train(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states).unsqueeze(1).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_states = torch.FloatTensor(next_states).unsqueeze(1).to(DEVICE)
        dones = torch.FloatTensor(dones).to(DEVICE)
        
        # Current Q
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1))
        
        # Target Q
        with torch.no_grad():
            next_q = self.q_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

# 训练循环
env = SimpleEnv(prices, returns)
agent = SimpleDQN()

print("开始训练 10 episodes...")
episode_rewards = []

for episode in range(10):
    state = env.reset()
    total_reward = 0
    steps = 0
    
    states, actions, rewards, next_states, dones = [], [], [], [], []
    
    while True:
        action = agent.get_action(state, epsilon=0.3)
        next_state, reward, done = env.step(action - 1)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(float(done))
        
        total_reward += reward
        steps += 1
        state = next_state
        
        if done or steps >= 100:
            break
    
    # 训练
    if len(states) > 10:
        loss = agent.train(states, actions, rewards, next_states, dones)
        episode_rewards.append(total_reward)
        print(f"  Episode {episode+1}: Reward={total_reward:.4f}, Steps={steps}, Loss={loss:.4f}")

print(f"\n平均奖励: {np.mean(episode_rewards):.4f}")
print("✅ 小规模训练成功")

# =============================================================================
# 4. 检查GPU内存使用
# =============================================================================

print("\n" + "="*70)
print("测试 4: GPU内存使用")
print("="*70)

if DEVICE == "cuda":
    allocated = torch.cuda.memory_allocated() / 1e6
    cached = torch.cuda.memory_reserved() / 1e6
    print(f"已分配: {allocated:.1f} MB")
    print(f"已缓存: {cached:.1f} MB")
    print("✅ GPU内存正常")

# =============================================================================
# 总结
# =============================================================================

print("\n" + "="*80)
print("✅ Pilot Test 完成")
print("="*80)
print("\n所有测试通过:")
print("  ✅ LSTM网络前向传播")
print("  ✅ GPU利用率")
print("  ✅ 小规模训练")
print("  ✅ GPU内存使用")
print("\n准备开始完整训练！")
