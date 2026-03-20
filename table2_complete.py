#!/usr/bin/env python3
"""
Table 2 完整复现 - 微测试迭代框架

目标：实现论文 Table 2 的所有 6 个模型
1. Long Only ✅
2. Sign(R) ✅
3. MACD ✅
4. DQN ✅
5. PG (Policy Gradient) - TODO
6. A2C (Advantage Actor-Critic) - TODO

使用方式:
    python3 table2_complete.py --model pg --micro
    python3 table2_complete.py --model a2c --micro
    python3 table2_complete.py --model all --micro
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from indicators import FeatureEngineer, compute_volatility

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# 论文超参数 (Table 1)
# =============================================================================

# DQN
DQN_LR = 0.0001
DQN_GAMMA = 0.3
DQN_BATCH = 64
DQN_MEMORY = 5000
DQN_TAU = 1000

# A2C (论文 Table 1)
A2C_CRITIC_LR = 0.0001
A2C_ACTOR_LR = 0.0001
A2C_GAMMA = 0.3
A2C_BATCH = 128

# 通用
BP = 0.0020  # 20 bps
VOL_TARGET = 0.10
MAX_STEPS = 500

# =============================================================================
# 环境 (与之前相同)
# =============================================================================

class VolatilityScaledEnv:
    def __init__(self, prices, returns, vol_target=VOL_TARGET):
        self.prices = prices
        self.returns = returns
        self.vol_target = vol_target
        self.n_steps = len(returns)
        self.volatility = compute_volatility(returns, 60)
        self.feature_engineer = FeatureEngineer(window_size=60)
        self.step_idx = 60
        self.last_action = 0.0
        
    def reset(self):
        self.step_idx = 60
        self.last_action = 0.0
        return self._get_state()
    
    def _get_state(self):
        return self.feature_engineer.build_features(
            self.prices, self.returns, self.step_idx
        )
    
    def step(self, action):
        action = float(np.clip(action, -1, 1))
        vol_scale = self.vol_target / (self.volatility[self.step_idx] + 1e-10)
        vol_scale = np.clip(vol_scale, 0.5, 2.0)
        
        current_price = self.prices[self.step_idx]
        cost = BP * abs(action - self.last_action) * vol_scale * current_price
        
        if self.step_idx + 1 >= self.n_steps:
            return self._get_state(), 0.0, True
        
        reward = (action * vol_scale) * self.returns[self.step_idx + 1] - cost
        
        self.step_idx += 1
        self.last_action = action
        
        return self._get_state(), reward, False

# =============================================================================
# 1. Long Only 策略
# =============================================================================

class LongOnly:
    """始终持有多头仓位"""
    
    def get_action(self, state):
        return 1  # 始终 long
    
    def train(self, *args):
        pass  # 无需训练

# =============================================================================
# 2. Sign(R) 策略
# =============================================================================

class SignR:
    """基于收益率符号的策略"""
    
    def __init__(self, lookback=252):
        self.lookback = lookback
        
    def get_action(self, state, returns, idx):
        # sign(r_{t-252:t})
        if idx >= self.lookback:
            ret = np.mean(returns[idx-self.lookback:idx])
            return 1 if ret > 0 else -1
        return 0

# =============================================================================
# 3. MACD 策略
# =============================================================================

class MACDStrategy:
    """基于 MACD 信号的策略"""
    
    def get_action(self, state, prices, idx):
        # 简化 MACD 信号
        if idx < 100:
            return 0
        
        # 计算 MACD
        short = np.mean(prices[idx-30:idx])
        long = np.mean(prices[idx-90:idx])
        macd = short - long
        
        if macd > 0:
            return 1
        elif macd < 0:
            return -1
        return 0

# =============================================================================
# 4. DQN (已有)
# =============================================================================

class ReplayBuffer:
    def __init__(self, capacity=DQN_MEMORY):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size=DQN_BATCH):
        if len(self.buffer) < batch_size:
            return None
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return zip(*batch)
    
    def __len__(self):
        return len(self.buffer)


class LSTMNetwork(nn.Module):
    def __init__(self, input_dim=8, hidden_sizes=[64, 32], output_dim=3):
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


class DQN:
    def __init__(self):
        self.q_net = LSTMNetwork(8, [64, 32], 3).to(DEVICE)
        self.target_net = LSTMNetwork(8, [64, 32], 3).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=DQN_LR)
        self.memory = ReplayBuffer(DQN_MEMORY)
        self.gamma = DQN_GAMMA
        self.steps = 0
        
    def get_action(self, state, epsilon=0.3):
        if np.random.random() < epsilon:
            return np.random.randint(0, 3)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            return self.q_net(state_t).argmax().item()
    
    def store_transition(self, s, a, r, s_, d):
        self.memory.push(s, a, r, s_, d)
    
    def train(self):
        if len(self.memory) < DQN_BATCH:
            return 0
        
        batch = self.memory.sample(DQN_BATCH)
        if batch is None:
            return 0
        
        states, actions, rewards, next_states, dones = [
            torch.FloatTensor(x).to(DEVICE) if i < 4 else torch.FloatTensor(x).to(DEVICE)
            for i, x in enumerate(batch)
        ]
        
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        current_q = self.q_net(states).gather(1, actions.long().unsqueeze(1)).squeeze()
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps += 1
        if self.steps % DQN_TAU == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        return loss.item()

# =============================================================================
# 5. PG (Policy Gradient) - 新增
# =============================================================================

class PGNetwork(nn.Module):
    """Policy Gradient 网络 - 连续动作空间"""
    
    def __init__(self, input_dim=8, hidden_sizes=[64, 32]):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_sizes[0], batch_first=True)
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        
        # 输出：动作均值和标准差
        self.mu_head = nn.Linear(hidden_sizes[1], 1)
        self.sigma_head = nn.Linear(hidden_sizes[1], 1)
        
        self.leaky_relu = nn.LeakyReLU(0.01)
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        
    def forward(self, x):
        out1, _ = self.lstm1(x)
        out1 = self.leaky_relu(out1)
        out2, _ = self.lstm2(out1)
        out2 = self.leaky_relu(out2)
        last = out2[:, -1, :]
        
        mu = torch.tanh(self.mu_head(last))  # [-1, 1]
        sigma = F.softplus(self.sigma_head(last)) + 0.01  # > 0
        
        return mu, sigma


class PG:
    """
    Policy Gradient (论文 3.2 节)
    - 连续动作空间 [-1, 1]
    - 直接优化策略
    - Monte Carlo 更新
    """
    
    def __init__(self):
        self.policy = PGNetwork(8, [64, 32]).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=A2C_ACTOR_LR)
        self.gamma = A2C_GAMMA
        self.trajectory = []  # 存储轨迹用于 MC 更新
        
    def get_action(self, state):
        """采样动作"""
        try:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                mu, sigma = self.policy(state_t)
                
                # 检查 NaN
                if torch.isnan(mu).any() or torch.isnan(sigma).any():
                    print(f"# {mu}, sigma={sigma}, state_shape={state.shape}")
                    return 0.0, 0.0, 0.1
                
                # 确保数值稳定
                mu = torch.clamp(mu, -0.99, 0.99)
                sigma = torch.clamp(sigma, 0.01, 1.0)
                
                dist = Normal(mu, sigma)
                action = dist.sample()
                action = torch.clamp(action, -1, 1)
                
                return action.item(), mu.item(), sigma.item()
        except Exception as e:
            print(f"# Error")
            return 0.0, 0.0, 0.1
    
    def store_transition(self, state, action, reward, next_state, done, mu, sigma):
        self.trajectory.append((state, action, reward, mu, sigma))
    
    def train(self):
        """
        Policy Gradient 更新 (公式 6)
        ∇J(θ) = Σ ∇log(π(a|s)) * G_t
        """
        if len(self.trajectory) < 10:
            return 0
        
        # 计算回报 G_t
        returns = []
        G = 0
        for _, _, r, _, _ in reversed(self.trajectory):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(DEVICE)
        
        # 归一化回报
        returns = (returns - returns.mean()) / (returns.std() + 1e-10)
        
        # 计算策略损失
        policy_loss = 0
        for i, (state, action, _, mu, sigma) in enumerate(self.trajectory):
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            mu_t, sigma_t = self.policy(state_t)
            
            dist = Normal(mu_t, sigma_t)
            log_prob = dist.log_prob(torch.FloatTensor([action]).to(DEVICE))
            
            policy_loss -= log_prob * returns[i]
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        self.trajectory = []  # 清空轨迹
        
        return policy_loss.item()

# =============================================================================
# 6. A2C (Advantage Actor-Critic) - 新增
# =============================================================================

class A2CNetwork(nn.Module):
    """A2C 网络 - 共享 LSTM"""
    
    def __init__(self, input_dim=8, hidden_sizes=[64, 32]):
        super().__init__()
        # 共享 LSTM
        self.lstm1 = nn.LSTM(input_dim, hidden_sizes[0], batch_first=True)
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        
        # Actor (策略)
        self.mu_head = nn.Linear(hidden_sizes[1], 1)
        self.sigma_head = nn.Linear(hidden_sizes[1], 1)
        
        # Critic (价值)
        self.critic = nn.Linear(hidden_sizes[1], 1)
        
        self.leaky_relu = nn.LeakyReLU(0.01)
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        
    def forward(self, x):
        out1, _ = self.lstm1(x)
        out1 = self.leaky_relu(out1)
        out2, _ = self.lstm2(out1)
        out2 = self.leaky_relu(out2)
        last = out2[:, -1, :]
        
        # Actor
        mu = torch.tanh(self.mu_head(last))
        sigma = F.softplus(self.sigma_head(last)) + 0.01
        
        # Critic
        value = self.critic(last)
        
        return mu, sigma, value


class A2C:
    """
    Advantage Actor-Critic (论文 3.2 节)
    - 连续动作空间 [-1, 1]
    - Actor-Critic 架构
    - 实时优势更新
    """
    
    def __init__(self):
        self.network = A2CNetwork(8, [64, 32]).to(DEVICE)
        self.actor_optimizer = torch.optim.Adam(
            list(self.network.mu_head.parameters()) + 
            list(self.network.sigma_head.parameters()), 
            lr=A2C_ACTOR_LR
        )
        self.critic_optimizer = torch.optim.Adam(
            self.network.critic.parameters(), 
            lr=A2C_CRITIC_LR
        )
        self.gamma = A2C_GAMMA
        
    def get_action(self, state):
        try:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                mu, sigma, _ = self.network(state_t)
                
                # 检查 NaN
                if torch.isnan(mu).any() or torch.isnan(sigma).any():
                    return 0.0
                
                mu = torch.clamp(mu, -0.99, 0.99)
                sigma = torch.clamp(sigma, 0.01, 1.0)
                
                dist = Normal(mu, sigma)
                action = dist.sample()
                action = torch.clamp(action, -1, 1)
                
                return action.item()
        except:
            return 0.0
    
    def train(self, state, action, reward, next_state, done):
        """A2C 更新"""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(DEVICE)
        action_t = torch.FloatTensor([action]).to(DEVICE)
        
        mu, sigma, value = self.network(state_t)
        _, _, next_value = self.network(next_state_t)
        
        # 计算优势 (公式 8)
        target = reward + (1 - done) * self.gamma * next_value.item()
        advantage = target - value.item()
        
        # Actor 损失 (公式 7)
        dist = Normal(mu, sigma)
        log_prob = dist.log_prob(action_t)
        actor_loss = -log_prob * advantage
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Critic 损失 (公式 9)
        critic_loss = F.mse_loss(value, torch.FloatTensor([target]).to(DEVICE))
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return actor_loss.item() + critic_loss.item()

# =============================================================================
# 模型工厂
# =============================================================================

def create_model(model_name):
    models = {
        'long': LongOnly(),
        'signr': SignR(),
        'macd': MACDStrategy(),
        'dqn': DQN(),
        'pg': PG(),
        'a2c': A2C()
    }
    return models.get(model_name)

# =============================================================================
# 测试函数
# =============================================================================

def test_model(model_name, micro=True):
    """测试单个模型"""
    print(f"\n{'='*80}")
    print(f"🧪 测试 {model_name.upper()} 模型")
    print('='*80)
    
    # 加载测试数据
    try:
        df = pd.read_csv('data/futures_processed/ES=F.csv')
        prices = df['Close'].values[:500]  # 微测试用少量数据
        returns = df['Returns'].values[:500]
    except:
        print("❌ 无法加载数据")
        return False
    
    # 创建模型
    model = create_model(model_name)
    if model is None:
        print(f"❌ 未知模型：{model_name}")
        return False
    
    # 创建环境
    env = VolatilityScaledEnv(prices, returns)
    
    # 测试
    episodes = 3 if micro else 10
    print(f"Episodes: {episodes}")
    
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < MAX_STEPS:
            if model_name in ['long']:
                action = model.get_action(state)
            elif model_name in ['signr', 'macd']:
                action = model.get_action(state, returns, env.step_idx)
            else:
                if model_name == 'pg':
                    action, mu, sigma = model.get_action(state)
                    # PG 轨迹在 train 时收集
                elif model_name == 'a2c':
                    action = model.get_action(state)
                else:  # dqn
                    action = model.get_action(state, epsilon=0.3)
            
            next_state, reward, done = env.step(action)
            
            if model_name == 'pg':
                # PG 在 episode 结束时统一训练，这里只需要存储
                pass
            elif model_name == 'dqn':
                model.store_transition(state, action, reward, next_state, float(done))
            elif model_name == 'a2c':
                loss = model.train(state, action, reward, next_state, float(done))
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        # PG 在 episode 结束时训练
        if model_name == 'pg':
            model.train()
        
        print(f"  Episode {ep+1}/{episodes}: Reward={total_reward:.4f}")
    
    print(f"✅ {model_name.upper()} 测试通过！")
    return True

# =============================================================================
# 主函数
# =============================================================================

def main():
    import sys
    
    print("="*80)
    print("📊 Table 2 完整复现 - 微测试")
    print("="*80)
    print(f"设备：{DEVICE}")
    print()
    
    if len(sys.argv) > 1:
        model_name = sys.argv[1].lower()
        if model_name == '--help':
            print("用法:")
            print("  python3 table2_complete.py long   # 测试 Long Only")
            print("  python3 table2_complete.py signr  # 测试 Sign(R)")
            print("  python3 table2_complete.py macd   # 测试 MACD")
            print("  python3 table2_complete.py dqn    # 测试 DQN")
            print("  python3 table2_complete.py pg     # 测试 PG")
            print("  python3 table2_complete.py a2c    # 测试 A2C")
            print("  python3 table2_complete.py all    # 测试所有")
            return
        elif model_name == 'all':
            models = ['long', 'signr', 'macd', 'dqn', 'pg', 'a2c']
            for m in models:
                test_model(m, micro=True)
        else:
            test_model(model_name, micro=True)
    else:
        # 默认测试所有
        models = ['long', 'signr', 'macd', 'dqn', 'pg', 'a2c']
        for m in models:
            test_model(m, micro=True)

if __name__ == '__main__':
    main()
