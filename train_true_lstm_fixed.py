#!/usr/bin/env python3
"""
完整复现论文 - 真正的LSTM实现 (修复版)
使用自定义LSTM网络，不依赖stable-baselines3
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
import pickle
from collections import deque
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# 论文配置
# =============================================================================

DATA_DIR = 'data/futures_processed'
TRANSACTION_COST = 0.002

# 论文超参数 (Table 1)
GAMMA = 0.3
BUFFER_SIZE = 5000
BATCH_SIZE_DQN = 64
BATCH_SIZE_A2C = 128
LEARNING_RATE = 0.0001
TARGET_UPDATE = 1000
TOTAL_EPISODES = 500

# LSTM配置 (论文 Section 4.3)
LSTM_HIDDEN_SIZES = [64, 32]

# =============================================================================
# LSTM网络 (论文架构)
# =============================================================================

class LSTMNetwork(nn.Module):
    """论文LSTM: 两层 [64, 32] + LeakyReLU"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.lstm1 = nn.LSTM(input_dim, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.fc = nn.Linear(32, output_dim)
        
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        if hidden is None:
            h1 = torch.zeros(1, batch_size, 64).to(x.device)
            c1 = torch.zeros(1, batch_size, 64).to(x.device)
            h2 = torch.zeros(1, batch_size, 32).to(x.device)
            c2 = torch.zeros(1, batch_size, 32).to(x.device)
            hidden = ((h1, c1), (h2, c2))
        
        out1, (h1_new, c1_new) = self.lstm1(x, hidden[0])
        out1 = self.leaky_relu(out1)
        
        out2, (h2_new, c2_new) = self.lstm2(out1, hidden[1])
        out2 = self.leaky_relu(out2)
        
        output = self.fc(out2[:, -1, :])
        
        return output, ((h1_new, c1_new), (h2_new, c2_new))

# =============================================================================
# DQN Agent with LSTM
# =============================================================================

class LSTMDQNAgent:
    """DQN with LSTM"""
    def __init__(self, state_dim, n_actions=3):
        self.state_dim = state_dim
        self.n_actions = n_actions
        
        self.q_net = LSTMNetwork(state_dim, n_actions).to(DEVICE)
        self.target_net = LSTMNetwork(state_dim, n_actions).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.buffer = deque(maxlen=BUFFER_SIZE)
        
        self.steps = 0
        self.epsilon = 1.0
        
    def get_action(self, state, hidden=None, epsilon=None):
        if epsilon is None:
            epsilon = max(0.01, 1.0 - self.steps / 10000)
        
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1), hidden
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(DEVICE)
            q_values, new_hidden = self.q_net(state_tensor, hidden)
            action = q_values.argmax(dim=1).item()
            return action, new_hidden
    
    def remember(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def train(self):
        if len(self.buffer) < BATCH_SIZE_DQN:
            return
        
        batch = random.sample(self.buffer, BATCH_SIZE_DQN)
        
        states = torch.FloatTensor([e[0] for e in batch]).unsqueeze(1).to(DEVICE)
        actions = torch.LongTensor([e[1] for e in batch]).to(DEVICE)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(DEVICE)
        next_states = torch.FloatTensor([e[3] for e in batch]).unsqueeze(1).to(DEVICE)
        dones = torch.FloatTensor([e[4] for e in batch]).to(DEVICE)
        
        current_q, _ = self.q_net(states)
        current_q = current_q.gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q, _ = self.target_net(next_states)
            max_next_q = next_q.max(1)[0]
            target_q = rewards + (1 - dones) * GAMMA * max_next_q
        
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps += 1
        if self.steps % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

# =============================================================================
# A2C Agent with LSTM
# =============================================================================

class LSTMA2CAgent:
    """A2C with LSTM"""
    def __init__(self, state_dim):
        self.state_dim = state_dim
        
        # Actor: 输出 mean, log_std
        self.actor = LSTMNetwork(state_dim, 2).to(DEVICE)
        # Critic: 输出 value
        self.critic = LSTMNetwork(state_dim, 1).to(DEVICE)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)
        
    def get_action(self, state, hidden_actor=None, hidden_critic=None):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            action_params, new_hidden_actor = self.actor(state_tensor, hidden_actor)
            mean = action_params[:, 0]
            std = torch.exp(action_params[:, 1]) + 0.1
            
            dist = Normal(mean, std)
            action = dist.sample()
            action = torch.tanh(action)
            
            return action.item(), new_hidden_actor, new_hidden_critic
    
    def train_episode(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states).unsqueeze(1).to(DEVICE)
        actions = torch.FloatTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_states = torch.FloatTensor(next_states).unsqueeze(1).to(DEVICE)
        dones = torch.FloatTensor(dones).to(DEVICE)
        
        # 计算returns
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + GAMMA * R * (1 - d)
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(DEVICE)
        
        # Critic loss
        values, _ = self.critic(states)
        values = values.squeeze()
        critic_loss = F.mse_loss(values, returns)
        
        # Actor loss
        action_params, _ = self.actor(states)
        means = action_params[:, 0]
        log_stds = action_params[:, 1]
        stds = torch.exp(log_stds) + 0.1
        
        dist = Normal(means, stds)
        raw_actions = torch.atanh(torch.clamp(actions, -0.99, 0.99))
        log_probs = dist.log_prob(raw_actions)
        
        advantages = returns - values.detach()
        actor_loss = -(log_probs * advantages).mean()
        
        # 更新
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

# =============================================================================
# 环境
# =============================================================================

class TradingEnv:
    """交易环境"""
    def __init__(self, prices, returns):
        self.prices = prices
        self.returns = returns
        self.n_steps = len(returns)
        self.current_step = 200
        self.last_action = 0.0
        
    def reset(self):
        self.current_step = 200
        self.last_action = 0.0
        return self._get_obs()
    
    def _get_obs(self):
        obs = np.zeros(16, dtype=np.float32)
        
        for i, window in enumerate([5, 10, 25, 50, 100, 200]):
            if self.current_step >= window:
                ret = np.mean(self.returns[self.current_step-window:self.current_step])
                vol = np.std(self.returns[self.current_step-window:self.current_step])
                obs[i] = ret / (vol + 1e-8)
        
        if self.current_step >= 26:
            ma_fast = np.mean(self.prices[self.current_step-12:self.current_step])
            ma_slow = np.mean(self.prices[self.current_step-26:self.current_step])
            obs[6] = (ma_fast - ma_slow) / (np.std(self.prices[self.current_step-63:self.current_step]) + 1e-8)
        
        if self.current_step >= 20:
            obs[9] = np.std(self.returns[self.current_step-20:self.current_step]) * np.sqrt(252)
        
        return obs
    
    def step(self, action):
        action = float(np.clip(action, -1, 1))
        cost = abs(action - self.last_action) * TRANSACTION_COST
        
        if self.current_step + 1 >= self.n_steps:
            return self._get_obs(), 0.0, True
        
        ret = self.returns[self.current_step + 1]
        strat_ret = action * ret - cost
        
        # Volatility scaling
        if self.current_step >= 60:
            current_vol = np.std(self.returns[self.current_step-60:self.current_step]) * np.sqrt(252)
            scale = 0.10 / (current_vol + 1e-8)
            strat_ret *= scale
        
        reward = strat_ret
        
        self.current_step += 1
        self.last_action = action
        
        return self._get_obs(), reward, False

# =============================================================================
# 训练函数
# =============================================================================

def train_dqn(env, n_episodes=TOTAL_EPISODES):
    """训练DQN"""
    agent = LSTMDQNAgent(state_dim=16, n_actions=3)
    
    for episode in range(n_episodes):
        state = env.reset()
        hidden = None
        total_reward = 0
        
        while True:
            action, hidden = agent.get_action(state, hidden)
            next_state, reward, done = env.step(action - 1)
            
            agent.remember(state, action, reward, next_state, float(done))
            agent.train()
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        if (episode + 1) % 100 == 0:
            print(f"    Episode {episode+1}/{n_episodes}, Reward: {total_reward:.2f}")
    
    return agent

def train_a2c(env, n_episodes=TOTAL_EPISODES):
    """训练A2C"""
    agent = LSTMA2CAgent(state_dim=16)
    
    for episode in range(n_episodes):
        state = env.reset()
        hidden_actor = None
        hidden_critic = None
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        while True:
            action, hidden_actor, hidden_critic = agent.get_action(state, hidden_actor, hidden_critic)
            next_state, reward, done = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(float(done))
            
            state = next_state
            
            if done:
                break
        
        agent.train_episode(states, actions, rewards, next_states, dones)
        
        if (episode + 1) % 100 == 0:
            total_reward = sum(rewards)
            print(f"    Episode {episode+1}/{n_episodes}, Reward: {total_reward:.2f}")
    
    return agent

# =============================================================================
# 主函数
# =============================================================================

def main():
    print("="*80)
    print("🔥 真正的LSTM实现 (自定义PyTorch)")
    print("="*80)
    print(f"设备: {DEVICE}")
    print(f"LSTM: {LSTM_HIDDEN_SIZES}")
    print(f"Episodes: {TOTAL_EPISODES}")
    print()
    
    CONTRACTS_BY_CLASS = {
        'Commodity': ['CL=F', 'GC=F', 'SI=F', 'HG=F', 'NG=F', 'ZC=F', 'ZS=F', 'ZW=F', 'KC=F', 'CC=F', 'SB=F', 'CT=F', 'OJ=F'],
        'Equity Index': ['ES=F', 'NQ=F', 'YM=F'],
        'Fixed Income': ['ZN=F', 'ZB=F', 'ZF=F', 'ZT=F', 'GE=F'],
        'FX': ['6E=F', '6J=F', '6B=F', '6A=F', '6C=F', '6S=F', '6N=F', '6M=F', '6R=F']
    }
    
    models = {}
    
    for asset_class, tickers in CONTRACTS_BY_CLASS.items():
        print(f"\n{'='*70}")
        print(f"📊 训练 {asset_class}")
        print('='*70)
        
        # 加载数据
        all_data = []
        for ticker in tickers:
            try:
                df = pd.read_csv(f'{DATA_DIR}/{ticker}.csv')
                df['Date'] = pd.to_datetime(df['Date'])
                train = df[(df['Date'] >= '2011-01-03') & (df['Date'] <= '2015-12-31')]
                if len(train) < 500:
                    continue
                all_data.append({
                    'prices': train['Close'].values,
                    'returns': train['Returns'].values
                })
            except:
                continue
        
        if not all_data:
            print("  ⚠️ 无数据")
            continue
        
        print(f"  合约数: {len(all_data)}")
        
        # 合并所有合约数据
        all_prices = np.concatenate([d['prices'] for d in all_data])
        all_returns = np.concatenate([d['returns'] for d in all_data])
        
        print(f"  总样本: {len(all_returns):,}")
        
        env = TradingEnv(all_prices, all_returns)
        
        print("  训练DQN (LSTM)...")
        dqn = train_dqn(env)
        
        print("  训练A2C (LSTM)...")
        a2c = train_a2c(env)
        
        models[asset_class] = {'dqn': dqn, 'a2c': a2c}
    
    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'models_true_lstm_{timestamp}.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    print(f"\n✅ 训练完成！")
    print(f"💾 模型已保存: models_true_lstm_{timestamp}.pkl")

if __name__ == '__main__':
    main()
