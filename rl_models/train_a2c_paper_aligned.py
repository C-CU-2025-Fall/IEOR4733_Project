#!/usr/bin/env python3
"""
Advantage Actor-Critic (A2C) 训练 - 论文对齐实现

✅ Actor Network (策略网络)
✅ Critic Network (价值网络)
✅ 优势函数 A(s,a) = R + γV(s') - V(s)
✅ 同步实时更新
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
# 论文 Table 1 超参数 (A2C)
# =============================================================================

LR_CRITIC = 0.001     # 论文：0.001
LR_ACTOR = 0.0001     # 论文：0.0001
GAMMA = 0.3           # 论文：0.3
BATCH_SIZE = 128      # 论文：128
BP = 0.0020           # 论文：20 bps
VOL_TARGET = 0.10     # 10% 年化波动率目标

CONTRACTS_BY_CLASS = {
    'Commodity': ['CL', 'GC', 'SI', 'HG', 'NG', 'ZC', 'ZS', 'ZW', 'KC', 'CC', 'SB', 'CT', 'OJ'],
    'Equity Index': ['ES', 'NQ', 'YM'],
    'Fixed Income': ['ZN', 'ZB', 'ZF', 'ZT', 'GE'],
    'FX': ['6E', '6J', '6B', '6A', '6C', '6S', '6N', '6M', '6R']
}

# =============================================================================
# LSTM 网络 (与 DQN/PG 相同)
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
# 环境 (与 DQN/PG 相同)
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
# ✅ Advantage Actor-Critic (A2C)
# =============================================================================

class A2C:
    def __init__(self):
        # Actor 网络 (输出 3 个动作的概率)
        self.actor = LSTM(8, [64, 32], 3).to(DEVICE)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        
        # Critic 网络 (输出状态价值 V(s))
        self.critic = LSTM(8, [64, 32], 1).to(DEVICE)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
        # 学习率衰减
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optim, step_size=50, gamma=0.9)
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optim, step_size=50, gamma=0.9)
        
        # 训练历史
        self.rewards = []
        self.actor_losses = []
        self.critic_losses = []
        
        # 缓冲区 (用于小批量更新)
        self.states_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.next_states_buffer = []
        self.dones_buffer = []
    
    def get_action(self, state):
        """从 Actor 网络采样动作"""
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            logits = self.actor(s)
            probs = F.softmax(logits, dim=1)[0]
            action = torch.multinomial(probs, 1).item()
            return action - 1  # 转换为 -1, 0, 1
    
    def get_value(self, state):
        """从 Critic 网络获取状态价值"""
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            value = self.critic(s).item()
            return value
    
    def store_transition(self, s, a, r, s_, done):
        """存储转移到缓冲区"""
        self.states_buffer.append(s)
        self.actions_buffer.append(a)
        self.rewards_buffer.append(r)
        self.next_states_buffer.append(s_)
        self.dones_buffer.append(done)
    
    def train_batch(self):
        """
        实时更新 A2C: 同步更新 Actor 和 Critic
        优势函数：A(s,a) = R + γV(s') - V(s)  [论文 Eq. 8]
        """
        if len(self.states_buffer) == 0:
            return 0, 0
        
        states = torch.FloatTensor(np.array(self.states_buffer)).to(DEVICE)
        actions = torch.LongTensor(np.array(self.actions_buffer)).to(DEVICE)
        rewards = torch.FloatTensor(np.array(self.rewards_buffer)).to(DEVICE)
        next_states = torch.FloatTensor(np.array(self.next_states_buffer)).to(DEVICE)
        dones = torch.FloatTensor(np.array(self.dones_buffer)).to(DEVICE)
        
        # ========== Critic 更新 ==========
        # V(s) 目标：R + γV(s')
        with torch.no_grad():
            next_values = self.critic(next_states).squeeze()
            td_target = rewards + (1 - dones) * GAMMA * next_values
        
        current_values = self.critic(states).squeeze()
        critic_loss = F.mse_loss(current_values, td_target)  # [论文 Eq. 9]
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optim.step()
        
        # ========== Actor 更新 ==========
        # 优势函数：A(s,a) = R + γV(s') - V(s)
        with torch.no_grad():
            advantages = td_target - current_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        logits = self.actor(states)
        log_probs = F.log_softmax(logits, dim=1)
        log_probs_selected = log_probs.gather(1, (actions + 1).unsqueeze(1)).squeeze()
        
        actor_loss = -(log_probs_selected * advantages).mean()  # [论文 Eq. 7]
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optim.step()
        
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        
        # 清空缓冲区
        self.states_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.next_states_buffer = []
        self.dones_buffer = []
        
        return actor_loss.item(), critic_loss.item()

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
    print(f"📊 训练 {name} - A2C (Advantage Actor-Critic)")
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
    a2c = A2C()
    episode_rewards = []
    step_count = 0
    
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(500):
            # 采样动作
            action = a2c.get_action(state)
            next_state, reward, done = env.step(action)
            
            # 存储转移
            a2c.store_transition(state, action, reward, next_state, done)
            
            # 实时更新 (每步更新)
            a2c.train_batch()
            
            total_reward += reward
            state = next_state
            step_count += 1
            
            # 每 BATCH_SIZE 步进行学习率调度
            if step_count % BATCH_SIZE == 0:
                a2c.actor_scheduler.step()
                a2c.critic_scheduler.step()
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        
        if (ep+1) % 50 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"    Episode {ep+1}/{episodes}: Avg Reward={avg:.4f}")
    
    a2c.rewards = episode_rewards
    print(f"  ✅ 完成，平均奖励：{np.mean(episode_rewards):.4f}")
    return a2c

# =============================================================================
# 主函数
# =============================================================================

def main():
    print("="*80)
    print("🔥 论文对齐 A2C 训练 - Advantage Actor-Critic (实时更新)")
    print("="*80)
    print(f"设备：{DEVICE}")
    print(f"数据：2011-2015")
    print(f"超参数：lr_critic={LR_CRITIC}, lr_actor={LR_ACTOR}, γ={GAMMA}, batch={BATCH_SIZE}")
    print("="*80)
    
    start = time.time()
    models = {}
    
    for name, tickers in CONTRACTS_BY_CLASS.items():
        models[name] = train_class(name, tickers)
    
    elapsed = (time.time() - start) / 60
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with open(f'models_a2c_paper_{ts}.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    print(f"\n{'='*80}")
    print(f"✅ 训练完成！用时：{elapsed:.1f} 分钟")
    print(f"📁 模型已保存：models_a2c_paper_{ts}.pkl")
    print("="*80)

if __name__ == "__main__":
    main()
