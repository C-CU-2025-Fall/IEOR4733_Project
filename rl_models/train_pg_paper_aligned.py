#!/usr/bin/env python3
"""
Policy Gradient (PG) 训练 - 论文对齐实现

✅ Policy Network (Actor)
✅ Monte Carlo 采样
✅ 梯度累积
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
# 论文 Table 1 超参数 (PG)
# =============================================================================

LR_ACTOR = 0.0001     # 论文：0.0001
GAMMA = 0.3           # 论文：0.3
BP = 0.0020           # 论文：20 bps
VOL_TARGET = 0.10     # 10% 年化波动率目标

CONTRACTS_BY_CLASS = {
    'Commodity': ['CL', 'GC', 'SI', 'HG', 'NG', 'ZC', 'ZS', 'ZW', 'KC', 'CC', 'SB', 'CT', 'OJ'],
    'Equity Index': ['ES', 'NQ', 'YM'],
    'Fixed Income': ['ZN', 'ZB', 'ZF', 'ZT', 'GE'],
    'FX': ['6E', '6J', '6B', '6A', '6C', '6S', '6N', '6M', '6R']
}

# =============================================================================
# LSTM 网络 (与 DQN 相同)
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
# 环境 (与 DQN 相同)
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
# ✅ Policy Gradient (PG)
# =============================================================================

class PG:
    def __init__(self):
        # Actor 网络 (输出 3 个动作的概率)
        self.actor = LSTM(8, [64, 32], 3).to(DEVICE)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        
        # 学习率衰减
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optim, step_size=50, gamma=0.9)
        
        # 训练历史
        self.rewards = []
        self.losses = []
        self.trajectory = []
        
    def get_action(self, state):
        """从策略网络采样动作"""
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            logits = self.actor(s)
            probs = F.softmax(logits, dim=1)[0]
            action = torch.multinomial(probs, 1).item()
            return action - 1  # 转换为 -1, 0, 1
    
    def get_action_prob(self, state, action):
        """计算状态-动作对的 log 概率"""
        s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        logits = self.actor(s)
        log_probs = F.log_softmax(logits, dim=1)
        return log_probs[0, action + 1]  # action 转换为 0, 1, 2
    
    def train(self, trajectory):
        """
        论文算法：使用 Monte Carlo 回报计算梯度
        τ = [S0, A0, R1, S1, A1, R2, ...]
        G_t = sum_{k=0}^{T-1} γ^k R_{t+k}
        """
        states, actions, rewards = trajectory
        
        # 计算回报 (Monte Carlo returns)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(DEVICE)
        
        # 标准化回报 (降低方差)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 计算 log 概率
        log_probs = []
        for s, a in zip(states, actions):
            log_p = self.get_action_prob(s, a)
            log_probs.append(log_p)
        log_probs = torch.stack(log_probs)
        
        # 策略梯度损失: J(θ) = E[log π(A|S) * G_t]
        loss = -(log_probs * returns).mean()
        
        self.actor_optim.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        
        self.actor_optim.step()
        self.scheduler.step()
        
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
    print(f"📊 训练 {name} - PG (Policy Gradient)")
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
    pg = PG()
    episode_rewards = []
    
    for ep in range(episodes):
        state = env.reset()
        trajectory = [[], [], []]  # states, actions, rewards
        total_reward = 0
        
        for step in range(500):
            # 采样动作
            action = pg.get_action(state)
            next_state, reward, done = env.step(action)
            
            trajectory[0].append(state.copy())
            trajectory[1].append(action)
            trajectory[2].append(reward)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # 训练 (Monte Carlo): 仅在回合结束时更新
        if len(trajectory[2]) > 0:
            pg.train(trajectory)
        
        episode_rewards.append(total_reward)
        
        if (ep+1) % 50 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"    Episode {ep+1}/{episodes}: Avg Reward={avg:.4f}")
    
    pg.rewards = episode_rewards
    print(f"  ✅ 完成，平均奖励：{np.mean(episode_rewards):.4f}")
    return pg

# =============================================================================
# 主函数
# =============================================================================

def main():
    print("="*80)
    print("🔥 论文对齐 PG 训练 - Policy Gradient (Monte Carlo)")
    print("="*80)
    print(f"设备：{DEVICE}")
    print(f"数据：2011-2015")
    print(f"超参数：lr_actor={LR_ACTOR}, γ={GAMMA}")
    print("="*80)
    
    start = time.time()
    models = {}
    
    for name, tickers in CONTRACTS_BY_CLASS.items():
        models[name] = train_class(name, tickers)
    
    elapsed = (time.time() - start) / 60
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with open(f'models_pg_paper_{ts}.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    print(f"\n{'='*80}")
    print(f"✅ 训练完成！用时：{elapsed:.1f} 分钟")
    print(f"📁 模型已保存：models_pg_paper_{ts}.pkl")
    print("="*80)

if __name__ == "__main__":
    main()
