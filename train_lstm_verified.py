#!/usr/bin/env python3
"""
完整训练 - 基于Pilot测试验证的代码
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# 论文配置
# =============================================================================

GAMMA = 0.3
LEARNING_RATE = 0.0001
N_EPISODES = 200  # 每个资产类别
MAX_STEPS = 500

CONTRACTS_BY_CLASS = {
    'Commodity': ['CL=F', 'GC=F', 'SI=F', 'HG=F', 'NG=F', 'ZC=F', 'ZS=F', 'ZW=F', 'KC=F', 'CC=F', 'SB=F', 'CT=F', 'OJ=F'],
    'Equity Index': ['ES=F', 'NQ=F', 'YM=F'],
    'Fixed Income': ['ZN=F', 'ZB=F', 'ZF=F', 'ZT=F', 'GE=F'],
    'FX': ['6E=F', '6J=F', '6B=F', '6A=F', '6C=F', '6S=F', '6N=F', '6M=F', '6R=F']
}

# =============================================================================
# LSTM网络
# =============================================================================

class LSTMNetwork(nn.Module):
    def __init__(self, input_dim, hidden_sizes=[64, 32], output_dim=3):
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

# =============================================================================
# 环境
# =============================================================================

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
        if self.step_idx >= 20:
            obs[0] = np.mean(self.returns[self.step_idx-20:self.step_idx])
            obs[1] = np.std(self.returns[self.step_idx-20:self.step_idx])
            obs[2] = self.returns[self.step_idx-1] if self.step_idx >= 1 else 0
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

# =============================================================================
# DQN Agent
# =============================================================================

class SimpleDQN:
    def __init__(self, state_dim=16, n_actions=3):
        self.q_net = LSTMNetwork(state_dim, [64, 32], n_actions).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.gamma = GAMMA
        
    def get_action(self, state, epsilon=0.3):
        if np.random.random() < epsilon:
            return np.random.randint(0, 3)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(DEVICE)
            q_values = self.q_net(state_t)
            return q_values.argmax().item()
    
    def train(self, states, actions, rewards, next_states, dones):
        if len(states) < 10:
            return 0
        
        states = torch.FloatTensor(states).unsqueeze(1).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_states = torch.FloatTensor(next_states).unsqueeze(1).to(DEVICE)
        dones = torch.FloatTensor(dones).to(DEVICE)
        
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q = self.q_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

# =============================================================================
# 训练函数
# =============================================================================

def train_asset_class(asset_class, tickers):
    print(f"\n{'='*70}")
    print(f"📊 训练 {asset_class}")
    print('='*70)
    
    # 加载数据
    all_prices = []
    all_returns = []
    
    for ticker in tickers:
        try:
            df = pd.read_csv(f'data/futures_processed/{ticker}.csv')
            df['Date'] = pd.to_datetime(df['Date'])
            train = df[(df['Date'] >= '2011-01-03') & (df['Date'] <= '2015-12-31')]
            if len(train) < 500:
                continue
            all_prices.append(train['Close'].values)
            all_returns.append(train['Returns'].values)
        except:
            continue
    
    if not all_prices:
        print("  ⚠️ 无数据")
        return None
    
    prices = np.concatenate(all_prices)
    returns = np.concatenate(all_returns)
    
    print(f"  合约数: {len(all_prices)}")
    print(f"  总样本: {len(returns):,}")
    
    env = SimpleEnv(prices, returns)
    agent = SimpleDQN()
    
    print(f"  开始训练 {N_EPISODES} episodes...")
    
    episode_rewards = []
    
    for episode in range(N_EPISODES):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        while steps < MAX_STEPS:
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
            
            if done:
                break
        
        # 训练
        loss = agent.train(states, actions, rewards, next_states, dones)
        episode_rewards.append(total_reward)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"    Episode {episode+1}/{N_EPISODES}: Avg Reward={avg_reward:.4f}")
    
    print(f"  ✅ 完成，平均奖励: {np.mean(episode_rewards):.4f}")
    
    return agent

# =============================================================================
# 主函数
# =============================================================================

def main():
    print("="*80)
    print("🔥 LSTM训练 - 基于Pilot验证")
    print("="*80)
    print(f"设备: {DEVICE}")
    print(f"Episodes/类别: {N_EPISODES}")
    print()
    
    models = {}
    start_time = time.time()
    
    for asset_class, tickers in CONTRACTS_BY_CLASS.items():
        models[asset_class] = train_asset_class(asset_class, tickers)
    
    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'models_lstm_{timestamp}.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    elapsed = time.time() - start_time
    print(f"\n✅ 训练完成！")
    print(f"⏱️ 总时间: {elapsed/60:.1f} 分钟")
    print(f"💾 模型: models_lstm_{timestamp}.pkl")
    
    # 测试模型
    print("\n" + "="*80)
    print("📊 测试LSTM模型")
    print("="*80)
    
    PAPER = {
        'Commodity': {'Long': -0.726, 'DQN': 0.723},
        'Equity Index': {'Long': 0.688, 'DQN': 0.648},
        'Fixed Income': {'Long': 0.698, 'DQN': 0.935},
        'FX': {'Long': -0.353, 'DQN': 0.546}
    }
    
    all_results = []
    
    for asset_class in ['Commodity', 'Equity Index', 'Fixed Income', 'FX']:
        print(f"\n{'='*70}")
        print(f"【{asset_class}】")
        print('='*70)
        
        if asset_class not in models or models[asset_class] is None:
            print("  ⚠️ 无模型")
            continue
        
        agent = models[asset_class]
        tickers = CONTRACTS_BY_CLASS[asset_class]
        
        # 测试Long
        long_sharpes = []
        for ticker in tickers:
            try:
                df = pd.read_csv(f'data/futures_processed/{ticker}.csv')
                df['Date'] = pd.to_datetime(df['Date'])
                test = df[(df['Date'] >= '2016-01-01') & (df['Date'] <= '2019-12-31')]
                if len(test) < 200:
                    continue
                
                returns = test['Returns'].values[50:]
                er = np.mean(returns) * 252
                std = np.std(returns) * np.sqrt(252)
                sharpe = er / std if std > 0 else 0
                long_sharpes.append(sharpe)
            except:
                continue
        
        avg_long = np.mean(long_sharpes) if long_sharpes else 0
        paper_long = PAPER[asset_class]['Long']
        print(f"  Long: {avg_long:.3f} vs 论文 {paper_long:.3f} ({avg_long-paper_long:+.3f})")
        
        # 测试DQN
        print(f"\n  测试DQN策略...")
        dqn_sharpes = []
        for ticker in tickers:
            try:
                df = pd.read_csv(f'data/futures_processed/{ticker}.csv')
                df['Date'] = pd.to_datetime(df['Date'])
                test = df[(df['Date'] >= '2016-01-01') & (df['Date'] <= '2019-12-31')]
                if len(test) < 200:
                    continue
                
                prices = test['Close'].values
                returns = test['Returns'].values
                
                # 测试DQN
                env = SimpleEnv(prices, returns)
                state = env.reset()
                test_returns = []
                
                for _ in range(len(returns) - 100):
                    action_idx = agent.get_action(state, epsilon=0.0)
                    action = float(action_idx - 1)
                    next_state, reward, done = env.step(action)
                    test_returns.append(reward)
                    state = next_state
                    if done:
                        break
                
                # 计算Sharpe
                if len(test_returns) > 0:
                    er = np.mean(test_returns) * 252
                    std = np.std(test_returns) * np.sqrt(252)
                    sharpe = er / std if std > 0 else 0
                    dqn_sharpes.append(sharpe)
                    print(f"    {ticker}: {sharpe:.3f}")
            except Exception as e:
                print(f"    {ticker}: ⚠️ {e}")
                continue
        
        avg_dqn = np.mean(dqn_sharpes) if dqn_sharpes else 0
        paper_dqn = PAPER[asset_class]['DQN']
        diff_dqn = avg_dqn - paper_dqn
        status = '✅' if abs(diff_dqn) < 0.5 else ('⚠️' if abs(diff_dqn) < 1.0 else '❌')
        print(f"\n  DQN平均: {avg_dqn:.3f} vs 论文 {paper_dqn:.3f} ({diff_dqn:+.3f}) {status}")
        
        all_results.append({
            'Asset Class': asset_class,
            'Long_Ours': avg_long,
            'Long_Paper': paper_long,
            'DQN_Ours': avg_dqn,
            'DQN_Paper': paper_dqn,
            'DQN_Diff': diff_dqn
        })
    
    # 保存结果
    df_results = pd.DataFrame(all_results)
    df_results.to_csv('lstm_test_results.csv', index=False)
    print(f"\n💾 结果已保存: lstm_test_results.csv")

if __name__ == '__main__':
    main()
