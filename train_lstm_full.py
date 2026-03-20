#!/usr/bin/env python3
"""
完整复现：32个合约 + LSTM + 可选交易成本
- 论文LSTM架构 [64, 32]
- 交易成本可选 10bps/20bps
- 所有论文超参数对齐
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from paper_components import (
    DifferentialSharpeRatio,
    MultiTimeScaleState,
    VolatilityScaler
)

# =============================================================================
# 论文配置
# =============================================================================

DATA_DIR = 'data/futures_processed'

# 交易成本（可选）
TRANSACTION_COST_10BPS = 0.001   # 10 basis points
TRANSACTION_COST_20BPS = 0.002   # 20 basis points (论文配置)
USE_TRANSACTION_COST = TRANSACTION_COST_20BPS  # 默认使用论文配置

# 论文超参数 (Table 1)
GAMMA = 0.3
BUFFER_SIZE = 5000
BATCH_SIZE_DQN = 64
BATCH_SIZE_A2C = 128
LEARNING_RATE = 0.0001
TARGET_UPDATE = 1000
TOTAL_TIMESTEPS = 50000

# LSTM配置 (论文 Section 4.3)
LSTM_HIDDEN_SIZES = [64, 32]  # 两层LSTM

# 训练/测试期
TRAIN_START = '2011-01-03'
TRAIN_END = '2015-12-31'
TEST_START = '2016-01-01'
TEST_END = '2019-12-31'

# 32个可用合约
CONTRACTS_BY_CLASS = {
    'Commodity': [
        'CL=F', 'GC=F', 'SI=F', 'HG=F', 'NG=F',
        'ZC=F', 'ZS=F', 'ZW=F',
        'KC=F', 'CC=F', 'SB=F', 'CT=F', 'OJ=F'
    ],
    'Equity Index': [
        'ES=F', 'NQ=F', 'YM=F'
    ],
    'Fixed Income': [
        'ZN=F', 'ZB=F', 'ZF=F', 'ZT=F', 'GE=F'
    ],
    'FX': [
        '6E=F', '6J=F', '6B=F', '6A=F', '6C=F',
        '6S=F', '6N=F', '6M=F', '6R=F'
    ]
}

# =============================================================================
# LSTM网络架构 (论文 Section 4.3)
# =============================================================================

class LSTMNetwork(nn.Module):
    """论文LSTM架构: 两层LSTM [64, 32] + LeakyReLU"""
    
    def __init__(self, input_size, hidden_sizes=[64, 32], output_size=1):
        super().__init__()
        
        self.hidden_sizes = hidden_sizes
        
        # 第一层LSTM
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        
        # 第二层LSTM
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        
        # 输出层
        self.fc = nn.Linear(hidden_sizes[1], output_size)
        
        # LeakyReLU (论文配置)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        
    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, seq_len, input_size)
            hidden: tuple of (h_n, c_n) for each layer
        Returns:
            output: (batch, output_size)
            new_hidden: tuple of hidden states
        """
        batch_size = x.size(0)
        
        # 初始化hidden state
        if hidden is None:
            h1 = torch.zeros(1, batch_size, self.hidden_sizes[0]).to(x.device)
            c1 = torch.zeros(1, batch_size, self.hidden_sizes[0]).to(x.device)
            h2 = torch.zeros(1, batch_size, self.hidden_sizes[1]).to(x.device)
            c2 = torch.zeros(1, batch_size, self.hidden_sizes[1]).to(x.device)
            hidden = ((h1, c1), (h2, c2))
        
        # 第一层LSTM
        out1, (h1_new, c1_new) = self.lstm1(x, hidden[0])
        out1 = self.leaky_relu(out1)
        
        # 第二层LSTM
        out2, (h2_new, c2_new) = self.lstm2(out1, hidden[1])
        out2 = self.leaky_relu(out2)
        
        # 取最后一个时间步
        out = out2[:, -1, :]
        
        # 输出层
        output = self.fc(out)
        
        new_hidden = ((h1_new, c1_new), (h2_new, c2_new))
        
        return output, new_hidden

# =============================================================================
# LSTM DQN
# =============================================================================

class LSTMDQNAgent:
    """LSTM版本的DQN"""
    
    def __init__(self, state_dim, n_actions=3, lr=0.0001, gamma=0.3, 
                 buffer_size=5000, batch_size=64, target_update=1000):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        
        # 网络
        self.q_network = LSTMNetwork(state_dim, [64, 32], n_actions).to(DEVICE)
        self.target_network = LSTMNetwork(state_dim, [64, 32], n_actions).to(DEVICE)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 经验回放
        self.buffer = []
        self.buffer_size = buffer_size
        
        self.steps = 0
        
    def get_action(self, state, hidden=None, epsilon=0.1):
        """选择动作"""
        if np.random.random() < epsilon:
            return np.random.randint(0, self.n_actions), hidden
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(DEVICE)
            q_values, new_hidden = self.q_network(state_tensor, hidden)
            action = q_values.argmax(dim=1).item()
            return action, new_hidden
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))
    
    def train(self):
        """训练"""
        if len(self.buffer) < self.batch_size:
            return
        
        # 采样
        batch = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        
        states = torch.FloatTensor([self.buffer[i][0] for i in batch]).unsqueeze(1).to(DEVICE)
        actions = torch.LongTensor([self.buffer[i][1] for i in batch]).to(DEVICE)
        rewards = torch.FloatTensor([self.buffer[i][2] for i in batch]).to(DEVICE)
        next_states = torch.FloatTensor([self.buffer[i][3] for i in batch]).unsqueeze(1).to(DEVICE)
        dones = torch.FloatTensor([self.buffer[i][4] for i in batch]).to(DEVICE)
        
        # 当前Q值
        current_q, _ = self.q_network(states)
        current_q = current_q.gather(1, actions.unsqueeze(1))
        
        # 目标Q值
        with torch.no_grad():
            next_q, _ = self.target_network(next_states)
            max_next_q = next_q.max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # 损失
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        # 更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

# =============================================================================
# LSTM A2C
# =============================================================================

class LSTMA2CAgent:
    """LSTM版本的A2C - 100%对齐论文Table 1"""
    
    def __init__(self, state_dim, lr_actor=0.0001, lr_critic=0.0001, gamma=0.3):
        self.state_dim = state_dim
        self.gamma = gamma
        
        # Actor网络 (输出动作分布)
        self.actor = LSTMNetwork(state_dim, [64, 32], 2).to(DEVICE)  # mean, std
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Critic网络
        self.critic = LSTMNetwork(state_dim, [64, 32], 1).to(DEVICE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
    def get_action(self, state, hidden_actor=None, hidden_critic=None):
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            # 获取动作分布参数
            action_params, new_hidden_actor = self.actor(state_tensor, hidden_actor)
            mean = action_params[:, 0]
            std = F.softplus(action_params[:, 1]) + 0.1
            
            # 采样
            dist = Normal(mean, std)
            action = dist.sample()
            action = torch.tanh(action)  # 限制到 [-1, 1]
            
            return action.item(), new_hidden_actor, new_hidden_critic
    
    def train(self, states, actions, rewards, next_states, dones):
        """训练一个episode"""
        states = torch.FloatTensor(states).unsqueeze(1).to(DEVICE)
        actions = torch.FloatTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_states = torch.FloatTensor(next_states).unsqueeze(1).to(DEVICE)
        dones = torch.FloatTensor(dones).to(DEVICE)
        
        # 计算returns
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(DEVICE)
        
        # Critic loss
        values, _ = self.critic(states)
        values = values.squeeze()
        critic_loss = F.mse_loss(values, returns)
        
        # Actor loss
        action_params, _ = self.actor(states)
        means = action_params[:, 0]
        stds = F.softplus(action_params[:, 1]) + 0.1
        
        dist = Normal(means, stds)
        log_probs = dist.log_prob(torch.atanh(torch.clamp(actions, -0.99, 0.99)))
        
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
    
    def __init__(self, prices, returns, transaction_cost=TRANSACTION_COST_20BPS, use_dsr=True):
        self.prices = prices
        self.returns = returns
        self.transaction_cost = transaction_cost
        self.use_dsr = use_dsr
        self.n_steps = len(returns)
        
        self.state_builder = MultiTimeScaleState()
        self.dsr = DifferentialSharpeRatio(eta=0.01)
        self.scaler = VolatilityScaler(target_vol=0.10)
        
        self.step_idx = 200
        self.last_action = 0.0
        self.returns_history = []
        self.positions_history = []
        
    def reset(self):
        self.step_idx = 200
        self.last_action = 0.0
        self.dsr.reset()
        self.returns_history = []
        self.positions_history = []
        return self._obs()
    
    def _obs(self):
        return self.state_builder.compute(
            self.prices[:self.step_idx+1],
            self.returns[:self.step_idx+1],
            self.step_idx
        ).astype(np.float32)
    
    def step(self, action):
        """action: float in [-1, 1]"""
        action = float(np.clip(action, -1, 1))
        cost = abs(action - self.last_action) * self.transaction_cost
        
        if self.step_idx + 1 >= self.n_steps:
            return self._obs(), 0, True
        
        ret = self.returns[self.step_idx + 1]
        strat_ret = action * ret - cost
        
        scaled = self.scaler.scale(1.0, self.returns[:self.step_idx+1], self.step_idx)
        strat_ret *= scaled
        
        self.returns_history.append(strat_ret)
        self.positions_history.append(action)
        
        reward = self.dsr.update(strat_ret) if self.use_dsr else strat_ret
        
        self.step_idx += 1
        self.last_action = action
        
        return self._obs(), reward, self.step_idx >= self.n_steps - 1

# =============================================================================
# 训练函数
# =============================================================================

def train_dqn_lstm(env_train, timesteps=50000):
    """训练LSTM DQN"""
    state_dim = 16
    agent = LSTMDQNAgent(
        state_dim=state_dim,
        n_actions=3,  # {-1, 0, 1}
        lr=LEARNING_RATE,
        gamma=GAMMA,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE_DQN,
        target_update=TARGET_UPDATE
    )
    
    state = env_train.reset()
    hidden = None
    
    for step in range(timesteps):
        # Epsilon decay
        epsilon = max(0.01, 1.0 - step / 10000)
        
        # 选择动作
        action_idx, hidden = agent.get_action(state, hidden, epsilon)
        action = float(action_idx - 1)  # {0,1,2} -> {-1,0,1}
        
        # 执行动作
        next_state, reward, done = env_train.step(action)
        
        # 存储经验
        agent.remember(state, action_idx, reward, next_state, float(done))
        
        # 训练
        agent.train()
        
        state = next_state
        if done:
            state = env_train.reset()
            hidden = None
    
    return agent

def train_a2c_lstm(env_train, n_episodes=100):
    """训练LSTM A2C"""
    state_dim = 16
    agent = LSTMA2CAgent(
        state_dim=state_dim,
        lr_actor=LEARNING_RATE,
        lr_critic=0.001,
        gamma=GAMMA
    )
    
    for episode in range(n_episodes):
        state = env_train.reset()
        hidden_actor = None
        hidden_critic = None
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        done = False
        while not done:
            action, hidden_actor, hidden_critic = agent.get_action(
                state, hidden_actor, hidden_critic
            )
            
            next_state, reward, done = env_train.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(float(done))
            
            state = next_state
        
        # Episode结束后训练
        agent.train(states, actions, rewards, next_states, dones)
    
    return agent

# =============================================================================
# 基线策略
# =============================================================================

def baseline_long(n): 
    return np.ones(n)

def baseline_sign(returns, window=252):
    signals = np.zeros(len(returns))
    for i in range(window, len(returns)):
        cum_ret = np.sum(returns[i-window:i])
        signals[i] = np.sign(cum_ret)
    return signals

def baseline_macd(prices):
    p = pd.Series(prices)
    macd = p.ewm(span=12).mean() - p.ewm(span=26).mean()
    sig = macd / (0.89 * np.exp(-macd**2 / 4))
    return np.tanh(sig.values)

# =============================================================================
# 指标计算
# =============================================================================

def calc_metrics(returns):
    """计算所有指标"""
    if len(returns) == 0:
        return {'E(R)': 0, 'Std(R)': 0, 'Sharpe': 0, 'Sortino': 0, 'MDD': 0, 'Calmar': 0}
    
    annual_factor = np.sqrt(252)
    er = np.mean(returns) * 252
    std_r = np.std(returns) * annual_factor
    
    neg_ret = returns[returns < 0]
    dd = np.std(neg_ret) * annual_factor if len(neg_ret) > 0 else 0.001
    
    sharpe = er / std_r if std_r > 0 else 0
    sortino = er / dd if dd > 0 else 0
    
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    drawdowns = (peak - cum) / peak
    mdd = np.max(drawdowns) if len(drawdowns) > 0 else 0
    calmar = er / mdd if mdd > 0 else 0
    
    return {
        'E(R)': er,
        'Std(R)': std_r,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'MDD': -mdd,
        'Calmar': calmar
    }

# =============================================================================
# 训练单个合约
# =============================================================================

def train_contract(ticker, transaction_cost=TRANSACTION_COST_20BPS, quiet=False):
    """训练单个合约"""
    try:
        df = pd.read_csv(f'{DATA_DIR}/{ticker}.csv')
    except:
        if not quiet:
            print(f"  ⚠️ {ticker} 数据不存在")
        return None, None
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    train = df[(df['Date'] >= TRAIN_START) & (df['Date'] <= TRAIN_END)]
    test = df[(df['Date'] >= TEST_START) & (df['Date'] <= TEST_END)]
    
    if len(train) < 500 or len(test) < 200:
        if not quiet:
            print(f"  ⚠️ {ticker} 数据不足")
        return None, None
    
    train_p = train['Close'].values
    train_r = train['Returns'].values
    test_p = test['Close'].values
    test_r = test['Returns'].values
    
    results = []
    daily_returns = {}
    
    # ===== 基线策略 =====
    
    # Long
    pos = baseline_long(len(test_r)-200)
    strat_ret = test_r[200:] * pos
    m = calc_metrics(strat_ret)
    m['Ticker'] = ticker
    m['Strategy'] = 'Long'
    results.append(m)
    daily_returns['Long'] = strat_ret
    
    # Sign(R)
    pos = baseline_sign(test_r)[200:]
    pos = pos[:len(test_r)-200]
    if len(pos) > 0:
        strat_ret = test_r[200:200+len(pos)] * pos
        m = calc_metrics(strat_ret)
        m['Ticker'] = ticker
        m['Strategy'] = 'Sign(R)'
        results.append(m)
        daily_returns['Sign(R)'] = strat_ret
    
    # MACD
    pos = baseline_macd(test_p)[200:]
    pos = pos[:len(test_r)-200]
    if len(pos) > 0:
        strat_ret = test_r[200:200+len(pos)] * pos
        m = calc_metrics(strat_ret)
        m['Ticker'] = ticker
        m['Strategy'] = 'MACD'
        results.append(m)
        daily_returns['MACD'] = strat_ret
    
    # ===== LSTM DQN =====
    try:
        env_train = TradingEnv(train_p, train_r, transaction_cost, use_dsr=True)
        agent = train_dqn_lstm(env_train, TOTAL_TIMESTEPS)
        
        # 测试
        env_test = TradingEnv(test_p, test_r, transaction_cost, use_dsr=False)
        state = env_test.reset()
        hidden = None
        done = False
        
        while not done:
            action_idx, hidden = agent.get_action(state, hidden, epsilon=0.0)
            action = float(action_idx - 1)
            state, _, done = env_test.step(action)
        
        m = calc_metrics(np.array(env_test.returns_history))
        m['Ticker'] = ticker
        m['Strategy'] = 'DQN'
        results.append(m)
        daily_returns['DQN'] = np.array(env_test.returns_history)
        
    except Exception as e:
        if not quiet:
            print(f"    ⚠️ {ticker} LSTM DQN失败: {e}")
    
    # ===== LSTM A2C =====
    try:
        env_train = TradingEnv(train_p, train_r, transaction_cost, use_dsr=True)
        agent = train_a2c_lstm(env_train, n_episodes=100)
        
        # 测试
        env_test = TradingEnv(test_p, test_r, transaction_cost, use_dsr=False)
        state = env_test.reset()
        hidden_actor = None
        hidden_critic = None
        done = False
        
        while not done:
            action, hidden_actor, hidden_critic = agent.get_action(
                state, hidden_actor, hidden_critic
            )
            state, _, done = env_test.step(action)
        
        m = calc_metrics(np.array(env_test.returns_history))
        m['Ticker'] = ticker
        m['Strategy'] = 'A2C'
        results.append(m)
        daily_returns['A2C'] = np.array(env_test.returns_history)
        
    except Exception as e:
        if not quiet:
            print(f"    ⚠️ {ticker} LSTM A2C失败: {e}")
    
    return results, daily_returns

# =============================================================================
# 主函数
# =============================================================================

def main(transaction_cost_bps=20):
    """训练所有合约"""
    
    # 设置交易成本
    global USE_TRANSACTION_COST
    USE_TRANSACTION_COST = transaction_cost_bps / 10000
    tc_name = f"{transaction_cost_bps}bps"
    
    print("="*80)
    print(f"🤖 LSTM完整复现：32个合约 + {tc_name}")
    print("="*80)
    print(f"设备: {DEVICE}")
    print(f"交易成本: {tc_name}")
    print(f"γ: {GAMMA}")
    print(f"LSTM架构: {LSTM_HIDDEN_SIZES}")
    print(f"步数: {TOTAL_TIMESTEPS:,}")
    print()
    
    all_results = []
    all_daily_returns = {}
    
    total_contracts = sum(len(v) for v in CONTRACTS_BY_CLASS.values())
    trained = 0
    
    for asset_class, tickers in CONTRACTS_BY_CLASS.items():
        print(f"\n{'='*70}")
        print(f"📊 {asset_class} ({len(tickers)} 合约)")
        print('='*70)
        
        for ticker in tickers:
            trained += 1
            print(f"  [{trained}/{total_contracts}] {ticker:<8} | ", end='', flush=True)
            
            results, daily_returns = train_contract(ticker, USE_TRANSACTION_COST, quiet=True)
            
            if results:
                all_results.extend(results)
                all_daily_returns[ticker] = daily_returns
                
                for r in results:
                    if r['Strategy'] in ['Long', 'DQN', 'A2C']:
                        print(f"{r['Strategy']}: {r['Sharpe']:>6.2f} | ", end='', flush=True)
                print("✅")
            else:
                print("❌")
    
    # 保存结果
    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results_lstm_{tc_name}_{timestamp}.csv'
    df.to_csv(filename, index=False)
    print(f"\n💾 结果已保存: {filename}")
    
    # 汇总
    print("\n" + "="*80)
    print("📊 总体汇总")
    print("="*80)
    
    summary = df.groupby('Strategy').agg({
        'Sharpe': ['mean', 'std'],
        'E(R)': 'mean',
        'MDD': 'mean'
    }).round(3)
    
    print(summary)

if __name__ == '__main__':
    import sys
    
    # 命令行参数：python train_lstm_full.py [10|20]
    tc = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    
    if tc not in [10, 20]:
        print("交易成本必须是 10 或 20 (bps)")
        sys.exit(1)
    
    main(transaction_cost_bps=tc)
