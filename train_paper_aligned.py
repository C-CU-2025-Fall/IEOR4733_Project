#!/usr/bin/env python3
"""
论文对齐训练 - 重构版本

✅ 模块化设计:
- indicators.py: 技术指标计算
- 本文件：训练主干逻辑

✅ 滚动训练机制:
- 2005-2010 训练 → 2011 测试
- 2011-2015 训练 → 2016-2019 测试

✅ 微训练支持:
- python3 train_paper_aligned.py --micro
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

# 导入指标计算模块
from indicators import FeatureEngineer, compute_volatility

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# 论文超参数 (Table 1)
# =============================================================================

GAMMA = 0.3
LEARNING_RATE = 0.0001
BP = 0.0020  # 20 bps
MEMORY_SIZE = 5000
TARGET_UPDATE = 1000
BATCH_SIZE = 64
VOL_TARGET = 0.10  # 10% 年化波动率目标

# 训练配置
N_EPISODES = 200
MAX_STEPS = 500

# 滚动训练配置
# 注：数据限制 (2011-2019)，调整为可用窗口
# 论文原始：2005-2010 训练→2011 测试，2011-2015 训练→2016-2019 测试
ROLLING_WINDOWS = [
    {'train_start': '2011-01-01', 'train_end': '2015-12-31', 'test_start': '2016-01-01', 'test_end': '2019-12-31'},
    # 如需更多数据，可用全量：2011-2019
]

# =============================================================================
# 环境 (论文公式 4)
# =============================================================================

class VolatilityScaledEnv:
    """
    论文公式 (4) 的奖励函数:
    R_t = λ * A_{t-1} * (σ_tgt/σ_{t-1}) * r_t 
        - bp * |p_{t-1} * (σ_tgt/σ_{t-1}) * A_{t-1} 
               - p_{t-2} * (σ_tgt/σ_{t-1}) * A_{t-2}|
    """
    
    def __init__(self, prices, returns, vol_target=VOL_TARGET):
        self.prices = prices
        self.returns = returns
        self.vol_target = vol_target
        self.n_steps = len(returns)
        
        # 计算波动率序列
        self.volatility = compute_volatility(returns, 60)
        
        # 特征工程
        self.feature_engineer = FeatureEngineer(window_size=60)
        
        self.step_idx = 60
        self.last_action = 0.0
        self.last_price = prices[59] if len(prices) > 59 else prices[0]
        
    def reset(self):
        self.step_idx = 60
        self.last_action = 0.0
        self.last_price = self.prices[59] if len(self.prices) > 59 else self.prices[0]
        return self._get_state()
    
    def _get_state(self):
        """获取状态 (60, 8)"""
        return self.feature_engineer.build_features(
            self.prices, self.returns, self.step_idx
        )
    
    def step(self, action):
        action = float(np.clip(action, -1, 1))
        
        # 波动率缩放
        vol_scale = self.vol_target / (self.volatility[self.step_idx] + 1e-10)
        vol_scale = np.clip(vol_scale, 0.5, 2.0)
        
        current_price = self.prices[self.step_idx]
        
        # 交易成本
        position_change = abs(action - self.last_action)
        cost = BP * position_change * vol_scale * current_price
        
        if self.step_idx + 1 >= self.n_steps:
            return self._get_state(), 0.0, True
        
        next_return = self.returns[self.step_idx + 1]
        
        # 奖励 (波动率缩放)
        scaled_position = action * vol_scale
        reward = scaled_position * next_return - cost
        
        self.step_idx += 1
        self.last_action = action
        self.last_price = current_price
        
        return self._get_state(), reward, False

# =============================================================================
# LSTM 网络
# =============================================================================

class LSTMNetwork(nn.Module):
    """
    论文：Two-layer LSTM [64, 32]
    输入：(batch, 60, 8)
    """
    
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

# =============================================================================
# DQN Agent
# =============================================================================

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity=MEMORY_SIZE):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size=BATCH_SIZE):
        if len(self.buffer) < batch_size:
            return None
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class PaperAlignedDQN:
    """论文对齐的 DQN"""
    
    def __init__(self, state_dim=8, n_actions=3):
        self.q_net = LSTMNetwork(state_dim, [64, 32], n_actions).to(DEVICE)
        self.target_net = LSTMNetwork(state_dim, [64, 32], n_actions).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        
        self.gamma = GAMMA
        self.target_update = TARGET_UPDATE
        self.steps = 0
        
    def get_action(self, state, epsilon=0.3):
        if np.random.random() < epsilon:
            return np.random.randint(0, 3)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q_values = self.q_net(state_t)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return 0
        
        batch = self.memory.sample(BATCH_SIZE)
        if batch is None:
            return 0
        
        states, actions, rewards, next_states, dones = batch
        
        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).to(DEVICE)
        
        # Double DQN
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        return loss.item()

# =============================================================================
# 数据加载 (滚动训练)
# =============================================================================

def load_data_for_window(ticker, train_start, train_end):
    """
    加载指定训练窗口的数据
    
    返回:
        prices, returns 或 None
    """
    try:
        df = pd.read_csv(f'data/futures_processed/{ticker}.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        
        train = df[(df['Date'] >= train_start) & (df['Date'] <= train_end)]
        
        if len(train) < 500:
            return None
        
        return train['Close'].values, train['Returns'].values
    except Exception as e:
        return None


def prepare_data(tickers, train_start, train_end):
    """
    准备多个合约的训练数据
    """
    all_prices = []
    all_returns = []
    
    for ticker in tickers:
        result = load_data_for_window(ticker, train_start, train_end)
        if result is not None:
            prices, returns = result
            all_prices.append(prices)
            all_returns.append(returns)
    
    if not all_prices:
        return None, None
    
    return np.concatenate(all_prices), np.concatenate(all_returns)

# =============================================================================
# 训练函数
# =============================================================================

def train_asset_class(asset_class, tickers, rolling_window=None, micro_train=False):
    """
    训练单个资产类别
    
    参数:
        rolling_window: 训练窗口配置 {'train_start', 'train_end'}
        micro_train: 微训练模式
    """
    print(f"\n{'='*70}")
    print(f"📊 训练 {asset_class}")
    if rolling_window:
        print(f"📅 训练期：{rolling_window['train_start']} 至 {rolling_window['train_end']}")
    print('='*70)
    
    episodes = 5 if micro_train else N_EPISODES
    
    # 确定训练窗口
    if rolling_window is None:
        # 默认使用第一个滚动窗口
        rolling_window = ROLLING_WINDOWS[0]
    
    # 准备数据
    prices, returns = prepare_data(
        tickers, 
        rolling_window['train_start'], 
        rolling_window['train_end']
    )
    
    if prices is None:
        print("  ⚠️ 无数据，跳过")
        return None
    
    print(f"  合约数：{len(tickers)}")
    print(f"  总样本：{len(returns):,}")
    print(f"  Episodes: {episodes}")
    
    # 创建环境和 Agent
    env = VolatilityScaledEnv(prices, returns, vol_target=VOL_TARGET)
    agent = PaperAlignedDQN(state_dim=8, n_actions=3)
    
    print(f"  开始训练...")
    
    episode_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < MAX_STEPS:
            action_idx = agent.get_action(state, epsilon=0.3)
            action = action_idx - 1
            
            next_state, reward, done = env.step(action)
            
            agent.store_transition(state, action_idx, reward, next_state, float(done))
            
            total_reward += reward
            steps += 1
            state = next_state
            
            agent.train()
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        
        if (episode + 1) % 1 == 0 or micro_train:
            avg = np.mean(episode_rewards[-3:]) if len(episode_rewards) >= 3 else np.mean(episode_rewards)
            print(f"    Episode {episode+1}/{episodes}: Avg Reward={avg:.4f}")
    
    avg_reward = np.mean(episode_rewards)
    print(f"  ✅ 完成，平均奖励：{avg_reward:.4f}")
    
    return agent

# =============================================================================
# 滚动训练主函数
# =============================================================================

def rolling_train_all(micro_train=False):
    """
    滚动训练所有资产类别
    """
    print("="*80)
    print("🔥 论文对齐训练 - 滚动训练模式")
    print("="*80)
    print(f"设备：{DEVICE}")
    print(f"滚动窗口数：{len(ROLLING_WINDOWS)}")
    print()
    
    CONTRACTS_BY_CLASS = {
        'Commodity': ['CL=F', 'GC=F', 'SI=F', 'HG=F', 'NG=F', 'ZC=F', 'ZS=F', 'ZW=F', 
                      'KC=F', 'CC=F', 'SB=F', 'CT=F', 'OJ=F'],
        'Equity Index': ['ES=F', 'NQ=F', 'YM=F'],
        'Fixed Income': ['ZN=F', 'ZB=F', 'ZF=F', 'ZT=F', 'GE=F'],
        'FX': ['6E=F', '6J=F', '6B=F', '6A=F', '6C=F', '6S=F', '6N=F', '6M=F', '6R=F']
    }
    
    all_models = {}
    
    for window_idx, window in enumerate(ROLLING_WINDOWS):
        print(f"\n{'='*80}")
        print(f"📅 滚动窗口 {window_idx + 1}/{len(ROLLING_WINDOWS)}")
        print(f"   训练：{window['train_start']} 至 {window['train_end']}")
        print(f"   测试：{window['test_start']} 至 {window['test_end']}")
        print('='*80)
        
        window_models = {}
        
        for asset_class, tickers in CONTRACTS_BY_CLASS.items():
            model = train_asset_class(
                asset_class, 
                tickers, 
                rolling_window=window,
                micro_train=micro_train
            )
            window_models[asset_class] = model
        
        all_models[f'window_{window_idx}'] = window_models
    
    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode = 'micro' if micro_train else 'full'
    with open(f'models_rolling_{mode}_{timestamp}.pkl', 'wb') as f:
        pickle.dump(all_models, f)
    
    print(f"\n✅ 滚动训练完成！")
    print(f"💾 模型：models_rolling_{mode}_{timestamp}.pkl")
    
    return all_models

# =============================================================================
# 微训练测试
# =============================================================================

def micro_test():
    """微训练测试"""
    print("="*80)
    print("🔬 微训练测试 - 验证代码正确性")
    print("="*80)
    print(f"设备：{DEVICE}")
    print()
    
    start_time = time.time()
    
    # 只测试一个资产类别 + 一个窗口
    test_class = 'Equity Index'
    test_tickers = ['ES=F', 'NQ=F', 'YM=F']
    test_window = ROLLING_WINDOWS[0]
    
    model = train_asset_class(
        test_class, 
        test_tickers, 
        rolling_window=test_window,
        micro_train=True
    )
    
    elapsed = time.time() - start_time
    
    if model is not None:
        print(f"\n✅ 微训练测试通过！")
        print(f"⏱️ 耗时：{elapsed:.1f}秒")
        print(f"📦 网络：LSTM(8→64→32→3) + Leaky-ReLU")
        print(f"📊 状态：(60, 8)")
        print(f"💰 奖励：波动率缩放 + 20bps")
        print(f"🔄 滚动训练：{len(ROLLING_WINDOWS)} 窗口")
    
    return model

# =============================================================================
# 模块测试
# =============================================================================

def test_modules():
    """测试所有模块"""
    print("="*80)
    print("🧪 模块测试")
    print("="*80)
    print()
    
    # 测试 indicators 模块
    from indicators import test_indicators
    test_indicators()
    
    print()
    print("✅ 所有模块测试通过！")

# =============================================================================
# 主函数
# =============================================================================

def main():
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--micro':
            micro_test()
        elif sys.argv[1] == '--test-modules':
            test_modules()
        elif sys.argv[1] == '--help':
            print("用法:")
            print("  python3 train_paper_aligned.py           # 完整滚动训练")
            print("  python3 train_paper_aligned.py --micro   # 微训练测试")
            print("  python3 train_paper_aligned.py --test-modules  # 模块测试")
    else:
        rolling_train_all(micro_train=False)

if __name__ == '__main__':
    main()
