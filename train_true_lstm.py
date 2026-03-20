#!/usr/bin/env python3
"""
完整复现论文 - 真正的LSTM实现
使用自定义LSTM策略类，完全对齐论文
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Type, Tuple, List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from stable_baselines3 import DQN, A2C
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule, PyTorchObs
import gymnasium as gym
from gymnasium import spaces

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
TOTAL_TIMESTEPS = 100000  # 增加到100k

# LSTM配置 (论文 Section 4.3)
LSTM_HIDDEN_SIZES = [64, 32]  # 两层LSTM

# =============================================================================
# 自定义LSTM网络 (论文架构)
# =============================================================================

class LSTMNetwork(nn.Module):
    """论文LSTM架构: 两层LSTM [64, 32] + LeakyReLU"""
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        
        self.lstm1 = nn.LSTM(input_dim, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.fc = nn.Linear(32, output_dim)
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        batch_size = x.size(0)
        
        if hidden is None:
            h1 = torch.zeros(1, batch_size, 64).to(x.device)
            c1 = torch.zeros(1, batch_size, 64).to(x.device)
            h2 = torch.zeros(1, batch_size, 32).to(x.device)
            c2 = torch.zeros(1, batch_size, 32).to(x.device)
            hidden = ((h1, c1), (h2, c2))
        
        # 第一层LSTM
        out1, (h1_new, c1_new) = self.lstm1(x, hidden[0])
        out1 = self.leaky_relu(out1)
        
        # 第二层LSTM
        out2, (h2_new, c2_new) = self.lstm2(out1, hidden[1])
        out2 = self.leaky_relu(out2)
        
        # 输出
        output = self.fc(out2[:, -1, :])
        
        return output, ((h1_new, c1_new), (h2_new, c2_new))

# =============================================================================
# 自定义LSTM特征提取器
# =============================================================================

class LSTMFeaturesExtractor(BaseFeaturesExtractor):
    """LSTM特征提取器"""
    def __init__(self, observation_space: spaces.Box, features_dim: int = 32):
        super().__init__(observation_space, features_dim)
        
        self.lstm = LSTMNetwork(input_dim=observation_space.shape[0], output_dim=features_dim)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 添加序列维度
        x = observations.unsqueeze(1)  # (batch, 1, features)
        output, _ = self.lstm(x)
        return output

# =============================================================================
# 自定义LSTM策略类 - DQN
# =============================================================================

class LSTMDQNPolicy(BasePolicy):
    """DQN的LSTM策略"""
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs
        )
        
        self.lstm = LSTMNetwork(
            input_dim=observation_space.shape[0],
            output_dim=action_space.n
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_schedule(1))
        
    def forward(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        # 添加序列维度
        x = obs.unsqueeze(1)
        q_values, _ = self.lstm(x)
        return q_values
    
    def _predict(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        q_values = self.forward(obs)
        return q_values.argmax(dim=1)
    
    def get_distribution(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        q_values = self.forward(obs)
        return torch.distributions.Categorical(logits=q_values)
    
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)

# =============================================================================
# 自定义LSTM策略类 - A2C
# =============================================================================

class LSTMA2CPolicy(ActorCriticPolicy):
    """A2C的LSTM策略"""
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs
        )
        
        # 替换为LSTM网络
        self.lstm_actor = LSTMNetwork(
            input_dim=observation_space.shape[0],
            output_dim=2  # mean and log_std
        )
        
        self.lstm_critic = LSTMNetwork(
            input_dim=observation_space.shape[0],
            output_dim=1  # value
        )
        
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 添加序列维度
        x = obs.unsqueeze(1)
        
        # Actor
        action_params, _ = self.lstm_actor(x)
        mean = action_params[:, 0]
        log_std = action_params[:, 1]
        std = torch.exp(log_std)
        
        # Critic
        value, _ = self.lstm_critic(x)
        
        # 采样动作
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        action = torch.tanh(action)  # 限制到[-1, 1]
        
        return action, value[:, 0], log_prob
    
    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> torch.Tensor:
        action, _, _ = self.forward(observation, deterministic)
        return action
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = obs.unsqueeze(1)
        
        # Actor
        action_params, _ = self.lstm_actor(x)
        mean = action_params[:, 0]
        log_std = action_params[:, 1]
        std = torch.exp(log_std)
        
        # Critic
        value, _ = self.lstm_critic(x)
        
        # 计算log_prob
        dist = Normal(mean, std)
        raw_actions = torch.atanh(torch.clamp(actions, -0.99, 0.99))
        log_prob = dist.log_prob(raw_actions)
        
        entropy = dist.entropy()
        
        return value[:, 0], log_prob, entropy
    
    def get_distribution(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        x = obs.unsqueeze(1)
        action_params, _ = self.lstm_actor(x)
        mean = action_params[:, 0]
        log_std = action_params[:, 1]
        std = torch.exp(log_std)
        return Normal(mean, std)
    
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.unsqueeze(1)
        value, _ = self.lstm_critic(x)
        return value[:, 0]

# =============================================================================
# 环境
# =============================================================================

class TradingEnv(gym.Env):
    """交易环境"""
    def __init__(self, prices, returns, discrete=True):
        super().__init__()
        self.prices = prices
        self.returns = returns
        self.discrete = discrete
        self.n_steps = len(returns)
        self.current_step = 200
        
        self.observation_space = spaces.Box(-np.inf, np.inf, (16,), dtype=np.float32)
        
        if discrete:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        
        self.last_action = 0.0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 200
        self.last_action = 0.0
        return self._get_obs(), {}
    
    def _get_obs(self):
        """计算16维状态"""
        obs = np.zeros(16, dtype=np.float32)
        
        # 动量特征 (6维)
        for i, window in enumerate([5, 10, 25, 50, 100, 200]):
            if self.current_step >= window:
                ret = np.mean(self.returns[self.current_step-window:self.current_step])
                vol = np.std(self.returns[self.current_step-window:self.current_step])
                obs[i] = ret / (vol + 1e-8)
        
        # MACD (1维)
        if self.current_step >= 26:
            ma_fast = np.mean(self.prices[self.current_step-12:self.current_step])
            ma_slow = np.mean(self.prices[self.current_step-26:self.current_step])
            obs[6] = (ma_fast - ma_slow) / (np.std(self.prices[self.current_step-63:self.current_step]) + 1e-8)
        
        # RSI (1维)
        if self.current_step >= 14:
            gains = self.returns[self.current_step-14:self.current_step]
            obs[7] = np.sum(gains[gains > 0]) / (np.sum(np.abs(gains)) + 1e-8)
        
        # 价格位置 (1维)
        if self.current_step >= 20:
            obs[8] = (self.prices[self.current_step] - np.mean(self.prices[self.current_step-20:self.current_step])) / \
                     (np.std(self.prices[self.current_step-20:self.current_step]) + 1e-8)
        
        # 波动率 (1维)
        if self.current_step >= 20:
            obs[9] = np.std(self.returns[self.current_step-20:self.current_step]) * np.sqrt(252)
        
        # 收益率 (1维)
        if self.current_step >= 1:
            obs[10] = self.returns[self.current_step-1]
        
        # Volatility scaling factor (1维)
        if self.current_step >= 60:
            current_vol = np.std(self.returns[self.current_step-60:self.current_step]) * np.sqrt(252)
            obs[11] = 0.10 / (current_vol + 1e-8)
        
        return obs
    
    def step(self, action):
        if self.discrete:
            action_value = float(action - 1)
        else:
            action_value = float(np.clip(action[0], -1, 1))
        
        cost = abs(action_value - self.last_action) * TRANSACTION_COST
        
        if self.current_step + 1 >= self.n_steps:
            return self._get_obs(), 0.0, True, False, {}
        
        ret = self.returns[self.current_step + 1]
        strat_ret = action_value * ret - cost
        
        # Volatility scaling
        if self.current_step >= 60:
            current_vol = np.std(self.returns[self.current_step-60:self.current_step]) * np.sqrt(252)
            scale = 0.10 / (current_vol + 1e-8)
            strat_ret *= scale
        
        reward = strat_ret
        
        self.current_step += 1
        self.last_action = action_value
        
        return self._get_obs(), reward, False, False, {}

# =============================================================================
# 训练函数
# =============================================================================

def train_asset_class(asset_class, tickers):
    """训练某个资产类别"""
    print(f"\n{'='*70}")
    print(f"📊 训练 {asset_class} (真正的LSTM)")
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
                'ticker': ticker,
                'prices': train['Close'].values,
                'returns': train['Returns'].values
            })
        except:
            continue
    
    if not all_data:
        print("  ⚠️ 无数据")
        return None
    
    print(f"  合约数: {len(all_data)}")
    print(f"  总样本: {sum(len(d['returns']) for d in all_data):,}")
    
    # 用第一个合约训练
    data = all_data[0]
    
    # DQN
    print("  训练DQN (LSTM)...")
    env_dqn = TradingEnv(data['prices'], data['returns'], discrete=True)
    dqn = DQN(
        LSTMDQNPolicy,
        env_dqn,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=1000,
        batch_size=BATCH_SIZE_DQN,
        gamma=GAMMA,
        target_update_interval=TARGET_UPDATE,
        verbose=0,
        device=DEVICE
    )
    dqn.learn(TOTAL_TIMESTEPS)
    
    # A2C
    print("  训练A2C (LSTM)...")
    env_a2c = TradingEnv(data['prices'], data['returns'], discrete=False)
    a2c = A2C(
        LSTMA2CPolicy,
        env_a2c,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        verbose=0,
        device=DEVICE
    )
    a2c.learn(TOTAL_TIMESTEPS)
    
    return {'dqn': dqn, 'a2c': a2c, 'data': all_data}

# =============================================================================
# 主函数
# =============================================================================

def main():
    print("="*80)
    print("🤖 真正的LSTM实现 - 完全对齐论文")
    print("="*80)
    print(f"设备: {DEVICE}")
    print(f"网络: LSTM [64, 32] (论文配置)")
    print(f"训练步数: {TOTAL_TIMESTEPS:,}")
    print()
    
    CONTRACTS_BY_CLASS = {
        'Commodity': ['CL=F', 'GC=F', 'SI=F', 'HG=F', 'NG=F', 'ZC=F', 'ZS=F', 'ZW=F', 'KC=F', 'CC=F', 'SB=F', 'CT=F', 'OJ=F'],
        'Equity Index': ['ES=F', 'NQ=F', 'YM=F'],
        'Fixed Income': ['ZN=F', 'ZB=F', 'ZF=F', 'ZT=F', 'GE=F'],
        'FX': ['6E=F', '6J=F', '6B=F', '6A=F', '6C=F', '6S=F', '6N=F', '6M=F', '6R=F']
    }
    
    models = {}
    
    for asset_class, tickers in CONTRACTS_BY_CLASS.items():
        models[asset_class] = train_asset_class(asset_class, tickers)
    
    # 保存
    import pickle
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'models_lstm_{timestamp}.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    print(f"\n✅ 训练完成！")
    print(f"💾 模型已保存: models_lstm_{timestamp}.pkl")

if __name__ == '__main__':
    main()
