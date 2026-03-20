#!/usr/bin/env python3
"""
完整对齐论文methodology - 按类别训练 + 宰剪版
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
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
TRANSACTION_COST_20BPS = 0.002  # 20 basis points (论文配置)
GAMMA = 0.3
BUFFER_SIZE = 5000
BATCH_SIZE_DQN = 64
BATCH_SIZE_A2C = 128
LEARNING_RATE = 0.0001
TARGET_UPDATE = 1000
TOTAL_TIMESTEPS = 50000

# LSTM配置 (论文 Section 4.3)
LSTM_HIDDEN_SIZES = [64, 32]

# 训练/测试期
TRAIN_START = '2011-01-03'
TRAIN_END = '2015-12-31'
TEST_START = '2016-01-01'
TEST_END = '2019-12-31'

# 按资产类别分组 (论文要求)
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
# LSTM网络 (论文架构)
# =============================================================================

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes=[64, 32]):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        if hidden is None:
            h1 = torch.zeros(1, batch_size, self.hidden_sizes[0]).to(x.device)
            c1 = torch.zeros(1, batch_size, self.hidden_sizes[0]).to(x.device)
            h2 = torch.zeros(1, batch_size, self.hidden_sizes[1]).to(x.device)
            c2 = torch.zeros(1, batch_size, self.hidden_sizes[1]).to(x.device)
            hidden = ((h1, c1), (h2, c2))

        out1, (h1_new, c1_new) = self.lstm1(x, hidden[0])
        out1 = self.leaky_relu(out1)
        out2, (h2_new, c2_new) = self.lstm2(out1, hidden[1])
        out2 = self.leaky_relu(out2)
        return out2[:, -1, :], ((h1_new, c1_new), (h2_new, c2_new))

        return self.leaky_relu(out2[:, -1, :]), ((h1_new, c1_new), (h2_new, c2_new))

# =============================================================================
# 按资产类别训练的模型
# =============================================================================

class AssetClassModel:
    """每个资产类别训练一个模型"""
    def __init__(self, asset_class):
        self.asset_class = asset_class
        self.contracts = CONTRACTS_BY_CLASS[asset_class]
        self.dqn_model = None
        self.a2c_model = None

    def train(self):
        """训练该资产类别的模型"""
        print(f"\n训练 {self.asset_class} 模型...")
        print(f"合约: {len(self.contracts)}")

        # 收集该类别所有合约的数据
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_dones = []

        for ticker in self.contracts:
            data = self.load_data(ticker)
            if data is None:
                continue

            states = data['states']
            actions = data['actions']
            rewards = data['rewards']
            next_states = data['next_states']
            dones = data['dones']

            all_states.extend(states)
            all_actions.extend(actions)
            all_rewards.extend(rewards)
            all_next_states.extend(next_states)
            all_dones.extend(dones)

        if len(all_states) == 0:
            print("  ⚠️ 没有足够数据")
            return

        print(f"  总样本数: {len(all_states):,}")

        # 转换为tensor
        all_states = torch.FloatTensor(all_states).to(DEVICE)
        all_actions = torch.LongTensor(all_actions).to(DEVICE)
        all_rewards = torch.FloatTensor(all_rewards).to(DEVICE)
        all_next_states = torch.FloatTensor(all_next_states).to(DEVICE)
        all_dones = torch.FloatTensor(all_dones).to(DEVICE)

        # 训练DQN
        print("  训练DQN...")
        self.dqn_model = self.train_dqn(
            all_states, all_actions, all_rewards, all_next_states, all_dones
        )

        # 训练A2C
        print("  训练A2C...")
        self.a2c_model = self.train_a2c(
            all_states, all_actions, all_rewards, all_next_states, all_dones
        )

    def train_dqn(self, states, actions, rewards, next_states, dones):
        """训练DQN"""
        state_dim = states.size(1)
        n_actions = 3

        q_net = LSTMNetwork(state_dim, [64, 32], n_actions).to(DEVICE)
        target_net = LSTMNetwork(state_dim, [64, 32], n_actions).to(DEVICE)
        target_net.load_state_dict(q_net.state_dict())

        optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
        buffer = []

        for step in range(TOTAL_TIMESTEPS):
            # 采样
            idx = np.random.randint(0, len(states), BATCH_SIZE_DQN)
            batch_states = states[idx]
            batch_actions = actions[idx]
            batch_rewards = rewards[idx]
            batch_next_states = next_states[idx]
            batch_dones = dones[idx]

            # 当前Q值
            q_values = q_net(batch_states.unsqueeze(1))
            current_q = q_values.gather(1, batch_actions.unsqueeze(1))

            # 目标Q值
            with torch.no_grad():
                next_q = target_net(batch_next_states.unsqueeze(1))
                max_next_q = next_q.max(1)[0]
                target_q = batch_rewards + (1 - batch_dones) * GAMMA * max_next_q

            # 损失
            loss = nn.MSELoss()(current_q.squeeze(), target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新目标网络
            if step % TARGET_UPDATE == 0:
                target_net.load_state_dict(q_net.state_dict())

        return q_net

