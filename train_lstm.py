#!/usr/bin/env python3
"""
完全对齐论文的 Deep RL Trading - LSTM版本
论文使用 2层LSTM (64, 32 units) + Leaky-ReLU
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
import json
import psutil
import time
from tqdm import tqdm

# 导入核心组件
from paper_components import (
    DifferentialSharpeRatio,
    MultiTimeScaleState,
    VolatilityScaler
)

# GPU 设置
import torch
import torch.nn as nn
print("=" * 80)
print("🤖 Deep RL Trading - LSTM版本 (论文对齐)")
print("=" * 80)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"设备: {DEVICE}")

# DRL 导入
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces

# =============================================================================
# 配置 (完全对齐论文)
# =============================================================================

# 数据
DATA_DIR = 'data/futures_processed'
TRAIN_START = '2011-01-03'
TRAIN_END = '2015-12-31'
TEST_START = '2016-01-01'
TEST_END = '2019-12-31'
TRANSACTION_COST = 0.001  # 10 bps

# 论文神经网络配置
# LSTM: 2层, 64和32 units, Leaky-ReLU
USE_LSTM = True
LSTM_UNITS = [64, 32]
MLP_LAYERS = [256, 256, 256]  # MLP备选

# 论文超参数 (Table 1)
# DQN: α=0.0001, batch=64, γ=0.3, bp=0.0020, memory=5000, τ=1000
# A2C: α_critic=0.001, α_actor=0.0001, batch=128, γ=0.3, bp=0.0020

TOTAL_TIMESTEPS = 100000
BATCH_SIZE_DQN = 64
BATCH_SIZE_A2C = 128
BUFFER_SIZE = 5000  # 论文用5000
LEARNING_RATE_DQN = 0.0001
LEARNING_RATE_ACTOR = 0.0001
LEARNING_RATE_CRITIC = 0.001
GAMMA = 0.3  # 论文用0.3!
TARGET_UPDATE = 1000  # τ

# 测试合约
TEST_TICKERS = ['ES=F', 'CL=F', 'GC=F']

# =============================================================================
# LSTM 特征提取器 (对齐论文)
# =============================================================================

class LSTMFeaturesExtractor(BaseFeaturesExtractor):
    """
    论文使用的 LSTM 特征提取器
    2层 LSTM: 64 -> 32 units
    Leaky-ReLU 激活
    """
    
    def __init__(self, observation_space: spaces.Box, 
                 lstm_units=[64, 32],
                 features_dim=32):
        super().__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0]
        
        # 2层 LSTM (论文配置)
        self.lstm1 = nn.LSTM(input_dim, lstm_units[0], batch_first=True)
        self.lstm2 = nn.LSTM(lstm_units[0], lstm_units[1], batch_first=True)
        
        # Leaky-ReLU (论文配置)
        self.activation = nn.LeakyReLU(0.01)
        
        # 输出层
        self.output_layer = nn.Linear(lstm_units[1], features_dim)
    
    def forward(self, observations):
        # observations: (batch, features)
        # 需要添加序列维度: (batch, seq=1, features)
        x = observations.unsqueeze(1)
        
        # LSTM layers
        x, _ = self.lstm1(x)
        x = self.activation(x)
        x, _ = self.lstm2(x)
        x = self.activation(x)
        
        # 取最后一个时间步
        x = x[:, -1, :]
        
        # 输出
        x = self.output_layer(x)
        x = self.activation(x)
        
        return x

class MLPFeaturesExtractor(BaseFeaturesExtractor):
    """标准 MLP 特征提取器 (备选)"""
    
    def __init__(self, observation_space: spaces.Box, 
                 hidden_layers=[256, 256, 256],
                 features_dim=256):
        super().__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0]
        
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.LeakyReLU(0.01))  # 论文用Leaky-ReLU
            prev_dim = h
        
        self.net = nn.Sequential(*layers)
        self.features_dim = hidden_layers[-1]
    
    def forward(self, observations):
        return self.net(observations)

# =============================================================================
# 交易环境 (复用 paper_components)
# =============================================================================

class FuturesTradingEnvLSTM(gym.Env):
    """
    期货交易环境 - LSTM版本
    使用论文的核心组件
    """
    
    def __init__(self, prices, returns, 
                 transaction_cost=0.001,
                 use_dsr=True):
        super().__init__()
        
        self.prices = prices
        self.returns = returns
        self.transaction_cost = transaction_cost
        self.use_dsr = use_dsr
        self.n_steps = len(returns)
        
        # 状态空间 (16维 - 多时间尺度)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32
        )
        
        # 动作空间: 连续 [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # 核心组件
        self.state_builder = MultiTimeScaleState()
        self.dsr = DifferentialSharpeRatio(eta=0.01)
        self.vol_scaler = VolatilityScaler(target_vol=0.10)
        
        # 状态变量
        self.current_step = 200  # 需要足够历史计算特征
        self.position = 0.0
        self.last_action = 0.0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 200
        self.position = 0.0
        self.last_action = 0.0
        # MultiTimeScaleState不需要reset，每次compute都是独立的
        self.dsr.reset()
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """获取多时间尺度状态"""
        return self.state_builder.compute(
            self.prices[:self.current_step+1],
            self.returns[:self.current_step+1],
            self.current_step
        ).astype(np.float32)
    
    def step(self, action):
        action = float(np.clip(action[0] if isinstance(action, np.ndarray) else action, -1, 1))
        
        # 计算交易成本
        trade_size = abs(action - self.last_action)
        cost = trade_size * self.transaction_cost
        
        # 获取下一天的收益
        if self.current_step + 1 >= self.n_steps:
            return self._get_observation(), 0, True, False, {}
        
        daily_return = self.returns[self.current_step + 1]
        
        # 策略收益
        strategy_return = action * daily_return - cost
        
        # Volatility scaling (对齐论文)
        scaled_position = self.vol_scaler.scale(
            1.0,  # 单位仓位
            self.returns[:self.current_step+1],
            self.current_step
        )
        scaled_return = strategy_return * scaled_position
        
        # 奖励函数
        if self.use_dsr:
            reward = self.dsr.update(scaled_return)
        else:
            reward = scaled_return
        
        # 更新状态
        self.current_step += 1
        self.last_action = action
        self.position = action
        
        done = self.current_step >= self.n_steps - 1
        
        return self._get_observation(), reward, done, False, {}

# 离散版本 (用于DQN)
class DiscreteEnv(FuturesTradingEnvLSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Discrete(3)  # 0=short, 1=neutral, 2=long
    
    def step(self, action):
        continuous_action = np.array([float(action - 1)])  # -1, 0, 1
        return super().step(continuous_action)

# =============================================================================
# 风险指标计算 (完全对齐论文)
# =============================================================================

def calculate_metrics_paper(returns, positions):
    """计算论文中的所有风险指标"""
    # 策略收益
    strategy_returns = returns[:len(positions)] * positions
    strategy_returns = strategy_returns[np.isfinite(strategy_returns)]
    
    if len(strategy_returns) == 0:
        return {'E(R)': 0, 'Std(R)': 0, 'DD': 0, 'Sharpe': 0, 
                'Sortino': 0, 'MDD': 0, 'Calmar': 0}
    
    # 年化
    annual_factor = np.sqrt(252)
    
    # E(R) - 年化期望收益
    er = np.mean(strategy_returns) * 252
    
    # Std(R) - 年化标准差
    std_r = np.std(strategy_returns) * annual_factor
    
    # Downside Deviation (DD) - 只计算负收益的标准差
    negative_returns = strategy_returns[strategy_returns < 0]
    dd = np.std(negative_returns) * annual_factor if len(negative_returns) > 0 else 0.001
    
    # Sharpe Ratio
    risk_free = 0.02  # 假设无风险利率2%
    sharpe = (er - risk_free) / std_r if std_r > 0 else 0
    
    # Sortino Ratio
    sortino = (er - risk_free) / dd if dd > 0 else 0
    
    # Maximum Drawdown
    cumulative = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / running_max
    mdd = np.max(drawdowns) if len(drawdowns) > 0 else 0
    
    # Calmar Ratio
    calmar = er / mdd if mdd > 0 else 0
    
    return {
        'E(R)': er,
        'Std(R)': std_r,
        'DD': dd,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'MDD': -mdd,  # 负数表示
        'Calmar': calmar
    }

# =============================================================================
# 基线策略
# =============================================================================

def baseline_long(returns):
    return np.ones(len(returns))

def baseline_macd(prices, fast=12, slow=26, signal=9):
    """MACD策略"""
    prices = pd.Series(prices)
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    
    # MACD signal (论文公式)
    macd_signal = macd / (0.89 * np.exp(-macd**2 / 4))
    macd_signal = np.tanh(macd_signal)  # 限制在 [-1, 1]
    
    return macd_signal.values

# =============================================================================
# 进度条回调
# =============================================================================

class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None
    
    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="训练", 
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
    
    def _on_step(self):
        if self.pbar:
            self.pbar.update(1)
        return True
    
    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()

# =============================================================================
# 主训练函数
# =============================================================================

def train_with_lstm(ticker, use_lstm=True):
    """训练单个合约"""
    
    # 加载数据
    df = pd.read_csv(f'{DATA_DIR}/{ticker}.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 分割
    train = df[(df['Date'] >= TRAIN_START) & (df['Date'] <= TRAIN_END)]
    test = df[(df['Date'] >= TEST_START) & (df['Date'] <= TEST_END)]
    
    train_prices = train['Close'].values
    train_returns = train['Returns'].values
    test_prices = test['Close'].values
    test_returns = test['Returns'].values
    
    print(f"训练: {len(train)} 天, 测试: {len(test)} 天")
    
    # 结果存储
    results = []
    
    # ========== 基线 ==========
    print(f"\n【基线】")
    
    # Long
    positions = baseline_long(test_returns[200:])
    m = calculate_metrics_paper(test_returns[200:], positions)
    print(f"  Long   | Sharpe: {m['Sharpe']:.3f} | Return: {m['E(R)']:.2%} | MDD: {m['MDD']:.2%}")
    results.append({'Ticker': ticker, 'Strategy': 'Long', **m})
    
    # MACD
    positions = baseline_macd(test_prices)[200:]
    positions = positions[:len(test_returns)-200]
    m = calculate_metrics_paper(test_returns[200:200+len(positions)], positions)
    print(f"  MACD   | Sharpe: {m['Sharpe']:.3f} | Return: {m['E(R)']:.2%} | MDD: {m['MDD']:.2%}")
    results.append({'Ticker': ticker, 'Strategy': 'MACD', **m})
    
    # ========== DRL ==========
    print(f"\n【DRL】({'LSTM' if use_lstm else 'MLP'})")
    
    # 选择特征提取器
    if use_lstm:
        policy_kwargs = dict(
            features_extractor_class=LSTMFeaturesExtractor,
            features_extractor_kwargs=dict(
                lstm_units=LSTM_UNITS,
                features_dim=32
            ),
            net_arch=[]  # LSTM已经提取特征，不需要额外网络
        )
    else:
        policy_kwargs = dict(
            features_extractor_class=MLPFeaturesExtractor,
            features_extractor_kwargs=dict(
                hidden_layers=MLP_LAYERS,
                features_dim=256
            ),
            net_arch=[]
        )
    
    # 创建环境
    train_env = FuturesTradingEnvLSTM(
        train_prices, train_returns,
        transaction_cost=TRANSACTION_COST, use_dsr=True
    )
    test_env = FuturesTradingEnvLSTM(
        test_prices, test_returns,
        transaction_cost=TRANSACTION_COST, use_dsr=False
    )
    
    # ----- DQN (离散) -----
    print(f"\n  🤖 DQN...")
    
    train_env_discrete = DiscreteEnv(
        train_prices, train_returns,
        transaction_cost=TRANSACTION_COST, use_dsr=True
    )
    test_env_discrete = DiscreteEnv(
        test_prices, test_returns,
        transaction_cost=TRANSACTION_COST, use_dsr=False
    )
    
    model_dqn = DQN(
        "MlpPolicy", train_env_discrete,
        learning_rate=LEARNING_RATE_DQN,
        buffer_size=BUFFER_SIZE,
        learning_starts=1000,
        batch_size=BATCH_SIZE_DQN,
        gamma=GAMMA,  # 论文用0.3
        train_freq=4,
        target_update_interval=TARGET_UPDATE,
        policy_kwargs=policy_kwargs,
        verbose=0, device=DEVICE
    )
    
    model_dqn.learn(total_timesteps=TOTAL_TIMESTEPS, 
                    callback=TqdmCallback(TOTAL_TIMESTEPS),
                    progress_bar=False)
    
    # 评估 DQN
    obs, _ = test_env_discrete.reset()
    positions = []
    done = False
    while not done:
        action, _ = model_dqn.predict(obs, deterministic=True)
        positions.append(float(action - 1))
        obs, _, done, _, _ = test_env_discrete.step(action)
    
    m = calculate_metrics_paper(test_returns[200:200+len(positions)], np.array(positions))
    print(f"\n  DQN    | Sharpe: {m['Sharpe']:.3f} | Return: {m['E(R)']:.2%} | MDD: {m['MDD']:.2%}")
    results.append({'Ticker': ticker, 'Strategy': 'DQN', **m})
    
    # ----- PPO (连续) -----
    print(f"\n  🤖 PPO...")
    
    model_ppo = PPO(
        "MlpPolicy", train_env,
        learning_rate=LEARNING_RATE_ACTOR,
        batch_size=BATCH_SIZE_A2C,
        gamma=GAMMA,
        policy_kwargs=policy_kwargs,
        verbose=0, device=DEVICE
    )
    
    model_ppo.learn(total_timesteps=TOTAL_TIMESTEPS,
                    callback=TqdmCallback(TOTAL_TIMESTEPS),
                    progress_bar=False)
    
    # 评估 PPO
    obs, _ = test_env.reset()
    positions = []
    done = False
    while not done:
        action, _ = model_ppo.predict(obs, deterministic=True)
        positions.append(float(action[0]))
        obs, _, done, _, _ = test_env.step(action)
    
    m = calculate_metrics_paper(test_returns[200:200+len(positions)], np.array(positions))
    print(f"\n  PPO    | Sharpe: {m['Sharpe']:.3f} | Return: {m['E(R)']:.2%} | MDD: {m['MDD']:.2%}")
    results.append({'Ticker': ticker, 'Strategy': 'PPO', **m})
    
    # ----- A2C (连续) -----
    print(f"\n  🤖 A2C...")
    
    model_a2c = A2C(
        "MlpPolicy", train_env,
        learning_rate=LEARNING_RATE_ACTOR,
        gamma=GAMMA,
        policy_kwargs=policy_kwargs,
        verbose=0, device=DEVICE
    )
    
    model_a2c.learn(total_timesteps=TOTAL_TIMESTEPS,
                    callback=TqdmCallback(TOTAL_TIMESTEPS),
                    progress_bar=False)
    
    # 评估 A2C
    obs, _ = test_env.reset()
    positions = []
    done = False
    while not done:
        action, _ = model_a2c.predict(obs, deterministic=True)
        positions.append(float(action[0]))
        obs, _, done, _, _ = test_env.step(action)
    
    m = calculate_metrics_paper(test_returns[200:200+len(positions)], np.array(positions))
    print(f"\n  A2C    | Sharpe: {m['Sharpe']:.3f} | Return: {m['E(R)']:.2%} | MDD: {m['MDD']:.2%}")
    results.append({'Ticker': ticker, 'Strategy': 'A2C', **m})
    
    return results

# =============================================================================
# 主函数
# =============================================================================

def train_all():
    """训练所有合约"""
    
    print(f"\n训练期: {TRAIN_START} ~ {TRAIN_END}")
    print(f"测试期: {TEST_START} ~ {TEST_END}")
    print(f"步数: {TOTAL_TIMESTEPS:,}")
    print(f"网络: {'LSTM ' + str(LSTM_UNITS) if USE_LSTM else 'MLP ' + str(MLP_LAYERS)}")
    print(f"γ (discount): {GAMMA} (论文配置)")
    
    all_results = []
    
    for ticker in TEST_TICKERS:
        print(f"\n{'='*60}")
        print(f"📊 {ticker}")
        print('='*60)
        
        try:
            results = train_with_lstm(ticker, use_lstm=USE_LSTM)
            all_results.extend(results)
        except Exception as e:
            print(f"❌ {ticker} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存结果
    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results_lstm_{timestamp}.csv'
    df.to_csv(filename, index=False)
    print(f"\n💾 结果已保存: {filename}")
    
    # 打印汇总
    print("\n" + "="*80)
    print("📊 结果汇总")
    print("="*80)
    
    for ticker in TEST_TICKERS:
        print(f"\n【{ticker}】")
        ticker_results = df[df['Ticker'] == ticker]
        for _, row in ticker_results.iterrows():
            print(f"  {row['Strategy']:<8} | Sharpe: {row['Sharpe']:>7.3f} | "
                  f"Return: {row['E(R)']:>7.2%} | MDD: {row['MDD']:>7.2%}")
    
    return df

if __name__ == '__main__':
    results = train_all()
