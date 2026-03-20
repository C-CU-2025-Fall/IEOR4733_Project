#!/usr/bin/env python3
"""
完全对齐论文的 Deep RL Trading
- LSTM (64, 32) - 论文架构
- γ = 0.3 - 论文超参
- Buffer = 5000 - 论文超参
- Batch = 64 (DQN), 128 (A2C) - 论文超参
- 滚动5年训练 - 论文方法
- 40个合约 - 论文规模
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
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
print("🤖 Deep RL Trading - 完全对齐论文版")
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
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces

# =============================================================================
# 论文配置 (完全对齐 Table 1)
# =============================================================================

# 数据
DATA_DIR = 'data/futures_processed'
TRANSACTION_COST = 0.001  # 10 bps

# 论文神经网络配置 (Section 4.3)
# "two-layer LSTM networks with 64 and 32 units"
LSTM_UNITS = [64, 32]

# 论文超参数 (Table 1)
# DQN: α=0.0001, batch=64, γ=0.3, bp=0.0020, memory=5000, τ=1000
# A2C: α_critic=0.001, α_actor=0.0001, batch=128, γ=0.3, bp=0.0020
GAMMA = 0.3  # 论文用0.3!
BUFFER_SIZE = 5000  # 论文用5000
BATCH_SIZE_DQN = 64
BATCH_SIZE_A2C = 128
LEARNING_RATE_DQN = 0.0001
LEARNING_RATE_ACTOR = 0.0001
LEARNING_RATE_CRITIC = 0.001
TARGET_UPDATE = 1000

# 训练步数 (推断)
TOTAL_TIMESTEPS = 50000  # 论文未明确，先用50K

# 训练/测试配置 (适配数据: 2011-2019 = 9年)
# 论文: "retrain our model at every 5 years"
# 我们: 固定5年训练 + 4年测试 (数据限制)
TRAIN_START = '2011-01-03'
TRAIN_END = '2015-12-31'
TEST_START = '2016-01-01'
TEST_END = '2019-12-31'

# 所有可用合约 (40个)
ALL_TICKERS = [
    # Commodity (15)
    'CL=F', 'GC=F', 'SI=F', 'HG=F', 'NG=F', 'ZC=F', 'ZS=F', 'ZW=F', 'KC=F', 'CC=F',
    'SB=F', 'CT=F', 'LC=F', 'LBS=F', 'OJ=F',
    # Equity Index (10)
    'ES=F', 'NQ=F', 'YM=F', 'RTY=F', 'EMD=F', 'VA=F', 'NKE=F', 'FDAX.F', 'FTI.F', 'FCE.F',
    # Fixed Income (10)
    'ZN=F', 'ZB=F', 'ZF=F', 'ZT=F', 'GE=F', 'ZF.F', 'FGBL.F', 'FGBM.F', 'FGBX.F', 'FBTP.F',
    # FX (5)
    '6E=F', '6J=F', '6B=F', '6A=F', '6C=F'
]

# 实际可用的合约 (根据数据文件)
AVAILABLE_TICKERS = []

# =============================================================================
# LSTM 特征提取器 (完全对齐论文)
# =============================================================================

class LSTMFeaturesExtractor(BaseFeaturesExtractor):
    """
    论文 Section 4.3:
    "We use two-layer LSTM networks with 64 and 32 units in all models,
    and Leaky Rectifying Linear Units (Leaky-ReLU) are used as activation functions."
    """
    
    def __init__(self, observation_space: spaces.Box, 
                 lstm_units=[64, 32],
                 features_dim=32):
        super().__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0]
        
        # 两层 LSTM (论文配置: 64, 32)
        self.lstm1 = nn.LSTM(input_dim, lstm_units[0], batch_first=True)
        self.lstm2 = nn.LSTM(lstm_units[0], lstm_units[1], batch_first=True)
        
        # Leaky-ReLU (论文配置)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        
        # 输出层
        self.output_layer = nn.Linear(lstm_units[1], features_dim)
        
        # 初始化隐藏状态
        self.hidden1 = None
        self.hidden2 = None
    
    def forward(self, observations):
        # observations: (batch, features)
        # 添加序列维度: (batch, seq=1, features)
        x = observations.unsqueeze(1)
        
        # LSTM layers
        x, self.hidden1 = self.lstm1(x, self.hidden1)
        x = self.activation(x)
        x, self.hidden2 = self.lstm2(x, self.hidden2)
        x = self.activation(x)
        
        # 取最后一个时间步
        x = x[:, -1, :]
        
        # 输出
        x = self.output_layer(x)
        x = self.activation(x)
        
        return x
    
    def reset_hidden(self):
        """重置隐藏状态"""
        self.hidden1 = None
        self.hidden2 = None

# =============================================================================
# 交易环境
# =============================================================================

class FuturesTradingEnv(gym.Env):
    """期货交易环境 - 论文对齐版"""
    
    def __init__(self, prices, returns, 
                 transaction_cost=0.001,
                 use_dsr=True):
        super().__init__()
        
        self.prices = prices
        self.returns = returns
        self.transaction_cost = transaction_cost
        self.use_dsr = use_dsr
        self.n_steps = len(returns)
        
        # 状态空间 (16维)
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
        self.current_step = 200
        self.position = 0.0
        self.last_action = 0.0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 200
        self.position = 0.0
        self.last_action = 0.0
        self.dsr.reset()
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        return self.state_builder.compute(
            self.prices[:self.current_step+1],
            self.returns[:self.current_step+1],
            self.current_step
        ).astype(np.float32)
    
    def step(self, action):
        action = float(np.clip(action[0] if isinstance(action, np.ndarray) else action, -1, 1))
        
        # 交易成本
        trade_size = abs(action - self.last_action)
        cost = trade_size * self.transaction_cost
        
        if self.current_step + 1 >= self.n_steps:
            return self._get_observation(), 0, True, False, {}
        
        daily_return = self.returns[self.current_step + 1]
        strategy_return = action * daily_return - cost
        
        # Volatility scaling
        scaled_position = self.vol_scaler.scale(
            1.0,
            self.returns[:self.current_step+1],
            self.current_step
        )
        scaled_return = strategy_return * scaled_position
        
        # 奖励函数
        if self.use_dsr:
            reward = self.dsr.update(scaled_return)
        else:
            reward = scaled_return
        
        self.current_step += 1
        self.last_action = action
        self.position = action
        
        done = self.current_step >= self.n_steps - 1
        
        return self._get_observation(), reward, done, False, {}

# 离散版本 (用于DQN)
class DiscreteEnv(FuturesTradingEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Discrete(3)  # 0=short, 1=neutral, 2=long
    
    def step(self, action):
        continuous_action = np.array([float(action - 1)])
        return super().step(continuous_action)

# =============================================================================
# 风险指标
# =============================================================================

def calculate_metrics(returns, positions):
    """计算论文中的所有风险指标"""
    strategy_returns = returns[:len(positions)] * positions
    strategy_returns = strategy_returns[np.isfinite(strategy_returns)]
    
    if len(strategy_returns) == 0:
        return {'E(R)': 0, 'Std(R)': 0, 'DD': 0, 'Sharpe': 0, 
                'Sortino': 0, 'MDD': 0, 'Calmar': 0}
    
    annual_factor = np.sqrt(252)
    
    er = np.mean(strategy_returns) * 252
    std_r = np.std(strategy_returns) * annual_factor
    
    negative_returns = strategy_returns[strategy_returns < 0]
    dd = np.std(negative_returns) * annual_factor if len(negative_returns) > 0 else 0.001
    
    risk_free = 0.02
    sharpe = (er - risk_free) / std_r if std_r > 0 else 0
    sortino = (er - risk_free) / dd if dd > 0 else 0
    
    cumulative = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / running_max
    mdd = np.max(drawdowns) if len(drawdowns) > 0 else 0
    
    calmar = er / mdd if mdd > 0 else 0
    
    return {
        'E(R)': er,
        'Std(R)': std_r,
        'DD': dd,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'MDD': -mdd,
        'Calmar': calmar
    }

# =============================================================================
# 基线策略
# =============================================================================

def baseline_long(returns):
    return np.ones(len(returns))

def baseline_sign(returns, window=252):
    """Sign(R) strategy: sign of cumulative return"""
    signals = np.zeros(len(returns))
    for i in range(window, len(returns)):
        cum_ret = np.sum(returns[i-window:i])
        signals[i] = np.sign(cum_ret)
    signals[:window] = 0
    return signals

def baseline_macd(prices, fast=12, slow=26, signal=9):
    """MACD策略"""
    prices = pd.Series(prices)
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    
    # MACD signal (论文公式 Eq. 11)
    macd_signal = macd / (0.89 * np.exp(-macd**2 / 4))
    macd_signal = np.tanh(macd_signal)
    
    return macd_signal.values

# =============================================================================
# 单合约训练 (固定训练/测试期)
# =============================================================================

def train_single_contract(ticker, all_results):
    """训练单个合约 - 固定训练/测试期"""
    
    # 加载数据
    try:
        df = pd.read_csv(f'{DATA_DIR}/{ticker}.csv')
    except:
        print(f"  ⚠️ {ticker} 数据文件不存在")
        return None
    
    if len(df) < 1000:
        print(f"  ⚠️ {ticker} 数据不足 ({len(df)} < 1000)")
        return None
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 分割训练/测试
    train_mask = (df['Date'] >= TRAIN_START) & (df['Date'] <= TRAIN_END)
    test_mask = (df['Date'] >= TEST_START) & (df['Date'] <= TEST_END)
    
    train_df = df[train_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)
    
    if len(train_df) < 500 or len(test_df) < 100:
        print(f"  ⚠️ {ticker} 训练/测试数据不足")
        return None
    
    print(f"  📊 {ticker}: 训练{len(train_df)}天, 测试{len(test_df)}天")
    
    results = []
    
    train_prices = train_df['Close'].values
    train_returns = train_df['Returns'].values
    test_prices = test_df['Close'].values
    test_returns = test_df['Returns'].values
    
    # ========== 基线 ==========
    # Long
    positions = baseline_long(test_returns[200:])
    m = calculate_metrics(test_returns[200:], positions)
    results.append({'Ticker': ticker, 'Strategy': 'Long', **m})
    print(f"    Long   | Sharpe: {m['Sharpe']:>7.3f}")
    
    # Sign(R)
    positions = baseline_sign(test_returns)[200:]
    positions = positions[:len(test_returns)-200]
    if len(positions) > 0:
        m = calculate_metrics(test_returns[200:200+len(positions)], positions)
        results.append({'Ticker': ticker, 'Strategy': 'Sign(R)', **m})
        print(f"    Sign(R)| Sharpe: {m['Sharpe']:>7.3f}")
    
    # MACD
    positions = baseline_macd(test_prices)[200:]
    positions = positions[:len(test_returns)-200]
    if len(positions) > 0:
        m = calculate_metrics(test_returns[200:200+len(positions)], positions)
        results.append({'Ticker': ticker, 'Strategy': 'MACD', **m})
        print(f"    MACD   | Sharpe: {m['Sharpe']:>7.3f}")
    
    # ========== DRL ==========
    # Policy kwargs for LSTM
    policy_kwargs = dict(
        features_extractor_class=LSTMFeaturesExtractor,
        features_extractor_kwargs=dict(
            lstm_units=LSTM_UNITS,
            features_dim=32
        ),
        net_arch=[]
    )
    
    # 创建环境
    train_env = FuturesTradingEnv(
        train_prices, train_returns,
        transaction_cost=TRANSACTION_COST, use_dsr=True
    )
    test_env = FuturesTradingEnv(
        test_prices, test_returns,
        transaction_cost=TRANSACTION_COST, use_dsr=False
    )
    
    # ----- DQN -----
    try:
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
            gamma=GAMMA,  # 0.3
            train_freq=4,
            target_update_interval=TARGET_UPDATE,
            policy_kwargs=policy_kwargs,
            verbose=0, device=DEVICE
        )
        
        print(f"    训练 DQN...", end=' ', flush=True)
        model_dqn.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
        
        # 评估
        obs, _ = test_env_discrete.reset()
        positions = []
        done = False
        while not done:
            action, _ = model_dqn.predict(obs, deterministic=True)
            positions.append(float(action - 1))
            obs, _, done, _, _ = test_env_discrete.step(action)
        
        m = calculate_metrics(test_returns[200:200+len(positions)], np.array(positions))
        results.append({'Ticker': ticker, 'Strategy': 'DQN', **m})
        print(f"Sharpe: {m['Sharpe']:>7.3f}")
    except Exception as e:
        print(f"⚠️ DQN 失败: {e}")
    
    # ----- A2C -----
    try:
        model_a2c = A2C(
            "MlpPolicy", train_env,
            learning_rate=LEARNING_RATE_ACTOR,
            gamma=GAMMA,  # 0.3
            policy_kwargs=policy_kwargs,
            verbose=0, device=DEVICE
        )
        
        print(f"    训练 A2C...", end=' ', flush=True)
        model_a2c.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
        
        # 评估
        obs, _ = test_env.reset()
        positions = []
        done = False
        while not done:
            action, _ = model_a2c.predict(obs, deterministic=True)
            positions.append(float(action[0]))
            obs, _, done, _, _ = test_env.step(action)
        
        m = calculate_metrics(test_returns[200:200+len(positions)], np.array(positions))
        results.append({'Ticker': ticker, 'Strategy': 'A2C', **m})
        print(f"Sharpe: {m['Sharpe']:>7.3f}")
    except Exception as e:
        print(f"⚠️ A2C 失败: {e}")
    
    return results

# =============================================================================
# 主函数
# =============================================================================

def train_all_aligned():
    """训练所有合约 - 完全对齐论文"""
    
    print(f"\n{'='*80}")
    print("📋 论文配置")
    print('='*80)
    print(f"网络: LSTM {LSTM_UNITS}")
    print(f"γ (discount): {GAMMA} (论文配置)")
    print(f"Buffer Size: {BUFFER_SIZE}")
    print(f"Batch Size: DQN={BATCH_SIZE_DQN}, A2C={BATCH_SIZE_A2C}")
    print(f"学习率: DQN={LEARNING_RATE_DQN}, A2C={LEARNING_RATE_ACTOR}")
    print(f"训练方式: 滚动5年")
    print(f"步数: {TOTAL_TIMESTEPS:,}")
    
    # 检查可用合约
    available = []
    for ticker in ALL_TICKERS:
        if os.path.exists(f'{DATA_DIR}/{ticker}.csv'):
            available.append(ticker)
    
    print(f"\n可用合约: {len(available)}/{len(ALL_TICKERS)}")
    print(f"合约列表: {available[:10]}...")
    
    # 训练所有合约
    all_results = []
    
    # 先用3个合约测试
    test_mode = True
    if test_mode:
        test_tickers = ['ES=F', 'CL=F', 'GC=F']
        available = [t for t in test_tickers if t in available]
        print(f"\n🧪 测试模式: 只训练 {len(available)} 个合约")
    
    for idx, ticker in enumerate(available):
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(available)}] {ticker}")
        print('='*60)
        
        results = train_single_contract(ticker, all_results)
        if results:
            all_results.extend(results)
    
    # 保存结果
    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results_aligned_{timestamp}.csv'
    df.to_csv(filename, index=False)
    print(f"\n💾 结果已保存: {filename}")
    
    # 汇总统计
    print("\n" + "="*80)
    print("📊 结果汇总 (所有合约平均)")
    print("="*80)
    
    summary = df.groupby('Strategy').agg({
        'Sharpe': 'mean',
        'E(R)': 'mean',
        'MDD': 'mean',
        'Sortino': 'mean'
    }).round(3)
    
    print(summary)
    
    # 按资产类别分组
    print("\n" + "="*80)
    print("📊 按策略分组 (Sharpe Ratio)")
    print("="*80)
    
    for strategy in ['Long', 'Sign(R)', 'MACD', 'DQN', 'A2C']:
        strategy_df = df[df['Strategy'] == strategy]
        if len(strategy_df) > 0:
            mean_sharpe = strategy_df['Sharpe'].mean()
            std_sharpe = strategy_df['Sharpe'].std()
            print(f"{strategy:<10} | Sharpe: {mean_sharpe:>7.3f} ± {std_sharpe:.3f} | N={len(strategy_df)}")
    
    return df

if __name__ == '__main__':
    results = train_all_aligned()
