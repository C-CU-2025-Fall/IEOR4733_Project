#!/usr/bin/env python3
"""
完整 DRL 训练 - 使用 GPU
实现论文中的 DQN, PPO, A2C 算法
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from datetime import datetime

# 检查 GPU
print("=" * 80)
print("🤖 完整 DRL 训练")
print("=" * 80)
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

# GB10 GPU 兼容性问题，暂时使用 CPU
DEVICE = "cpu"
print(f"设备: {DEVICE} (GB10 GPU 需要 PyTorch nightly)")

# 导入 stable-baselines3
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym
from gymnasium import spaces

# =============================================================================
# 配置
# =============================================================================

DATA_DIR = 'data/futures_processed'
TRAIN_START = '2011-01-01'
TRAIN_END = '2017-06-30'
TEST_START = '2017-07-01'
TEST_END = '2019-12-31'
TRANSACTION_COST = 0.001
LOOKBACK = 50

# 测试合约
TEST_TICKERS = ['ES=F', 'CL=F', 'GC=F']

# 训练参数
TOTAL_TIMESTEPS = 50000  # 快速测试用，完整训练用 100000+

# =============================================================================
# 交易环境
# =============================================================================

class FuturesTradingEnv(gym.Env):
    """期货交易环境"""
    
    def __init__(self, prices, returns, lookback=50, transaction_cost=0.001):
        super().__init__()
        
        self.prices = prices
        self.returns = returns
        self.lookback = lookback
        self.transaction_cost = transaction_cost
        
        # 动作空间: 0=short, 1=neutral, 2=long
        self.action_space = spaces.Discrete(3)
        
        # 观察空间: lookback 天的收益率
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(lookback,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.lookback
        self.position = 0
        self.entry_price = 0
        self.total_reward = 0
        return self._get_obs(), {}
    
    def _get_obs(self):
        """返回过去 lookback 天的标准化收益率"""
        obs = self.returns[self.t-self.lookback:self.t]
        # 标准化
        mean = np.mean(obs)
        std = np.std(obs) + 1e-8
        return ((obs - mean) / std).astype(np.float32)
    
    def step(self, action):
        # 映射动作到仓位: -1, 0, 1
        new_position = action - 1
        
        # 计算交易成本
        trade_cost = abs(new_position - self.position) * self.transaction_cost
        
        # 获取当前收益
        ret = self.returns[self.t]
        
        # 计算奖励 (仓位收益 - 交易成本)
        reward = new_position * ret - trade_cost
        self.total_reward += reward
        
        # 更新状态
        self.position = new_position
        self.t += 1
        
        # 检查是否结束
        terminated = self.t >= len(self.returns) - 1
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {}

# =============================================================================
# 辅助函数
# =============================================================================

def load_data(ticker):
    """加载数据"""
    filepath = os.path.join(DATA_DIR, f"{ticker}.csv")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    train_mask = (df.index >= TRAIN_START) & (df.index < TRAIN_END)
    test_mask = (df.index >= TEST_START) & (df.index <= TEST_END)
    
    train = df[train_mask].copy()
    test = df[test_mask].copy()
    
    return train, test

def calculate_metrics(returns, positions):
    """计算绩效指标"""
    # 组合收益
    portfolio_returns = positions * returns
    
    # 年化指标
    annual_return = np.mean(portfolio_returns) * 252
    annual_std = np.std(portfolio_returns) * np.sqrt(252)
    sharpe = annual_return / annual_std if annual_std > 0 else 0
    
    # 最大回撤
    cumulative = np.cumprod(1 + portfolio_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = np.min(drawdown)
    
    # Sortino
    downside = portfolio_returns[portfolio_returns < 0]
    downside_std = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 1
    sortino = annual_return / downside_std
    
    return {
        'Return': f"{annual_return:.2%}",
        'Sharpe': f"{sharpe:.3f}",
        'Sortino': f"{sortino:.3f}",
        'MDD': f"{max_dd:.2%}",
    }

def evaluate_agent(model, env):
    """评估智能体"""
    obs, _ = env.reset()
    positions = []
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        positions.append(action - 1)  # -1, 0, 1
        obs, _, done, _, _ = env.step(action)
    
    return np.array(positions)

# =============================================================================
# 主训练循环
# =============================================================================

print("\n" + "=" * 80)
print("开始训练...")
print("=" * 80)

results = []

for ticker in TEST_TICKERS:
    print(f"\n{'='*60}")
    print(f"📊 {ticker}")
    print(f"{'='*60}")
    
    # 加载数据
    train, test = load_data(ticker)
    train_returns = train['Returns'].dropna().values
    test_returns = test['Returns'].dropna().values
    
    print(f"训练集: {len(train)} 天")
    print(f"测试集: {len(test)} 天")
    
    # 创建环境
    train_env = FuturesTradingEnv(
        train['Close'].values, train_returns, 
        lookback=LOOKBACK, transaction_cost=TRANSACTION_COST
    )
    test_env = FuturesTradingEnv(
        test['Close'].values, test_returns,
        lookback=LOOKBACK, transaction_cost=TRANSACTION_COST
    )
    
    # === 基线策略 ===
    # Long
    positions = np.ones(len(test_returns[LOOKBACK:]))
    metrics = calculate_metrics(test_returns[LOOKBACK:], positions)
    print(f"\n  Long   | Sharpe: {metrics['Sharpe']:>6} | Return: {metrics['Return']:>8} | MDD: {metrics['MDD']}")
    results.append({'Ticker': ticker, 'Strategy': 'Long', **metrics})
    
    # MACD
    prices = test['Close']
    ema_fast = prices.ewm(span=12).mean()
    ema_slow = prices.ewm(span=26).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=9).mean()
    positions = np.where(macd > signal_line, 1, -1)[LOOKBACK:]
    metrics = calculate_metrics(test_returns[LOOKBACK:LOOKBACK+len(positions)], positions)
    print(f"  MACD   | Sharpe: {metrics['Sharpe']:>6} | Return: {metrics['Return']:>8} | MDD: {metrics['MDD']}")
    results.append({'Ticker': ticker, 'Strategy': 'MACD', **metrics})
    
    # === DRL 模型 ===
    
    # DQN
    print(f"\n  🤖 训练 DQN...")
    model_dqn = DQN(
        "MlpPolicy", train_env,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=100,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        verbose=0,
        device=DEVICE
    )
    model_dqn.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    positions = evaluate_agent(model_dqn, test_env)
    metrics = calculate_metrics(test_returns[LOOKBACK:LOOKBACK+len(positions)], positions)
    print(f"  DQN    | Sharpe: {metrics['Sharpe']:>6} | Return: {metrics['Return']:>8} | MDD: {metrics['MDD']}")
    results.append({'Ticker': ticker, 'Strategy': 'DQN', **metrics})
    
    # PPO
    print(f"  🤖 训练 PPO...")
    model_ppo = PPO(
        "MlpPolicy", train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=0,
        device=DEVICE
    )
    model_ppo.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    positions = evaluate_agent(model_ppo, test_env)
    metrics = calculate_metrics(test_returns[LOOKBACK:LOOKBACK+len(positions)], positions)
    print(f"  PPO    | Sharpe: {metrics['Sharpe']:>6} | Return: {metrics['Return']:>8} | MDD: {metrics['MDD']}")
    results.append({'Ticker': ticker, 'Strategy': 'PPO', **metrics})
    
    # A2C
    print(f"  🤖 训练 A2C...")
    model_a2c = A2C(
        "MlpPolicy", train_env,
        learning_rate=7e-4,
        n_steps=5,
        gamma=0.99,
        verbose=0,
        device=DEVICE
    )
    model_a2c.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    positions = evaluate_agent(model_a2c, test_env)
    metrics = calculate_metrics(test_returns[LOOKBACK:LOOKBACK+len(positions)], positions)
    print(f"  A2C    | Sharpe: {metrics['Sharpe']:>6} | Return: {metrics['Return']:>8} | MDD: {metrics['MDD']}")
    results.append({'Ticker': ticker, 'Strategy': 'A2C', **metrics})

# =============================================================================
# 结果汇总
# =============================================================================

print("\n" + "=" * 80)
print("📊 结果汇总")
print("=" * 80)

df_results = pd.DataFrame(results)

for ticker in TEST_TICKERS:
    print(f"\n{ticker}:")
    ticker_results = df_results[df_results['Ticker'] == ticker]
    for _, row in ticker_results.iterrows():
        print(f"  {row['Strategy']:<6} | Sharpe: {row['Sharpe']:>6} | Return: {row['Return']:>8} | MDD: {row['MDD']}")

# 保存结果
output_file = 'drl_results_gpu.csv'
df_results.to_csv(output_file, index=False)
print(f"\n💾 结果已保存到: {output_file}")

print("\n" + "=" * 80)
print("✅ DRL 训练完成!")
print("=" * 80)
