#!/usr/bin/env python3
"""
Deep Reinforcement Learning for Trading
兼容本地 GPU (GB10) 和 Colab GPU

论文: "Deep Reinforcement Learning for Trading" (Zhang, Zohren, Roberts, 2019)
算法: DQN, PPO, A2C
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
import json

# =============================================================================
# GPU 检测和设置
# =============================================================================

import torch

def setup_device():
    """检测并设置最佳设备"""
    print("=" * 80)
    print("🤖 Deep RL Trading - 设备检测")
    print("=" * 80)
    
    print(f"PyTorch 版本: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA 可用")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return "cuda"
    else:
        print("⚠️ CUDA 不可用，使用 CPU")
        return "cpu"

DEVICE = setup_device()

# 如果 GPU 有兼容性问题，强制使用 CPU
# =============================================================================

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gymnasium import spaces

# =============================================================================
# 配置
# =============================================================================

# 数据配置
DATA_DIR = 'data/futures_processed'
TRAIN_START = '2011-01-01'
TRAIN_END = '2017-06-30'
TEST_START = '2017-07-01'
TEST_END = '2019-12-31'
TRANSACTION_COST = 0.001  # 10 bps
LOOKBACK = 50

# 测试合约
TEST_TICKERS = ['ES=F', 'CL=F', 'GC=F']

# 训练参数 (根据设备调整)
if DEVICE == "cuda":
    TOTAL_TIMESTEPS = 100000  # GPU: 更长训练
    HIDDEN_SIZE = 128
else:
    TOTAL_TIMESTEPS = 30000   # CPU: 较短训练
    HIDDEN_SIZE = 64

print(f"\n设备: {DEVICE}")
print(f"训练步数: {TOTAL_TIMESTEPS}")
print(f"测试合约: {TEST_TICKERS}")

# =============================================================================
# 交易环境
# =============================================================================

class FuturesTradingEnv(gym.Env):
    """
    期货交易环境
    State: 过去 LOOKBACK 天的归一化收益率
    Action: 0=short, 1=neutral, 2=long
    Reward: 组合收益 - 交易成本
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, prices, returns, lookback=50, transaction_cost=0.001):
        super().__init__()
        
        self.prices = prices
        self.returns = returns.astype(np.float32)
        self.lookback = lookback
        self.transaction_cost = transaction_cost
        
        # 动作空间
        self.action_space = spaces.Discrete(3)
        
        # 观察空间
        self.observation_space = spaces.Box(
            low=-10, high=10, 
            shape=(lookback,), 
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.lookback
        self.position = 0
        self.done = False
        self.total_reward = 0.0
        self.trades = 0
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """归一化的历史收益率"""
        window = self.returns[self.t - self.lookback:self.t]
        mean = np.mean(window)
        std = np.std(window) + 1e-8
        normalized = (window - mean) / std
        return normalized.astype(np.float32)
    
    def step(self, action):
        # 转换动作: 0,1,2 -> -1,0,1
        new_position = action - 1
        
        # 交易成本
        trade_size = abs(new_position - self.position)
        trade_cost = trade_size * self.transaction_cost
        
        # 计算收益
        ret = self.returns[self.t]
        portfolio_return = new_position * ret - trade_cost
        
        # 奖励 (放大以便训练)
        reward = portfolio_return * 100
        
        # 更新状态
        if trade_size > 0:
            self.trades += 1
        self.position = new_position
        self.t += 1
        self.total_reward += reward
        
        # 检查结束
        if self.t >= len(self.returns) - 1:
            self.done = True
        
        return self._get_obs(), reward, self.done, False, {}

# =============================================================================
# 辅助函数
# =============================================================================

def load_data(ticker):
    """加载训练和测试数据"""
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
    
    # Calmar
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
    
    return {
        'Annual Return': annual_return,
        'Annual Std': annual_std,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Max DD': max_dd,
        'Calmar': calmar
    }

def evaluate_model(model, env):
    """评估模型并返回仓位"""
    obs, _ = env.reset()
    positions = []
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        positions.append(action - 1)  # -1, 0, 1
        obs, _, done, _, _ = env.step(action)
    
    return np.array(positions)

# =============================================================================
# 训练回调
# =============================================================================

class ProgressCallback(BaseCallback):
    """训练进度回调"""
    
    def __init__(self, check_freq=5000, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
    
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            print(f"    Step {self.n_calls}/{TOTAL_TIMESTEPS}")
        return True

# =============================================================================
# 主训练函数
# =============================================================================

def train_all():
    """训练所有模型"""
    
    print("\n" + "=" * 80)
    print("🚀 开始训练")
    print("=" * 80)
    
    results = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for ticker in TEST_TICKERS:
        print(f"\n{'='*70}")
        print(f"📊 {ticker}")
        print(f"{'='*70}")
        
        # 加载数据
        train, test = load_data(ticker)
        train_returns = train['Returns'].dropna().values
        test_returns = test['Returns'].dropna().values
        test_prices = test['Close']
        
        print(f"训练集: {len(train)} 天 ({train.index[0].strftime('%Y-%m-%d')} ~ {train.index[-1].strftime('%Y-%m-%d')})")
        print(f"测试集: {len(test)} 天 ({test.index[0].strftime('%Y-%m-%d')} ~ {test.index[-1].strftime('%Y-%m-%d')})")
        
        # 创建环境
        train_env = FuturesTradingEnv(
            train['Close'].values, train_returns,
            lookback=LOOKBACK, transaction_cost=TRANSACTION_COST
        )
        test_env = FuturesTradingEnv(
            test['Close'].values, test_returns,
            lookback=LOOKBACK, transaction_cost=TRANSACTION_COST
        )
        
        # ===== 基线策略 =====
        
        # 1. Long
        print(f"\n【基线策略】")
        positions = np.ones(len(test_returns[LOOKBACK:]))
        metrics = calculate_metrics(test_returns[LOOKBACK:], positions)
        print(f"  Long   | Sharpe: {metrics['Sharpe']:>6.3f} | Return: {metrics['Annual Return']:>7.2%} | MDD: {metrics['Max DD']:>7.2%}")
        results.append({
            'Ticker': ticker, 'Strategy': 'Long',
            'Sharpe': metrics['Sharpe'],
            'Return': metrics['Annual Return'],
            'MDD': metrics['Max DD'],
            'Sortino': metrics['Sortino']
        })
        
        # 2. MACD
        ema_fast = test_prices.ewm(span=12).mean()
        ema_slow = test_prices.ewm(span=26).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=9).mean()
        positions = np.where(macd > signal_line, 1, -1)[LOOKBACK:]
        metrics = calculate_metrics(test_returns[LOOKBACK:LOOKBACK+len(positions)], positions)
        print(f"  MACD   | Sharpe: {metrics['Sharpe']:>6.3f} | Return: {metrics['Annual Return']:>7.2%} | MDD: {metrics['Max DD']:>7.2%}")
        results.append({
            'Ticker': ticker, 'Strategy': 'MACD',
            'Sharpe': metrics['Sharpe'],
            'Return': metrics['Annual Return'],
            'MDD': metrics['Max DD'],
            'Sortino': metrics['Sortino']
        })
        
        # ===== DRL 模型 =====
        print(f"\n【DRL 训练】")
        
        # 3. DQN
        print(f"  🤖 DQN...", end=" ", flush=True)
        model_dqn = DQN(
            "MlpPolicy", train_env,
            learning_rate=1e-3,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            target_update_interval=100,
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
            policy_kwargs=dict(net_arch=[HIDDEN_SIZE, HIDDEN_SIZE]),
            verbose=0,
            device=DEVICE
        )
        model_dqn.learn(total_timesteps=TOTAL_TIMESTEPS, callback=ProgressCallback(check_freq=10000), progress_bar=False)
        positions = evaluate_model(model_dqn, test_env)
        metrics = calculate_metrics(test_returns[LOOKBACK:LOOKBACK+len(positions)], positions)
        print(f"Sharpe: {metrics['Sharpe']:>6.3f} | Return: {metrics['Annual Return']:>7.2%} | MDD: {metrics['Max DD']:>7.2%}")
        results.append({
            'Ticker': ticker, 'Strategy': 'DQN',
            'Sharpe': metrics['Sharpe'],
            'Return': metrics['Annual Return'],
            'MDD': metrics['Max DD'],
            'Sortino': metrics['Sortino']
        })
        
        # 4. PPO
        print(f"  🤖 PPO...", end=" ", flush=True)
        model_ppo = PPO(
            "MlpPolicy", train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            policy_kwargs=dict(net_arch=[HIDDEN_SIZE, HIDDEN_SIZE]),
            verbose=0,
            device=DEVICE
        )
        model_ppo.learn(total_timesteps=TOTAL_TIMESTEPS, callback=ProgressCallback(check_freq=10000), progress_bar=False)
        positions = evaluate_model(model_ppo, test_env)
        metrics = calculate_metrics(test_returns[LOOKBACK:LOOKBACK+len(positions)], positions)
        print(f"Sharpe: {metrics['Sharpe']:>6.3f} | Return: {metrics['Annual Return']:>7.2%} | MDD: {metrics['Max DD']:>7.2%}")
        results.append({
            'Ticker': ticker, 'Strategy': 'PPO',
            'Sharpe': metrics['Sharpe'],
            'Return': metrics['Annual Return'],
            'MDD': metrics['Max DD'],
            'Sortino': metrics['Sortino']
        })
        
        # 5. A2C
        print(f"  🤖 A2C...", end=" ", flush=True)
        model_a2c = A2C(
            "MlpPolicy", train_env,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            policy_kwargs=dict(net_arch=[HIDDEN_SIZE, HIDDEN_SIZE]),
            verbose=0,
            device=DEVICE
        )
        model_a2c.learn(total_timesteps=TOTAL_TIMESTEPS, callback=ProgressCallback(check_freq=10000), progress_bar=False)
        positions = evaluate_model(model_a2c, test_env)
        metrics = calculate_metrics(test_returns[LOOKBACK:LOOKBACK+len(positions)], positions)
        print(f"Sharpe: {metrics['Sharpe']:>6.3f} | Return: {metrics['Annual Return']:>7.2%} | MDD: {metrics['Max DD']:>7.2%}")
        results.append({
            'Ticker': ticker, 'Strategy': 'A2C',
            'Sharpe': metrics['Sharpe'],
            'Return': metrics['Annual Return'],
            'MDD': metrics['Max DD'],
            'Sortino': metrics['Sortino']
        })
        
        # 保存模型
        model_dir = f"models/{timestamp}"
        os.makedirs(model_dir, exist_ok=True)
        model_dqn.save(f"{model_dir}/{ticker}_dqn")
        model_ppo.save(f"{model_dir}/{ticker}_ppo")
        model_a2c.save(f"{model_dir}/{ticker}_a2c")
        print(f"  💾 模型已保存到 {model_dir}/")
    
    return results, timestamp

# =============================================================================
# 结果汇总
# =============================================================================

def summarize_results(results, timestamp):
    """汇总并保存结果"""
    
    print("\n" + "=" * 80)
    print("📊 结果汇总")
    print("=" * 80)
    
    df = pd.DataFrame(results)
    
    # 按合约显示
    for ticker in TEST_TICKERS:
        print(f"\n{ticker}:")
        ticker_df = df[df['Ticker'] == ticker]
        for _, row in ticker_df.iterrows():
            print(f"  {row['Strategy']:<6} | Sharpe: {row['Sharpe']:>6.3f} | Return: {row['Return']:>7.2%} | Sortino: {row['Sortino']:>6.3f} | MDD: {row['MDD']:>7.2%}")
    
    # 最佳策略
    print(f"\n【最佳策略】")
    best_per_ticker = df.loc[df.groupby('Ticker')['Sharpe'].idxmax()]
    for _, row in best_per_ticker.iterrows():
        print(f"  {row['Ticker']}: {row['Strategy']} (Sharpe: {row['Sharpe']:.3f})")
    
    # 保存结果
    output_file = f'drl_results_{timestamp}.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 结果已保存到: {output_file}")
    
    # 保存 JSON
    with open(f'drl_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return df

# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🤖 Deep Reinforcement Learning for Trading")
    print("论文: Zhang, Zohren, Roberts (2019)")
    print("=" * 80)
    
    # 训练
    results, timestamp = train_all()
    
    # 汇总
    df_results = summarize_results(results, timestamp)
    
    print("\n" + "=" * 80)
    print("✅ 训练完成!")
    print("=" * 80)
    print(f"""
下一步建议:
1. 调整超参数 (learning_rate, batch_size, network architecture)
2. 增加训练步数 (TOTAL_TIMESTEPS = 500000+)
3. 扩展到所有 45 个合约
4. 实现论文中的 Differential Sharpe Ratio 奖励
5. 添加 Walk-Forward 验证
""")
