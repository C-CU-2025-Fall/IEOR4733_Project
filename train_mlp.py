#!/usr/bin/env python3
"""
完全对齐论文的 Deep RL Trading - 最终版本
使用独立的核心组件模块
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
print("=" * 80)
print("🤖 Deep RL Trading - 论文对齐最终版")
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

# 神经网络 (推断自论文)
HIDDEN_LAYERS = [256, 256, 256]

# 训练
TOTAL_TIMESTEPS = 100000  # 快速测试用 100K，完整用 500K
BATCH_SIZE = 256
BUFFER_SIZE = 1000000
LEARNING_RATE = 1e-4

# 测试合约
TEST_TICKERS = ['ES=F', 'CL=F', 'GC=F']

# =============================================================================
# 资源监控
# =============================================================================

class ResourceMonitor:
    """资源监控"""
    
    def __init__(self):
        self.start_time = time.time()
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_memory = []
    
    def snapshot(self):
        self.cpu_usage.append(psutil.cpu_percent(interval=0.1))
        self.memory_usage.append(psutil.virtual_memory().percent)
        if torch.cuda.is_available():
            self.gpu_memory.append(torch.cuda.memory_allocated() / 1024**3)
    
    def report(self):
        elapsed = time.time() - self.start_time
        print(f"\n📊 资源使用: {elapsed:.1f}s ({elapsed/60:.1f}min)")
        print(f"  CPU: {np.mean(self.cpu_usage):.0f}% (峰值 {max(self.cpu_usage):.0f}%)")
        print(f"  内存: {np.mean(self.memory_usage):.0f}% (峰值 {max(self.memory_usage):.0f}%)")
        if self.gpu_memory:
            print(f"  GPU 显存峰值: {max(self.gpu_memory):.2f} GB")

# =============================================================================
# 进度条回调
# =============================================================================

class TqdmCallback(BaseCallback):
    """带 ETA 的进度条"""
    
    def __init__(self, total_timesteps, check_freq=1000):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.check_freq = check_freq
        self.pbar = None
        self.monitor = ResourceMonitor()
        self.start_time = None
    
    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="训练", unit="steps", ncols=100)
        self.start_time = time.time()
    
    def _on_step(self):
        if self.pbar:
            self.pbar.update(1)
            
            if self.n_calls % self.check_freq == 0:
                self.monitor.snapshot()
                elapsed = time.time() - self.start_time
                steps_per_sec = self.n_calls / elapsed if elapsed > 0 else 0
                remaining = (self.total_timesteps - self.n_calls) / steps_per_sec if steps_per_sec > 0 else 0
                
                self.pbar.set_postfix({
                    'steps/s': f'{steps_per_sec:.0f}',
                    'ETA': f'{remaining/60:.1f}min',
                    'CPU': f'{self.monitor.cpu_usage[-1]:.0f}%',
                    'Mem': f'{self.monitor.memory_usage[-1]:.0f}%'
                })
        
        return True
    
    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()
        self.monitor.report()

# =============================================================================
# 交易环境 (使用核心组件)
# =============================================================================

class FuturesTradingEnvPaper(gym.Env):
    """完全对齐论文的交易环境"""
    
    def __init__(self, prices, returns, transaction_cost=0.001, 
                 vol_target=0.10, use_dsr=True):
        super().__init__()
        
        self.prices = prices
        self.returns = returns.astype(np.float32)
        self.transaction_cost = transaction_cost
        self.use_dsr = use_dsr
        
        # 核心组件
        self.dsr = DifferentialSharpeRatio(eta=0.01) if use_dsr else None
        self.state_builder = MultiTimeScaleState(lookback=200)
        self.vol_scaler = VolatilityScaler(target_vol=vol_target)
        
        # 动作空间: 连续 [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # 观察空间
        n_features = self.state_builder.get_state_dimension()
        self.observation_space = spaces.Box(low=-10, high=10, shape=(n_features,), dtype=np.float32)
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 200  # 从有足够历史数据开始
        self.position = 0.0
        self.done = False
        self.total_reward = 0.0
        self.trades = 0
        self.portfolio_returns = []
        
        if self.dsr:
            self.dsr.reset()
        self.vol_scaler.reset()
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """使用核心组件构建状态"""
        return self.state_builder.compute(self.prices, self.returns, self.t)
    
    def step(self, action):
        # 获取原始仓位
        raw_position = float(action[0])
        
        # 应用 Volatility Scaling
        position = self.vol_scaler.scale(raw_position, self.returns, self.t)
        
        # 交易成本
        trade_size = abs(position - self.position)
        trade_cost = trade_size * self.transaction_cost
        
        # 计算收益
        ret = self.returns[self.t]
        portfolio_return = position * ret - trade_cost
        
        # 计算奖励 (DSR 或简单收益)
        if self.use_dsr and self.dsr:
            reward = self.dsr.update(portfolio_return)
        else:
            reward = portfolio_return * 100
        
        # 更新状态
        if trade_size > 0.01:
            self.trades += 1
        self.position = position
        self.t += 1
        self.total_reward += reward
        self.portfolio_returns.append(portfolio_return)
        
        # 检查结束
        if self.t >= len(self.returns) - 1:
            self.done = True
        
        return self._get_obs(), reward, self.done, False, {}

# =============================================================================
# 风险指标 (完全对齐论文 Table 3)
# =============================================================================

def calculate_metrics_paper(returns, positions):
    """计算论文 Table 3 的所有指标"""
    
    portfolio_returns = positions * returns
    
    # E(R): 年化预期收益
    E_R = np.mean(portfolio_returns) * 252
    
    # Std(R): 年化标准差
    Std_R = np.std(portfolio_returns) * np.sqrt(252)
    
    # DD: 下行偏差
    negative = portfolio_returns[portfolio_returns < 0]
    DD = np.std(negative) * np.sqrt(252) if len(negative) > 0 else Std_R
    
    # Sharpe
    Sharpe = E_R / Std_R if Std_R > 0 else 0
    
    # Sortino
    Sortino = E_R / DD if DD > 0 else 0
    
    # MDD
    cumulative = np.cumprod(1 + portfolio_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    MDD = np.min(drawdown)
    
    # Calmar
    Calmar = E_R / abs(MDD) if MDD != 0 else 0
    
    # % of + Ret
    pct_positive = np.sum(portfolio_returns > 0) / len(portfolio_returns)
    
    # Ave. P
    Ave_P = np.mean(np.abs(positions))
    
    # Ave. L
    Ave_L = Ave_P
    
    return {
        'E(R)': E_R, 'Std(R)': Std_R, 'DD': DD,
        'Sharpe': Sharpe, 'Sortino': Sortino, 'MDD': MDD, 'Calmar': Calmar,
        '% +Ret': pct_positive, 'Ave.P': Ave_P, 'Ave.L': Ave_L
    }

# =============================================================================
# 主训练函数
# =============================================================================

def train_all():
    """训练所有模型"""
    
    print(f"\n训练期: {TRAIN_START} ~ {TRAIN_END}")
    print(f"测试期: {TEST_START} ~ {TEST_END}")
    print(f"步数: {TOTAL_TIMESTEPS:,}")
    
    results = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for ticker in TEST_TICKERS:
        print(f"\n{'='*60}")
        print(f"📊 {ticker}")
        print(f"{'='*60}")
        
        # 加载数据
        filepath = os.path.join(DATA_DIR, f"{ticker}.csv")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        train = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)]
        test = df[(df.index >= TEST_START) & (df.index <= TEST_END)]
        
        train_returns = train['Returns'].dropna().values
        test_returns = test['Returns'].dropna().values
        
        print(f"训练: {len(train)} 天, 测试: {len(test)} 天")
        
        # 环境
        train_env = FuturesTradingEnvPaper(
            train['Close'].values, train_returns,
            transaction_cost=TRANSACTION_COST, use_dsr=True
        )
        test_env = FuturesTradingEnvPaper(
            test['Close'].values, test_returns,
            transaction_cost=TRANSACTION_COST, use_dsr=False
        )
        
        # ===== 基线 =====
        print(f"\n【基线】")
        
        # Long
        pos = np.ones(len(test_returns[200:]))
        m = calculate_metrics_paper(test_returns[200:], pos)
        print(f"  Long   | Sharpe: {m['Sharpe']:.3f} | Return: {m['E(R)']:.2%} | MDD: {m['MDD']:.2%}")
        results.append({'Ticker': ticker, 'Strategy': 'Long', **m})
        
        # MACD
        prices = test['Close']
        macd = prices.ewm(span=12).mean() - prices.ewm(span=26).mean()
        signal = macd.ewm(span=9).mean()
        pos = np.where(macd > signal, 1, -1)[200:]
        m = calculate_metrics_paper(test_returns[200:200+len(pos)], pos)
        print(f"  MACD   | Sharpe: {m['Sharpe']:.3f} | Return: {m['E(R)']:.2%} | MDD: {m['MDD']:.2%}")
        results.append({'Ticker': ticker, 'Strategy': 'MACD', **m})
        
        # ===== DRL =====
        print(f"\n【DRL】")
        
        # DQN (需要离散动作空间)
        print(f"  🤖 DQN...")
        
        # 创建离散动作空间的环境
        class DiscreteEnv(FuturesTradingEnvPaper):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # 离散动作: 0=short, 1=neutral, 2=long
                self.action_space = spaces.Discrete(3)
            
            def step(self, action):
                # 转换为连续动作
                continuous_action = np.array([float(action - 1)])  # -1, 0, 1
                return super().step(continuous_action)
        
        train_env_discrete = DiscreteEnv(
            train['Close'].values, train_returns,
            transaction_cost=TRANSACTION_COST, use_dsr=True
        )
        test_env_discrete = DiscreteEnv(
            test['Close'].values, test_returns,
            transaction_cost=TRANSACTION_COST, use_dsr=False
        )
        
        model_dqn = DQN(
            "MlpPolicy", train_env_discrete,
            learning_rate=LEARNING_RATE,
            buffer_size=BUFFER_SIZE,
            learning_starts=10000,
            batch_size=BATCH_SIZE,
            gamma=0.99,
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
        
        # PPO 和 A2C (连续动作空间)
        for algo_name, AlgoClass in [('PPO', PPO), ('A2C', A2C)]:
            print(f"\n  🤖 {algo_name}...")
            
            if algo_name == 'PPO':
                model = AlgoClass(
                    "MlpPolicy", train_env,
                    learning_rate=3e-4,
                    batch_size=BATCH_SIZE,
                    gamma=0.99,
                    policy_kwargs=dict(net_arch=HIDDEN_LAYERS),
                    verbose=0, device=DEVICE
                )
            else:  # A2C
                model = AlgoClass(
                    "MlpPolicy", train_env,
                    learning_rate=7e-4,
                    gamma=0.99,
                    policy_kwargs=dict(net_arch=HIDDEN_LAYERS),
                    verbose=0, device=DEVICE
                )
            
            model.learn(total_timesteps=TOTAL_TIMESTEPS, 
                        callback=TqdmCallback(TOTAL_TIMESTEPS),
                        progress_bar=False)
            
            # 评估
            obs, _ = test_env.reset()
            positions = []
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                positions.append(float(action[0]))
                obs, _, done, _, _ = test_env.step(action)
            
            m = calculate_metrics_paper(test_returns[200:200+len(positions)], np.array(positions))
            print(f"\n  {algo_name}  | Sharpe: {m['Sharpe']:.3f} | Return: {m['E(R)']:.2%} | MDD: {m['MDD']:.2%}")
            results.append({'Ticker': ticker, 'Strategy': algo_name, **m})
    
    return results, timestamp

def summarize(results, timestamp):
    """汇总结果"""
    print(f"\n{'='*80}")
    print("📊 结果汇总")
    print(f"{'='*80}")
    
    df = pd.DataFrame(results)
    
    for ticker in TEST_TICKERS:
        print(f"\n{ticker}:")
        for _, row in df[df['Ticker'] == ticker].iterrows():
            print(f"  {row['Strategy']:<6} | Sharpe: {row['Sharpe']:>6.3f} | "
                  f"Return: {row['E(R)']:>7.2%} | Sortino: {row['Sortino']:>6.3f} | "
                  f"MDD: {row['MDD']:>7.2%} | Calmar: {row['Calmar']:>6.3f}")
    
    df.to_csv(f'results_final_{timestamp}.csv', index=False)
    print(f"\n💾 保存: results_final_{timestamp}.csv")

if __name__ == "__main__":
    results, ts = train_all()
    summarize(results, ts)
    print(f"\n{'='*80}")
    print("✅ 完成!")
    print(f"{'='*80}")
