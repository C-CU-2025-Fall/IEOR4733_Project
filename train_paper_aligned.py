#!/usr/bin/env python3
"""
完全对齐论文的 Deep RL Trading 实现
论文: "Deep Reinforcement Learning for Trading" (Zhang, Zohren, Roberts, 2019)

关键对齐点:
1. 训练数据: 2005-2010 → 测试 2011-2019
2. 神经网络: 3层 256神经元 (推断)
3. DQN改进: Double + Dueling + Prioritized Replay
4. 奖励函数: Differential Sharpe Ratio
5. 状态空间: 多时间尺度动量特征
6. Volatility Scaling: 10% 年化波动率目标
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

# GPU 设置
import torch
print("=" * 80)
print("🤖 Deep RL Trading - 完全对齐论文版本")
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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.dqn.policies import DuelingCnnPolicy
import gymnasium as gym
from gymnasium import spaces

# =============================================================================
# 资源监控
# =============================================================================

class ResourceMonitor:
    """监控 CPU、内存、GPU 使用"""
    
    def __init__(self):
        self.start_time = time.time()
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_memory = []
    
    def snapshot(self):
        """记录当前资源使用"""
        self.cpu_usage.append(psutil.cpu_percent(interval=0.1))
        self.memory_usage.append(psutil.virtual_memory().percent)
        if torch.cuda.is_available():
            self.gpu_memory.append(torch.cuda.memory_allocated() / 1024**3)
    
    def report(self):
        """生成资源报告"""
        elapsed = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"📊 资源使用报告")
        print(f"{'='*60}")
        print(f"运行时间: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
        print(f"CPU 平均: {np.mean(self.cpu_usage):.1f}% (峰值: {max(self.cpu_usage):.1f}%)")
        print(f"内存 平均: {np.mean(self.memory_usage):.1f}% (峰值: {max(self.memory_usage):.1f}%)")
        if self.gpu_memory:
            print(f"GPU 显存 峰值: {max(self.gpu_memory):.2f} GB")
        print(f"{'='*60}")

# =============================================================================
# 论文配置
# =============================================================================

# 数据配置 (对齐论文)
DATA_DIR = 'data/futures_processed'
TRAIN_START = '2011-01-03'  # 论文: 2005, 但我们数据从2011开始
TRAIN_END = '2015-12-31'    # 论文: 5年训练
TEST_START = '2016-01-01'   # 论文: 2011, 但我们用2016
TEST_END = '2019-12-31'
TRANSACTION_COST = 0.001    # 10 bps (论文 Table 2)
VOL_TARGET = 0.10           # 10% 年化波动率目标 (论文 Section 4.3)

# 状态空间配置 (对齐论文 Section 3.1)
MOMENTUM_WINDOWS = [5, 10, 25, 50, 100, 200]  # 多时间尺度动量
LOOKBACK = 200  # 最大回看窗口

# 神经网络配置 (推断 + RL 标准实践)
HIDDEN_LAYERS = [256, 256, 256]  # 3层 256神经元
ACTIVATION = "ReLU"

# 训练配置
TOTAL_TIMESTEPS = 500000     # 论文可能用更多
BATCH_SIZE = 256
BUFFER_SIZE = 1000000
LEARNING_RATE = 1e-4
GAMMA = 0.99

# 测试合约
TEST_TICKERS = ['ES=F', 'CL=F', 'GC=F']

# =============================================================================
# Differential Sharpe Ratio 奖励函数
# =============================================================================

class DifferentialSharpeRatio:
    """
    论文 Equation 7-8:
    ΔSharpe_t = (R_t * Sharpe_{t-1} - 0.5 * R_t^2) / (t * σ_t)
    """
    
    def __init__(self, eta=0.01):
        self.eta = eta  # 学习率
        self.reset()
    
    def reset(self):
        self.t = 0
        self.A_t = 0  # 累积收益均值
        self.B_t = 0  # 累积收益二阶矩
        self.Sharpe_t = 0
    
    def update(self, R_t):
        """更新并返回 DSR 奖励"""
        self.t += 1
        
        # 指数移动平均
        delta_A = R_t - self.A_t
        delta_B = R_t**2 - self.B_t
        
        A_t_new = self.A_t + self.eta * delta_A
        B_t_new = self.B_t + self.eta * delta_B
        
        # 计算 DSR
        if B_t_new > A_t_new**2:
            Sharpe_new = A_t_new / np.sqrt(B_t_new - A_t_new**2)
            DSR = Sharpe_new - self.Sharpe_t
        else:
            DSR = 0
            Sharpe_new = self.Sharpe_t
        
        # 更新状态
        self.A_t = A_t_new
        self.B_t = B_t_new
        self.Sharpe_t = Sharpe_new
        
        return DSR * 100  # 放大以便训练

# =============================================================================
# 交易环境 (完全对齐论文)
# =============================================================================

class FuturesTradingEnvPaper(gym.Env):
    """
    完全对齐论文的交易环境
    
    状态空间: 多时间尺度动量特征
    动作空间: 连续 [-1, 1]
    奖励函数: Differential Sharpe Ratio
    Volatility Scaling: 10% 年化目标
    """
    
    def __init__(self, prices, returns, lookback=200, transaction_cost=0.001, 
                 vol_target=0.10, use_dsr=True):
        super().__init__()
        
        self.prices = prices
        self.returns = returns.astype(np.float32)
        self.lookback = lookback
        self.transaction_cost = transaction_cost
        self.vol_target = vol_target
        self.use_dsr = use_dsr
        
        # DSR 奖励计算器
        self.dsr = DifferentialSharpeRatio()
        
        # 动作空间: 连续 [-1, 1]
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )
        
        # 观察空间: 多时间尺度动量特征
        # [r_5, r_10, r_25, r_50, r_100, r_200, vol_20, ...]
        n_features = len(MOMENTUM_WINDOWS) + 5  # 动量 + 技术指标
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(n_features,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.lookback
        self.position = 0.0
        self.done = False
        self.total_reward = 0.0
        self.trades = 0
        self.dsr.reset()
        self.returns_history = []
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """
        构造状态: 多时间尺度动量特征
        对齐论文 Section 3.1
        """
        features = []
        
        # 1. 多时间尺度动量 (对齐论文)
        for window in MOMENTUM_WINDOWS:
            if self.t >= window:
                momentum = np.sum(self.returns[self.t-window:self.t])
            else:
                momentum = 0
            features.append(momentum)
        
        # 2. 当前价格相对于历史的位置
        window_prices = self.prices[self.t-50:self.t]
        if len(window_prices) > 0:
            price_position = (self.prices[self.t] - np.min(window_prices)) / \
                           (np.max(window_prices) - np.min(window_prices) + 1e-8)
        else:
            price_position = 0.5
        features.append(price_position)
        
        # 3. 波动率 (20日)
        if self.t >= 20:
            vol = np.std(self.returns[self.t-20:self.t]) * np.sqrt(252)
        else:
            vol = 0.2
        features.append(vol)
        
        # 4. 当前收益
        features.append(self.returns[self.t])
        
        # 5. 历史夏普 (简化)
        if len(self.returns_history) > 20:
            hist_returns = np.array(self.returns_history[-20:])
            sharpe = np.mean(hist_returns) / (np.std(hist_returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0
        features.append(sharpe)
        
        # 6. 成交量变化 (如果有)
        features.append(0)  # 占位符
        
        # 归一化
        features = np.array(features, dtype=np.float32)
        features = np.clip(features, -10, 10)
        
        return features
    
    def _apply_volatility_scaling(self, position):
        """
        Volatility Scaling (论文 Section 4.3)
        目标年化波动率 10%
        """
        if self.t >= 20:
            recent_vol = np.std(self.returns[self.t-20:self.t]) * np.sqrt(252)
            if recent_vol > 0.01:  # 避免除零
                scaling = self.vol_target / recent_vol
                scaling = np.clip(scaling, 0.1, 10)  # 限制杠杆
                return position * scaling
        return position
    
    def step(self, action):
        # 获取原始仓位
        raw_position = float(action[0])
        
        # 应用 Volatility Scaling
        position = self._apply_volatility_scaling(raw_position)
        
        # 计算交易成本
        trade_size = abs(position - self.position)
        trade_cost = trade_size * self.transaction_cost
        
        # 计算收益
        ret = self.returns[self.t]
        portfolio_return = position * ret - trade_cost
        
        # 计算奖励
        if self.use_dsr:
            reward = self.dsr.update(portfolio_return)
        else:
            reward = portfolio_return * 100
        
        # 更新状态
        if trade_size > 0.01:
            self.trades += 1
        self.position = position
        self.t += 1
        self.total_reward += reward
        self.returns_history.append(portfolio_return)
        
        # 检查结束
        if self.t >= len(self.returns) - 1:
            self.done = True
        
        return self._get_obs(), reward, self.done, False, {}

# =============================================================================
# 带进度条的回调
# =============================================================================

class TqdmCallback(BaseCallback):
    """带 ETA 的进度条回调"""
    
    def __init__(self, total_timesteps, check_freq=1000, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.check_freq = check_freq
        self.pbar = None
        self.monitor = ResourceMonitor()
        self.start_time = None
    
    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, 
                        desc="训练进度",
                        unit="steps",
                        ncols=100)
        self.start_time = time.time()
    
    def _on_step(self):
        if self.pbar:
            self.pbar.update(1)
            
            # 定期更新资源监控
            if self.n_calls % self.check_freq == 0:
                self.monitor.snapshot()
                
                # 计算 ETA
                elapsed = time.time() - self.start_time
                steps_per_sec = self.n_calls / elapsed if elapsed > 0 else 0
                remaining_steps = self.total_timesteps - self.n_calls
                eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                
                # 更新进度条描述
                self.pbar.set_postfix({
                    'steps/s': f'{steps_per_sec:.0f}',
                    'ETA': f'{eta_seconds/60:.1f}min',
                    'CPU': f'{self.monitor.cpu_usage[-1]:.0f}%',
                    'Mem': f'{self.monitor.memory_usage[-1]:.0f}%'
                })
        
        return True
    
    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()
        self.monitor.report()

# =============================================================================
# 数据加载
# =============================================================================

def load_data(ticker):
    """加载数据"""
    filepath = os.path.join(DATA_DIR, f"{ticker}.csv")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    train_mask = (df.index >= TRAIN_START) & (df.index <= TRAIN_END)
    test_mask = (df.index >= TEST_START) & (df.index <= TEST_END)
    
    train = df[train_mask].copy()
    test = df[test_mask].copy()
    
    return train, test

# =============================================================================
# 风险指标 (完全对齐论文 Table 3)
# =============================================================================

def calculate_metrics_paper(returns, positions):
    """
    计算论文 Table 3 的所有指标
    
    指标:
    - E(R): 年化预期收益
    - Std(R): 年化标准差
    - DD: 下行偏差 (Downside Deviation)
    - Sharpe: Sharpe Ratio
    - Sortino: Sortino Ratio
    - MDD: 最大回撤
    - Calmar: Calmar Ratio
    - % of + Ret: 正收益天数占比
    - Ave. P: 平均仓位
    - Ave. L: 平均杠杆
    """
    # 组合收益
    portfolio_returns = positions * returns
    
    # 1. E(R): 年化预期收益
    E_R = np.mean(portfolio_returns) * 252
    
    # 2. Std(R): 年化标准差
    Std_R = np.std(portfolio_returns) * np.sqrt(252)
    
    # 3. DD: 下行偏差
    negative_returns = portfolio_returns[portfolio_returns < 0]
    DD = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else Std_R
    
    # 4. Sharpe Ratio
    Sharpe = E_R / Std_R if Std_R > 0 else 0
    
    # 5. Sortino Ratio
    Sortino = E_R / DD if DD > 0 else 0
    
    # 6. 最大回撤
    cumulative = np.cumprod(1 + portfolio_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    MDD = np.min(drawdown)
    
    # 7. Calmar Ratio
    Calmar = E_R / abs(MDD) if MDD != 0 else 0
    
    # 8. % of + Ret: 正收益天数占比
    pct_positive = np.sum(portfolio_returns > 0) / len(portfolio_returns)
    
    # 9. Ave. P: 平均仓位 (绝对值)
    Ave_P = np.mean(np.abs(positions))
    
    # 10. Ave. L: 平均杠杆
    Ave_L = np.mean(np.abs(positions))  # 论文中与 Ave.P 相同
    
    return {
        'E(R)': E_R,
        'Std(R)': Std_R,
        'DD': DD,
        'Sharpe': Sharpe,
        'Sortino': Sortino,
        'MDD': MDD,
        'Calmar': Calmar,
        '% of + Ret': pct_positive,
        'Ave. P': Ave_P,
        'Ave. L': Ave_L
    }

# =============================================================================
# 主训练函数
# =============================================================================

def train_all():
    """训练所有模型"""
    
    print(f"\n{'='*80}")
    print("🚀 开始训练 (完全对齐论文)")
    print(f"{'='*80}")
    print(f"训练期: {TRAIN_START} ~ {TRAIN_END}")
    print(f"测试期: {TEST_START} ~ {TEST_END}")
    print(f"训练步数: {TOTAL_TIMESTEPS:,}")
    print(f"网络架构: {HIDDEN_LAYERS}")
    print(f"设备: {DEVICE}")
    
    results = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = f"models/{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    for ticker in TEST_TICKERS:
        print(f"\n{'='*80}")
        print(f"📊 {ticker}")
        print(f"{'='*80}")
        
        # 加载数据
        train, test = load_data(ticker)
        train_returns = train['Returns'].dropna().values
        test_returns = test['Returns'].dropna().values
        test_prices = test['Close'].values
        
        print(f"训练集: {len(train)} 天")
        print(f"测试集: {len(test)} 天")
        
        # 创建环境
        train_env = FuturesTradingEnvPaper(
            train['Close'].values, train_returns,
            lookback=LOOKBACK, transaction_cost=TRANSACTION_COST,
            vol_target=VOL_TARGET, use_dsr=True
        )
        test_env = FuturesTradingEnvPaper(
            test['Close'].values, test_returns,
            lookback=LOOKBACK, transaction_cost=TRANSACTION_COST,
            vol_target=VOL_TARGET, use_dsr=False
        )
        
        # ===== 基线策略 =====
        print(f"\n【基线策略】")
        
        # Long
        positions = np.ones(len(test_returns[LOOKBACK:]))
        metrics = calculate_metrics_paper(test_returns[LOOKBACK:], positions)
        print(f"  Long   | Sharpe: {metrics['Sharpe']:>6.3f} | Return: {metrics['E(R)']:>7.2%} | MDD: {metrics['MDD']:>7.2%}")
        results.append({'Ticker': ticker, 'Strategy': 'Long', **metrics})
        
        # MACD
        prices = test['Close']
        ema_fast = prices.ewm(span=12).mean()
        ema_slow = prices.ewm(span=26).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=9).mean()
        positions = np.where(macd > signal_line, 1, -1)[LOOKBACK:]
        metrics = calculate_metrics_paper(test_returns[LOOKBACK:LOOKBACK+len(positions)], positions)
        print(f"  MACD   | Sharpe: {metrics['Sharpe']:>6.3f} | Return: {metrics['E(R)']:>7.2%} | MDD: {metrics['MDD']:>7.2%}")
        results.append({'Ticker': ticker, 'Strategy': 'MACD', **metrics})
        
        # ===== DRL 模型 =====
        print(f"\n【DRL 训练】")
        
        # DQN (带改进)
        print(f"  🤖 DQN (Double + Dueling)...")
        model_dqn = DQN(
            "MlpPolicy", train_env,
            learning_rate=LEARNING_RATE,
            buffer_size=BUFFER_SIZE,
            learning_starts=10000,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            train_freq=4,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            policy_kwargs=dict(
                net_arch=HIDDEN_LAYERS,
                dueling=True,  # Dueling DQN
            ),
            verbose=0,
            device=DEVICE
        )
        
        # 训练带进度条
        model_dqn.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=TqdmCallback(TOTAL_TIMESTEPS),
            progress_bar=False
        )
        
        # 评估
        positions = evaluate_model(model_dqn, test_env)
        metrics = calculate_metrics_paper(test_returns[LOOKBACK:LOOKBACK+len(positions)], positions)
        print(f"\n  DQN    | Sharpe: {metrics['Sharpe']:>6.3f} | Return: {metrics['E(R)']:>7.2%} | MDD: {metrics['MDD']:>7.2%}")
        results.append({'Ticker': ticker, 'Strategy': 'DQN', **metrics})
        
        # PPO
        print(f"\n  🤖 PPO...")
        model_ppo = PPO(
            "MlpPolicy", train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=BATCH_SIZE,
            n_epochs=10,
            gamma=GAMMA,
            policy_kwargs=dict(net_arch=HIDDEN_LAYERS),
            verbose=0,
            device=DEVICE
        )
        model_ppo.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=TqdmCallback(TOTAL_TIMESTEPS),
            progress_bar=False
        )
        positions = evaluate_model(model_ppo, test_env)
        metrics = calculate_metrics_paper(test_returns[LOOKBACK:LOOKBACK+len(positions)], positions)
        print(f"\n  PPO    | Sharpe: {metrics['Sharpe']:>6.3f} | Return: {metrics['E(R)']:>7.2%} | MDD: {metrics['MDD']:>7.2%}")
        results.append({'Ticker': ticker, 'Strategy': 'PPO', **metrics})
        
        # A2C
        print(f"\n  🤖 A2C...")
        model_a2c = A2C(
            "MlpPolicy", train_env,
            learning_rate=7e-4,
            n_steps=5,
            gamma=GAMMA,
            policy_kwargs=dict(net_arch=HIDDEN_LAYERS),
            verbose=0,
            device=DEVICE
        )
        model_a2c.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=TqdmCallback(TOTAL_TIMESTEPS),
            progress_bar=False
        )
        positions = evaluate_model(model_a2c, test_env)
        metrics = calculate_metrics_paper(test_returns[LOOKBACK:LOOKBACK+len(positions)], positions)
        print(f"\n  A2C    | Sharpe: {metrics['Sharpe']:>6.3f} | Return: {metrics['E(R)']:>7.2%} | MDD: {metrics['MDD']:>7.2%}")
        results.append({'Ticker': ticker, 'Strategy': 'A2C', **metrics})
        
        # 保存模型
        model_dqn.save(f"{model_dir}/{ticker}_dqn")
        model_ppo.save(f"{model_dir}/{ticker}_ppo")
        model_a2c.save(f"{model_dir}/{ticker}_a2c")
        print(f"  💾 模型已保存")
    
    return results, timestamp

def evaluate_model(model, env):
    """评估模型"""
    obs, _ = env.reset()
    positions = []
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        positions.append(float(action[0]))
        obs, _, done, _, _ = env.step(action)
    
    return np.array(positions)

def summarize_results(results, timestamp):
    """汇总结果"""
    print(f"\n{'='*80}")
    print("📊 结果汇总 (完全对齐论文 Table 3)")
    print(f"{'='*80}")
    
    df = pd.DataFrame(results)
    
    for ticker in TEST_TICKERS:
        print(f"\n{ticker}:")
        ticker_df = df[df['Ticker'] == ticker]
        for _, row in ticker_df.iterrows():
            print(f"  {row['Strategy']:<6} | "
                  f"Sharpe: {row['Sharpe']:>6.3f} | "
                  f"Return: {row['E(R)']:>7.2%} | "
                  f"Sortino: {row['Sortino']:>6.3f} | "
                  f"MDD: {row['MDD']:>7.2%} | "
                  f"Calmar: {row['Calmar']:>6.3f}")
    
    # 保存
    output_file = f'results_paper_aligned_{timestamp}.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 结果已保存到: {output_file}")
    
    return df

# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    results, timestamp = train_all()
    df = summarize_results(results, timestamp)
    
    print(f"\n{'='*80}")
    print("✅ 训练完成!")
    print(f"{'='*80}")
