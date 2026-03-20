#!/usr/bin/env python3
"""
Deep Reinforcement Learning for Trading
实现论文中的三个 DRL 算法: DQN, PPO, A2C

Based on: "Deep Reinforcement Learning for Trading" (Zhang, Zohren, Roberts, 2019)
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces

# DRL 算法
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# =============================================================================
# 配置
# =============================================================================

DATA_DIR = 'data/futures_processed'
TRAIN_START = '2011-01-01'
TRAIN_END = '2017-06-30'
TEST_START = '2017-07-01'
TEST_END = '2019-12-31'

# 论文参数
TRANSACTION_COST = 0.001  # 10 bps
LOOKBACK = 50  # 状态历史长度

# 训练参数
TOTAL_TIMESTEPS = 10000  # 快速测试
EVAL_FREQ = 5000
N_EVAL_EPISODES = 10

print("=" * 80)
print("🤖 Deep Reinforcement Learning for Trading")
print("=" * 80)
print(f"数据目录: {DATA_DIR}")
print(f"训练期间: {TRAIN_START} ~ {TRAIN_END}")
print(f"测试期间: {TEST_START} ~ {TEST_END}")
print(f"训练步数: {TOTAL_TIMESTEPS}")
print("=" * 80)

# =============================================================================
# 自定义交易环境
# =============================================================================

class FuturesTradingEnv(gym.Env):
    """
    期货交易环境
    State: 过去 LOOKBACK 天的收益率
    Action: 连续 [-1, 1] 或离散 [0, 1, 2] (short, neutral, long)
    Reward: 组合收益 - 交易成本
    """
    
    def __init__(self, returns, lookback=50, transaction_cost=0.001, continuous_action=False):
        super().__init__()
        
        self.returns = returns
        self.lookback = lookback
        self.transaction_cost = transaction_cost
        self.continuous_action = continuous_action
        
        # 定义空间
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(lookback,), dtype=np.float32
        )
        
        if continuous_action:
            # 连续动作空间 (for PPO/A2C)
            self.action_space = spaces.Box(
                low=-1, high=1, shape=(1,), dtype=np.float32
            )
        else:
            # 离散动作空间 (for DQN)
            self.action_space = spaces.Discrete(3)  # 0=short, 1=neutral, 2=long
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.lookback
        self.position = 0.0
        self.done = False
        self.total_reward = 0.0
        
        return self._get_state(), {}
    
    def _get_state(self):
        """获取状态: 归一化的历史收益率"""
        state = self.returns[self.t - self.lookback:self.t]
        # 归一化
        mean = np.mean(state)
        std = np.std(state) + 1e-8
        state = (state - mean) / std
        return state.astype(np.float32)
    
    def step(self, action):
        """执行动作"""
        # 转换动作到持仓
        if self.continuous_action:
            new_position = float(action[0])  # [-1, 1]
        else:
            new_position = float(action - 1)  # [0,1,2] -> [-1,0,1]
        
        # 计算交易成本
        trade_cost = abs(new_position - self.position) * self.transaction_cost
        
        # 计算收益
        ret = self.returns[self.t]
        portfolio_return = new_position * ret - trade_cost
        
        # 奖励 = 收益率 (论文使用 Differential Sharpe Ratio，这里简化)
        reward = portfolio_return * 100  # 放大以便训练
        
        # 更新状态
        self.position = new_position
        self.t += 1
        self.total_reward += reward
        
        # 检查是否结束
        if self.t >= len(self.returns) - 1:
            self.done = True
        
        return self._get_state(), reward, self.done, False, {}

# =============================================================================
# 加载数据
# =============================================================================

def load_data(ticker):
    """加载单个合约的数据"""
    filepath = os.path.join(DATA_DIR, f"{ticker}.csv")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    train_mask = (df.index >= TRAIN_START) & (df.index < TRAIN_END)
    test_mask = (df.index >= TEST_START) & (df.index <= TEST_END)
    
    train_returns = df.loc[train_mask, 'Returns'].dropna().values
    test_returns = df.loc[test_mask, 'Returns'].dropna().values
    
    return train_returns, test_returns

# =============================================================================
# 训练和评估
# =============================================================================

def train_and_evaluate(ticker, train_returns, test_returns, algorithm='DQN', verbose=0):
    """训练并评估 DRL 模型"""
    
    print(f"\n{'='*60}")
    print(f"📊 {ticker} - {algorithm}")
    print(f"{'='*60}")
    
    # 创建环境
    if algorithm == 'DQN':
        train_env = DummyVecEnv([lambda: FuturesTradingEnv(train_returns, LOOKBACK, TRANSACTION_COST, continuous_action=False)])
        test_env = DummyVecEnv([lambda: FuturesTradingEnv(test_returns, LOOKBACK, TRANSACTION_COST, continuous_action=False)])
    else:  # PPO, A2C
        train_env = DummyVecEnv([lambda: FuturesTradingEnv(train_returns, LOOKBACK, TRANSACTION_COST, continuous_action=True)])
        test_env = DummyVecEnv([lambda: FuturesTradingEnv(test_returns, LOOKBACK, TRANSACTION_COST, continuous_action=True)])
    
    # 选择算法
    if algorithm == 'DQN':
        model = DQN(
            'MlpPolicy', train_env,
            learning_rate=1e-3,
            buffer_size=10000,
            learning_starts=1000,
            batch_size=64,
            tau=0.01,
            gamma=0.99,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            verbose=verbose
        )
    elif algorithm == 'PPO':
        model = PPO(
            'MlpPolicy', train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=verbose
        )
    elif algorithm == 'A2C':
        model = A2C(
            'MlpPolicy', train_env,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            verbose=verbose
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # 训练
    print(f"  训练中... ({TOTAL_TIMESTEPS} timesteps)")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    
    # 测试
    print(f"  测试中...")
    obs = test_env.reset()
    done = False
    total_reward = 0
    positions = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        total_reward += reward
        if algorithm == 'DQN':
            positions.append(action[0] - 1)
        else:
            positions.append(action[0][0])
    
    # 计算指标
    positions = np.array(positions)
    portfolio_returns = positions * test_returns[LOOKBACK:LOOKBACK+len(positions)]
    
    # 年化收益
    annual_return = np.mean(portfolio_returns) * 252
    
    # 年化波动
    annual_std = np.std(portfolio_returns) * np.sqrt(252)
    
    # Sharpe
    sharpe = annual_return / annual_std if annual_std > 0 else 0
    
    # 最大回撤
    cumulative = np.cumprod(1 + portfolio_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = np.min(drawdown)
    
    print(f"  ✅ 完成!")
    print(f"     Sharpe: {sharpe:.3f}")
    print(f"     年化收益: {annual_return:.2%}")
    print(f"     最大回撤: {max_dd:.2%}")
    
    return {
        'ticker': ticker,
        'algorithm': algorithm,
        'sharpe': sharpe,
        'annual_return': annual_return,
        'max_dd': max_dd,
        'total_reward': total_reward
    }

# =============================================================================
# 主程序
# =============================================================================

def main():
    """运行所有测试"""
    
    # 选择测试合约
    test_tickers = ['ES=F', 'CL=F', 'GC=F']
    algorithms = ['DQN', 'PPO', 'A2C']
    
    print(f"\n测试合约: {test_tickers}")
    print(f"算法: {algorithms}")
    
    all_results = []
    
    for ticker in test_tickers:
        print(f"\n{'='*80}")
        print(f"📊 {ticker}")
        print(f"{'='*80}")
        
        try:
            train_returns, test_returns = load_data(ticker)
            print(f"  训练集: {len(train_returns)} 天")
            print(f"  测试集: {len(test_returns)} 天")
            
            for algo in algorithms:
                result = train_and_evaluate(ticker, train_returns, test_returns, algo)
                all_results.append(result)
                
        except Exception as e:
            print(f"  ❌ 错误: {str(e)[:50]}")
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("📊 结果汇总")
    print("=" * 80)
    
    df_results = pd.DataFrame(all_results)
    
    # 按合约分组显示
    for ticker in test_tickers:
        print(f"\n{ticker}:")
        ticker_results = df_results[df_results['ticker'] == ticker]
        for _, row in ticker_results.iterrows():
            print(f"  {row['algorithm']:<6} | Sharpe: {row['sharpe']:>6.3f} | Return: {row['annual_return']:>7.2%} | MDD: {row['max_dd']:>7.2%}")
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'drl_results_{timestamp}.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\n💾 结果已保存到: {output_file}")
    
    print("\n" + "=" * 80)
    print("✅ DRL 训练完成!")
    print("=" * 80)
    
    return df_results

if __name__ == "__main__":
    results = main()
