#!/usr/bin/env python3
"""
快速 DRL 测试 - 使用现有的 pilot_test.py 中的 SimpleDQN
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# 添加当前目录到路径
sys.path.insert(0, '.')

from pilot_test import SimpleDQN, FuturesEnv, calculate_metrics

# 配置
DATA_DIR = 'data/futures_processed'
TRAIN_START = '2011-01-01'
TRAIN_END = '2017-06-30'
TEST_START = '2017-07-01'
TEST_END = '2019-12-31'
TRANSACTION_COST = 0.001
LOOKBACK = 50

print("=" * 80)
print("🤖 快速 DRL 测试 - SimpleDQN (纯 NumPy)")
print("=" * 80)

# 加载数据
def load_data(ticker):
    filepath = os.path.join(DATA_DIR, f"{ticker}.csv")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    train_mask = (df.index >= TRAIN_START) & (df.index < TRAIN_END)
    test_mask = (df.index >= TEST_START) & (df.index <= TEST_END)
    
    train = df.loc[train_mask].copy()
    test = df.loc[test_mask].copy()
    
    return train, test

# 测试合约
test_tickers = ['ES=F', 'CL=F', 'GC=F']
results = []

for ticker in test_tickers:
    print(f"\n{'='*60}")
    print(f"📊 {ticker}")
    print(f"{'='*60}")
    
    try:
        train, test = load_data(ticker)
        train_returns = train['Returns'].dropna().values
        test_returns = test['Returns'].dropna().values
        test_prices = test['Close']  # 保持为 pandas Series
        
        print(f"  训练集: {len(train_returns)} 天")
        print(f"  测试集: {len(test_returns)} 天")
        
        # 基线策略
        from pilot_test import strategy_long, strategy_sign, strategy_macd
        
        # 1. Long
        positions = strategy_long(test_returns)
        metrics = calculate_metrics(test_returns, positions)
        print(f"\n  Long:  Sharpe={metrics['Sharpe']:.3f}, MDD={metrics['MDD']:.2%}")
        results.append({'ticker': ticker, 'strategy': 'Long', **metrics})
        
        # 2. MACD
        positions = strategy_macd(test_prices)
        metrics = calculate_metrics(test_returns, positions)
        print(f"  MACD:  Sharpe={metrics['Sharpe']:.3f}, MDD={metrics['MDD']:.2%}")
        results.append({'ticker': ticker, 'strategy': 'MACD', **metrics})
        
        # 3. DQN
        print(f"\n  🤖 训练 DQN...")
        
        env = FuturesEnv(train_returns, lookback=LOOKBACK)
        agent = SimpleDQN(state_size=LOOKBACK, n_actions=3, hidden_size=64)
        
        # 训练
        n_episodes = 50  # 增加训练轮数
        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0
            
            while not env.done:
                action = agent.act(state, training=True)
                next_state, reward, done = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.train(batch_size=64)
                state = next_state
                total_reward += reward
            
            if (episode + 1) % 10 == 0:
                print(f"    Episode {episode+1}/{n_episodes}, Reward={total_reward:.2f}, Eps={agent.epsilon:.3f}")
        
        # 测试 DQN
        test_env = FuturesEnv(test_returns, lookback=LOOKBACK)
        state = test_env.reset()
        positions = []
        
        while not test_env.done:
            action = agent.act(state, training=False)
            positions.append(action - 1)
            state, _, done = test_env.step(action)
        
        positions = np.array(positions)
        metrics = calculate_metrics(test_returns[LOOKBACK:LOOKBACK+len(positions)], positions)
        print(f"\n  DQN:   Sharpe={metrics['Sharpe']:.3f}, MDD={metrics['MDD']:.2%}")
        results.append({'ticker': ticker, 'strategy': 'DQN', **metrics})
        
    except Exception as e:
        print(f"  ❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()

# 汇总
print("\n" + "=" * 80)
print("📊 结果汇总")
print("=" * 80)

df_results = pd.DataFrame(results)

for ticker in test_tickers:
    print(f"\n{ticker}:")
    ticker_results = df_results[df_results['ticker'] == ticker]
    for _, row in ticker_results.iterrows():
        print(f"  {row['strategy']:<6} | Sharpe: {row['Sharpe']:>6.3f} | Sortino: {row['Sortino']:>6.3f} | MDD: {row['MDD']:>7.2%}")

# 保存
df_results.to_csv('drl_quick_results.csv', index=False)
print(f"\n💾 结果已保存到: drl_quick_results.csv")

print("\n" + "=" * 80)
print("✅ 快速 DRL 测试完成!")
print("=" * 80)
