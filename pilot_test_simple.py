#!/usr/bin/env python3
"""
简化版 Pilot Test - 快速验证数据和基线策略
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# 使用处理后的数据
DATA_DIR = 'data/futures_processed'
TRAIN_START = '2011-01-01'
TRAIN_END = '2017-06-30'
TEST_START = '2017-07-01'
TEST_END = '2019-12-31'
TRANSACTION_COST = 0.001  # 10 bps per trade

print("=" * 80)
print("📊 3️⃣ Pilot Test - 基线策略验证")
print("=" * 80)

# =============================================================================
# 1. 加载数据
# =============================================================================
print("\n【1. 加载数据】")

futures_data = {}
for filename in os.listdir(DATA_DIR):
    if filename.endswith('.csv'):
        ticker = filename.replace('.csv', '')
        filepath = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        futures_data[ticker] = df

print(f"  加载合约数: {len(futures_data)}")

# 选择几个代表性合约进行测试
test_tickers = ['ES=F', 'CL=F', 'GC=F']  # S&P500, 原油, 黄金
print(f"  测试合约: {test_tickers}")

# =============================================================================
# 2. 定义基线策略
# =============================================================================
print("\n【2. 基线策略】")

def strategy_long(df):
    """买入持有"""
    return np.ones(len(df))

def strategy_sign(df, lookback=252):
    """动量策略 - 过去一年收益的符号"""
    signals = np.zeros(len(df))
    returns = df['Returns'].values
    for i in range(lookback, len(df)):
        past_return = np.sum(returns[i-lookback:i])
        signals[i] = np.sign(past_return)
    return signals

def strategy_macd(df, fast=12, slow=26, signal=9):
    """MACD 策略"""
    ema_fast = df['Close'].ewm(span=fast).mean()
    ema_slow = df['Close'].ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    signals = np.where(macd > signal_line, 1, -1)
    return signals

strategies = {
    'Long': strategy_long,
    'Sign': strategy_sign,
    'MACD': strategy_macd,
}

print(f"  策略数: {len(strategies)}")
for name in strategies:
    print(f"    - {name}")

# =============================================================================
# 3. 回测引擎
# =============================================================================
print("\n【3. 回测引擎】")

def backtest(df, signals, transaction_cost=0.001):
    """简单回测"""
    returns = df['Returns'].values
    
    # 计算组合收益
    portfolio_returns = signals * returns
    
    # 交易成本
    trades = np.abs(np.diff(signals))
    costs = trades * transaction_cost
    portfolio_returns[1:] -= costs
    
    # 累积收益
    cumulative = np.cumprod(1 + portfolio_returns)
    
    return portfolio_returns, cumulative

def calculate_metrics(portfolio_returns):
    """计算绩效指标"""
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
    
    # Calmar
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
    
    return {
        'Annual Return': f"{annual_return:.2%}",
        'Annual Std': f"{annual_std:.2%}",
        'Sharpe': f"{sharpe:.3f}",
        'Max DD': f"{max_dd:.2%}",
        'Calmar': f"{calmar:.3f}",
    }

# =============================================================================
# 4. 运行测试
# =============================================================================
print("\n【4. 运行测试】")

results = []

for ticker in test_tickers:
    if ticker not in futures_data:
        print(f"  ⚠️ {ticker} 不存在")
        continue
    
    df = futures_data[ticker]
    
    # 分割训练/测试集
    train_mask = (df.index >= TRAIN_START) & (df.index < TRAIN_END)
    test_mask = (df.index >= TEST_START) & (df.index <= TEST_END)
    
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()
    
    print(f"\n  📊 {ticker}")
    print(f"     训练集: {len(df_train)} rows ({df_train.index[0].strftime('%Y-%m-%d')} ~ {df_train.index[-1].strftime('%Y-%m-%d')})")
    print(f"     测试集: {len(df_test)} rows ({df_test.index[0].strftime('%Y-%m-%d')} ~ {df_test.index[-1].strftime('%Y-%m-%d')})")
    
    for strategy_name, strategy_func in strategies.items():
        try:
            # 生成信号
            if strategy_name == 'Long':
                signals = strategy_func(df_test)
            elif strategy_name == 'Sign':
                signals = strategy_func(df_test, lookback=252)
            else:
                signals = strategy_func(df_test)
            
            # 回测
            portfolio_returns, cumulative = backtest(df_test, signals, TRANSACTION_COST)
            
            # 计算指标
            metrics = calculate_metrics(portfolio_returns)
            
            print(f"     {strategy_name}: Sharpe={metrics['Sharpe']}, MDD={metrics['Max DD']}, Return={metrics['Annual Return']}")
            
            results.append({
                'Ticker': ticker,
                'Strategy': strategy_name,
                **metrics
            })
            
        except Exception as e:
            print(f"     {strategy_name}: ERROR - {str(e)[:30]}")

# =============================================================================
# 5. 结果汇总
# =============================================================================
print("\n" + "=" * 80)
print("【5. 结果汇总】")
print("=" * 80)

df_results = pd.DataFrame(results)

# 按合约分组显示
for ticker in test_tickers:
    print(f"\n{ticker}:")
    ticker_results = df_results[df_results['Ticker'] == ticker]
    for _, row in ticker_results.iterrows():
        print(f"  {row['Strategy']:<8} | Sharpe: {row['Sharpe']:>6} | MDD: {row['Max DD']:>8} | Return: {row['Annual Return']:>8}")

# 保存结果
output_file = 'pilot_test_results.csv'
df_results.to_csv(output_file, index=False)
print(f"\n💾 结果已保存到: {output_file}")

print("\n" + "=" * 80)
print("✅ Pilot Test 完成！")
print("=" * 80)
print("""
下一步建议:
1. 在云端 GPU 上运行完整 DRL 训练 (drl_trading_cloud.ipynb)
2. 添加更多基线策略对比
3. 进行参数敏感性分析
4. 实现 DDPG/PPO/A2C 算法
""")
