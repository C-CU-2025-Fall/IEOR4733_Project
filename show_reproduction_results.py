#!/usr/bin/env python3
"""生成复现版结果汇总"""

import pandas as pd

# 读取结果
df = pd.read_csv('results_gamma03_20260319_191044.csv')

print("="*70)
print("📊 复现版结果汇总 (γ=0.3, 论文超参数)")
print("="*70)
print()
print("配置:")
print("  • γ (discount) = 0.3 (论文配置)")
print("  • Buffer Size = 5000 (论文配置)")
print("  • Batch Size = 64/128 (论文配置)")
print("  • 网络架构 = [64, 32] (论文配置)")
print("  • 训练步数 = 50,000")
print("  • 训练期 = 2011-2015")
print("  • 测试期 = 2016-2019")
print()
print("="*70)
print("结果:")
print("="*70)
print()

# 按合约分组
for ticker in ['ES=F', 'CL=F', 'GC=F']:
    print(f"【{ticker}】")
    ticker_df = df[df['Ticker'] == ticker]
    for _, row in ticker_df.iterrows():
        print(f"  {row['Strategy']:<8} | Sharpe: {row['Sharpe']:>7.3f} | Return: {row['Return']:>7.2%} | MDD: {row['MDD']:>7.2%}")
    print()

print("="*70)
print("平均结果:")
print("="*70)
print()

summary = df.groupby('Strategy').agg({
    'Sharpe': 'mean',
    'Return': 'mean',
    'MDD': 'mean'
}).round(3)

for strategy in ['Long', 'MACD', 'DQN', 'A2C']:
    if strategy in summary.index:
        row = summary.loc[strategy]
        print(f"{strategy:<8} | Sharpe: {row['Sharpe']:>7.3f} | Return: {row['Return']:>7.2%} | MDD: {row['MDD']:>7.2%}")

print()
print("="*70)
print("与论文对比 (All Portfolio):")
print("="*70)
print()
print(f"{'策略':<8} | {'论文':>10} | {'复现':>10} | {'差距':>10}")
print("-" * 50)

paper_results = {
    'Long': 0.06,
    'MACD': 0.09,
    'DQN': 1.29,
    'A2C': 1.05
}

for strategy in ['Long', 'MACD', 'DQN', 'A2C']:
    if strategy in summary.index:
        paper = paper_results.get(strategy, 0)
        ours = summary.loc[strategy, 'Sharpe']
        diff = ours - paper
        print(f"{strategy:<8} | {paper:>10.3f} | {ours:>10.3f} | {diff:>+10.3f}")
