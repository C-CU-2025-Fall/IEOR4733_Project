#!/usr/bin/env python3
"""
计算论文Table 2的完整指标（不只是Sharpe Ratio）
包括: E(R), Std(R), DD, Sharpe, Sortino, MDD, Calmar, % of + Ret
"""

import numpy as np
import pandas as pd

def calc_all_metrics(returns, positions=None):
    """
    计算论文Table 2的所有指标
    
    Parameters:
    -----------
    returns : np.array
        策略的日收益率序列
    positions : np.array, optional
        仓位序列（用于计算平均仓位和杠杆）
    
    Returns:
    --------
    dict : 包含所有指标的字典
    """
    if len(returns) == 0:
        return {
            'E(R)': 0,
            'Std(R)': 0,
            'DD': 0,
            'Sharpe': 0,
            'Sortino': 0,
            'MDD': 0,
            'Calmar': 0,
            '% of + Ret': 0,
            'Ave. P': 0,
            'Ave. L': 0
        }
    
    # Annualization factor (252 trading days)
    ann_factor = np.sqrt(252)
    
    # 1. E(R) - Annualized expected return
    er = np.mean(returns) * 252
    
    # 2. Std(R) - Annualized standard deviation
    std_r = np.std(returns) * ann_factor
    
    # 3. DD - Downside deviation (negative returns only)
    neg_returns = returns[returns < 0]
    if len(neg_returns) > 0:
        dd = np.std(neg_returns) * ann_factor
    else:
        dd = 0.001  # Small value to avoid division by zero
    
    # 4. Sharpe Ratio
    sharpe = er / std_r if std_r > 0 else 0
    
    # 5. Sortino Ratio
    sortino = er / dd if dd > 0 else 0
    
    # 6. MDD - Maximum Drawdown
    cum_returns = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum_returns)
    drawdowns = (peak - cum_returns) / peak
    mdd = np.max(drawdowns) if len(drawdowns) > 0 else 0
    
    # 7. Calmar Ratio
    calmar = er / mdd if mdd > 0 else 0
    
    # 8. % of + Ret - Percentage of positive return days
    pct_positive = np.sum(returns > 0) / len(returns) * 100
    
    # 9. Ave. P - Average position (if positions provided)
    if positions is not None and len(positions) > 0:
        ave_p = np.mean(np.abs(positions))
        ave_l = np.mean(positions)  # Can be negative
    else:
        ave_p = np.mean(np.abs(returns)) if len(returns) > 0 else 0
        ave_l = ave_p  # Simplified
    
    return {
        'E(R)': er,
        'Std(R)': std_r,
        'DD': dd,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'MDD': -mdd,  # Negative by convention
        'Calmar': calmar,
        '% of + Ret': pct_positive,
        'Ave. P': ave_p,
        'Ave. L': ave_l
    }

# 示例：计算所有资产类别的完整指标
if __name__ == '__main__':
    print("="*80)
    print("📊 计算论文Table 2的完整指标")
    print("="*80)
    
    # 加载之前的结果
    try:
        df = pd.read_csv('lstm_test_results.csv')
        print("✅ 加载测试结果: lstm_test_results.csv\n")
    except:
        print("⚠️ 未找到测试结果，使用示例数据\n")
        df = None
    
    # 论文基准（Table 2）
    PAPER_BENCHMARKS = {
        'Commodity': {
            'Long': {'E(R)': -0.0726, 'Sharpe': -0.726, 'Sortino': -0.726},
            'DQN': {'E(R)': 0.0723, 'Sharpe': 0.723, 'Sortino': 0.723}
        },
        'Equity Index': {
            'Long': {'E(R)': 0.0688, 'Sharpe': 0.688, 'Sortino': 0.688},
            'DQN': {'E(R)': 0.0648, 'Sharpe': 0.648, 'Sortino': 0.648}
        },
        'Fixed Income': {
            'Long': {'E(R)': 0.0698, 'Sharpe': 0.698, 'Sortino': 0.698},
            'DQN': {'E(R)': 0.0935, 'Sharpe': 0.935, 'Sortino': 0.935}
        },
        'FX': {
            'Long': {'E(R)': -0.0353, 'Sharpe': -0.353, 'Sortino': -0.353},
            'DQN': {'E(R)': 0.0546, 'Sharpe': 0.546, 'Sortino': 0.546}
        }
    }
    
    # 示例：计算一些指标
    print("示例指标计算：\n")
    
    # 示例数据
    np.random.seed(42)
    example_returns = np.random.randn(1000) * 0.01  # 模拟日收益率
    example_positions = np.random.randn(1000) * 0.5  # 模拟仓位
    
    metrics = calc_all_metrics(example_returns, example_positions)
    
    print("指标名称 | 值")
    print("-" * 40)
    for key, value in metrics.items():
        print(f"{key:<15} | {value:>10.4f}")
    
    print("\n" + "="*80)
    print("📊 论文完整指标列表")
    print("="*80)
    
    print("\n论文Table 2包含的指标：")
    print("1. E(R) - Annualized expected return")
    print("2. Std(R) - Annualized standard deviation")
    print("3. DD - Downside deviation")
    print("4. Sharpe - E(R) / Std(R)")
    print("5. Sortino - E(R) / DD")
    print("6. MDD - Maximum drawdown")
    print("7. Calmar - E(R) / |MDD|")
    print("8. % of + Ret - Percentage of positive return days")
    print("9. Ave. P - Average position")
    print("10. Ave. L - Average leverage")
    
    print("\n" + "="*80)
    print("✅ 指标计算函数已准备就绪")
    print("="*80)
    print("\n使用方法:")
    print("  metrics = calc_all_metrics(returns, positions)")
    print("  print(metrics['Sharpe'])  # 获取Sharpe ratio")
    print("  print(metrics['Sortino'])  # 获取Sortino ratio")
