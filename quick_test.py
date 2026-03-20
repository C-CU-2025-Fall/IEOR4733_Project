#!/usr/bin/env python3
"""
快速测试已训练的模型并对比论文Table 2
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
import pickle

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 论文Table 2
PAPER = {
    'Commodity': {'Long': -0.726, 'DQN': 0.723, 'A2C': 0.234},
    'Equity Index': {'Long': 0.688, 'DQN': 0.648, 'A2C': 0.510},
    'Fixed Income': {'Long': 0.698, 'DQN': 0.935, 'A2C': 0.714},
    'FX': {'Long': -0.353, 'DQN': 0.546, 'A2C': 0.328}
}

CONTRACTS = {
    'Commodity': ['CL=F', 'GC=F', 'SI=F', 'HG=F', 'NG=F', 'ZC=F', 'ZS=F', 'ZW=F', 'KC=F', 'CC=F', 'SB=F', 'CT=F', 'OJ=F'],
    'Equity Index': ['ES=F', 'NQ=F', 'YM=F'],
    'Fixed Income': ['ZN=F', 'ZB=F', 'ZF=F', 'ZT=F', 'GE=F'],
    'FX': ['6E=F', '6J=F', '6B=F', '6A=F', '6C=F', '6S=F', '6N=F', '6M=F', '6R=F']
}

def test_strategy(prices, returns, positions):
    """测试策略"""
    strat_ret = returns * positions
    er = np.mean(strat_ret) * 252
    std = np.std(strat_ret) * np.sqrt(252)
    sharpe = er / std if std > 0 else 0
    return sharpe

print("="*80)
print("📊 快速测试 - 使用已训练模型")
print("="*80)

# 加载之前训练的结果
df = pd.read_csv('results_all33_20260319_203604.csv')

print("\n按资产类别汇总:\n")

for asset_class in ['Commodity', 'Equity Index', 'Fixed Income', 'FX']:
    print(f"\n【{asset_class}】")
    print(f"{'策略':<10} | {'我们':>8} | {'论文':>8} | {'差距':>8} | {'状态':>10}")
    print("-" * 50)
    
    tickers = CONTRACTS[asset_class]
    class_df = df[df['Ticker'].isin(tickers)]
    
    if len(class_df) == 0:
        print("  ⚠️ 无数据")
        continue
    
    for strategy in ['Long', 'DQN', 'A2C']:
        strat_df = class_df[class_df['Strategy'] == strategy]
        
        if len(strat_df) > 0:
            ours = strat_df['Sharpe'].mean()
            paper = PAPER[asset_class][strategy]
            diff = ours - paper
            status = '✅' if abs(diff) < 0.3 else ('⚠️' if abs(diff) < 1.0 else '❌')
            
            print(f"{strategy:<10} | {ours:>8.3f} | {paper:>8.3f} | {diff:>+8.3f} | {status:>10}")

print("\n\n" + "="*80)
print("📊 总结")
print("="*80)

print("\n最接近论文的:")
for asset_class in ['Commodity', 'Equity Index', 'Fixed Income', 'FX']:
    tickers = CONTRACTS[asset_class]
    class_df = df[df['Ticker'].isin(tickers)]
    
    for strategy in ['Long', 'DQN', 'A2C']:
        strat_df = class_df[class_df['Strategy'] == strategy]
        if len(strat_df) > 0:
            ours = strat_df['Sharpe'].mean()
            paper = PAPER[asset_class][strategy]
            diff = abs(ours - paper)
            if diff < 0.5:
                print(f"  {asset_class} {strategy}: {ours:.3f} vs {paper:.3f} (差{diff:.3f}) ⚠️")
