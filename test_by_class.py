#!/usr/bin/env python3
"""
测试按类别训练的模型并与论文Table 2对比
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
from datetime import datetime

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型
with open('models_20260319_223619.pkl', 'rb') as f:
    models = pickle.load(f)

print("="*80)
print("📊 测试结果 vs 论文Table 2")
print("="*80)

# 论文Table 2数据
PAPER_RESULTS = {
    'Commodity': {
        'Long': {'Sharpe': -0.726, 'Sortino': -1.177, 'MDD': 0.350},
        'Sign(R)': {'Sharpe': 0.354, 'Sortino': 0.606, 'MDD': 0.116},
        'MACD': {'Sharpe': -0.175, 'Sortino': -0.293, 'MDD': 0.190},
        'DQN': {'Sharpe': 0.723, 'Sortino': 1.275, 'MDD': 0.066},
        'A2C': {'Sharpe': 0.234, 'Sortino': 0.399, 'MDD': 0.141},
    },
    'Equity Index': {
        'Long': {'Sharpe': 0.688, 'Sortino': 1.102, 'MDD': 0.132},
        'Sign(R)': {'Sharpe': 0.236, 'Sortino': 0.374, 'MDD': 0.344},
        'MACD': {'Sharpe': 0.017, 'Sortino': 0.027, 'MDD': 0.311},
        'DQN': {'Sharpe': 0.648, 'Sortino': 1.038, 'MDD': 0.161},
        'A2C': {'Sharpe': 0.510, 'Sortino': 0.798, 'MDD': 0.124},
    },
    'Fixed Income': {
        'Long': {'Sharpe': 0.698, 'Sortino': 1.180, 'MDD': 0.061},
        'Sign(R)': {'Sharpe': 0.221, 'Sortino': 0.363, 'MDD': 0.080},
        'MACD': {'Sharpe': 0.228, 'Sortino': 0.380, 'MDD': 0.065},
        'DQN': {'Sharpe': 0.935, 'Sortino': 1.617, 'MDD': 0.062},
        'A2C': {'Sharpe': 0.714, 'Sortino': 1.203, 'MDD': 0.067},
    },
    'FX': {
        'Long': {'Sharpe': -0.353, 'Sortino': -0.590, 'MDD': 0.423},
        'Sign(R)': {'Sharpe': -0.306, 'Sortino': -0.502, 'MDD': 0.434},
        'MACD': {'Sharpe': 0.007, 'Sortino': 0.011, 'MDD': 0.329},
        'DQN': {'Sharpe': 0.546, 'Sortino': 0.955, 'MDD': 0.183},
        'A2C': {'Sharpe': 0.328, 'Sortino': 0.561, 'MDD': 0.165},
    }
}

# 合约分组
CONTRACTS_BY_CLASS = {
    'Commodity': ['CL=F', 'GC=F', 'SI=F', 'HG=F', 'NG=F', 'ZC=F', 'ZS=F', 'ZW=F', 'KC=F', 'CC=F', 'SB=F', 'CT=F', 'OJ=F'],
    'Equity Index': ['ES=F', 'NQ=F', 'YM=F'],
    'Fixed Income': ['ZN=F', 'ZB=F', 'ZF=F', 'ZT=F', 'GE=F'],
    'FX': ['6E=F', '6J=F', '6B=F', '6A=F', '6C=F', '6S=F', '6N=F', '6M=F', '6R=F']
}

TRAIN_START = '2011-01-03'
TRAIN_END = '2015-12-31'
TEST_START = '2016-01-01'
TEST_END = '2019-12-31'
DATA_DIR = 'data/futures_processed'
TRANSACTION_COST = 0.002

# 简化的测试环境
def test_model(model, prices, returns, discrete=True):
    """测试模型"""
    n_steps = len(returns)
    position = 0.0
    returns_list = []
    
    for i in range(200, n_steps-1):
        # 简化状态
        obs = np.zeros(16, dtype=np.float32)
        
        # 动量特征
        for j, window in enumerate([5, 10, 25, 50, 100, 200]):
            if i >= window:
                ret = np.mean(returns[i-window:i])
                vol = np.std(returns[i-window:i])
                obs[j] = ret / (vol + 1e-8)
        
        # 预测动作
        action, _ = model.predict(obs, deterministic=True)
        
        if discrete:
            action = float(action - 1)  # {-1, 0, 1}
        else:
            action = float(action[0])
        
        # 计算收益
        cost = abs(action - position) * TRANSACTION_COST
        strat_ret = action * returns[i+1] - cost
        returns_list.append(strat_ret)
        
        position = action
    
    returns_list = np.array(returns_list)
    
    # 计算指标
    if len(returns_list) == 0:
        return {'Sharpe': 0, 'Sortino': 0, 'MDD': 0}
    
    er = np.mean(returns_list) * 252
    std_r = np.std(returns_list) * np.sqrt(252)
    sharpe = er / std_r if std_r > 0 else 0
    
    neg_ret = returns_list[returns_list < 0]
    dd = np.std(neg_ret) * np.sqrt(252) if len(neg_ret) > 0 else 0.001
    sortino = er / dd if dd > 0 else 0
    
    cum = np.cumprod(1 + returns_list)
    peak = np.maximum.accumulate(cum)
    drawdowns = (peak - cum) / peak
    mdd = np.max(drawdowns) if len(drawdowns) > 0 else 0
    
    return {
        'Sharpe': sharpe,
        'Sortino': sortino,
        'MDD': mdd
    }

# 测试每个资产类别
all_results = []

for asset_class in ['Commodity', 'Equity Index', 'Fixed Income', 'FX']:
    print(f"\n{'='*70}")
    print(f"【{asset_class}】")
    print('='*70)
    
    model_dict = models.get(asset_class)
    if not model_dict:
        print("  ⚠️ 模型不存在")
        continue
    
    tickers = CONTRACTS_BY_CLASS[asset_class]
    class_results = []
    
    for ticker in tickers:
        try:
            df = pd.read_csv(f'{DATA_DIR}/{ticker}.csv')
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            test = df[(df['Date'] >= TEST_START) & (df['Date'] <= TEST_END)]
            if len(test) < 200:
                continue
            
            prices = test['Close'].values
            returns = test['Returns'].values
            
            # 测试DQN
            dqn_metrics = test_model(model_dict['dqn'], prices, returns, discrete=True)
            dqn_metrics['Ticker'] = ticker
            dqn_metrics['Strategy'] = 'DQN'
            dqn_metrics['AssetClass'] = asset_class
            class_results.append(dqn_metrics)
            
            # 测试A2C
            a2c_metrics = test_model(model_dict['a2c'], prices, returns, discrete=False)
            a2c_metrics['Ticker'] = ticker
            a2c_metrics['Strategy'] = 'A2C'
            a2c_metrics['AssetClass'] = asset_class
            class_results.append(a2c_metrics)
            
        except Exception as e:
            continue
    
    if class_results:
        # 计算平均
        df = pd.DataFrame(class_results)
        
        print(f"\n{'策略':<10} | {'论文Sharpe':>10} | {'我们Sharpe':>10} | {'差距':>10} | {'状态':>10}")
        print("-" * 60)
        
        for strategy in ['DQN', 'A2C']:
            strat_df = df[df['Strategy'] == strategy]
            if len(strat_df) > 0:
                our_sharpe = strat_df['Sharpe'].mean()
                paper_sharpe = PAPER_RESULTS[asset_class][strategy]['Sharpe']
                diff = our_sharpe - paper_sharpe
                status = '✅' if abs(diff) < 0.3 else ('⚠️' if abs(diff) < 1.0 else '❌')
                
                print(f"{strategy:<10} | {paper_sharpe:>10.3f} | {our_sharpe:>10.3f} | {diff:>+10.3f} | {status:>10}")
        
        all_results.extend(class_results)

# 保存结果
df = pd.DataFrame(all_results)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'results_by_class_{timestamp}.csv'
df.to_csv(filename, index=False)

print(f"\n\n{'='*80}")
print("📊 总体对比")
print("="*80)

# 按资产类别汇总
for asset_class in ['Commodity', 'Equity Index', 'Fixed Income', 'FX']:
    class_df = df[df['AssetClass'] == asset_class]
    if len(class_df) == 0:
        continue
    
    print(f"\n【{asset_class}】")
    for strategy in ['DQN', 'A2C']:
        strat_df = class_df[class_df['Strategy'] == strategy]
        if len(strat_df) > 0:
            mean_sharpe = strat_df['Sharpe'].mean()
            paper = PAPER_RESULTS[asset_class][strategy]['Sharpe']
            diff_pct = (mean_sharpe - paper) / abs(paper) * 100 if paper != 0 else 0
            print(f"  {strategy}: {mean_sharpe:.3f} vs 论文 {paper:.3f} ({diff_pct:+.1f}%)")

print(f"\n💾 结果已保存: {filename}")
