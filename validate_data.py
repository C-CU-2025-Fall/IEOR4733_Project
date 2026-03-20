#!/usr/bin/env python3
"""
验证下载的期货数据
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

DATA_DIR = 'data/futures'
START_DATE = '2011-01-01'
END_DATE = '2019-12-31'

print("=" * 80)
print("📊 1️⃣ 验证下载数据")
print("=" * 80)

# 读取所有 CSV 文件
files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
print(f"\n总文件数: {len(files)}")

issues = []
stats = []

for f in sorted(files):
    ticker = f.replace('.csv', '')
    filepath = os.path.join(DATA_DIR, f)
    
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        # 基本信息
        rows = len(df)
        cols = list(df.columns)
        start = df.index[0].strftime('%Y-%m-%d')
        end = df.index[-1].strftime('%Y-%m-%d')
        
        # 检查必要列
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [c for c in required if c not in cols]
        
        # 检查缺失值
        missing_pct = df[required].isna().sum().sum() / (rows * 5) * 100 if rows > 0 else 0
        
        # 检查价格异常
        price_issues = 0
        if 'High' in cols and 'Low' in cols:
            price_issues = (df['High'] < df['Low']).sum()
        
        stats.append({
            'ticker': ticker,
            'rows': rows,
            'start': start,
            'end': end,
            'missing_pct': missing_pct,
            'price_issues': price_issues
        })
        
        if missing_cols or price_issues > 0:
            issues.append(f"{ticker}: missing_cols={missing_cols}, price_issues={price_issues}")
            
    except Exception as e:
        issues.append(f"{ticker}: ERROR - {str(e)[:50]}")

# 打印统计
df_stats = pd.DataFrame(stats)
print(f"\n【数据统计】")
print(f"  平均行数: {df_stats['rows'].mean():.0f}")
print(f"  行数范围: {df_stats['rows'].min()} - {df_stats['rows'].max()}")
print(f"  平均缺失: {df_stats['missing_pct'].mean():.2f}%")
print(f"  价格异常: {df_stats['price_issues'].sum()} 条")

# 按行数排序显示
print(f"\n【按行数排序（前5最少）】")
for _, row in df_stats.nsmallest(5, 'rows').iterrows():
    print(f"  {row['ticker']}: {row['rows']} rows ({row['start']} ~ {row['end']})")

if issues:
    print(f"\n⚠️ 问题 ({len(issues)}):")
    for issue in issues[:10]:
        print(f"  {issue}")
else:
    print(f"\n✅ 所有数据验证通过！")

print("=" * 80)
