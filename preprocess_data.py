#!/usr/bin/env python3
"""
期货数据预处理
- 处理 yfinance 多行标题
- 计算收益率
- 标准化格式
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

DATA_DIR = 'data/futures'
PROCESSED_DIR = 'data/futures_processed'
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("=" * 80)
print("📊 2️⃣ 数据预处理")
print("=" * 80)

files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
print(f"\n处理 {len(files)} 个文件...")

processed = 0
failed = []

for f in sorted(files):
    ticker = f.replace('.csv', '')
    filepath = os.path.join(DATA_DIR, f)
    output_path = os.path.join(PROCESSED_DIR, f)
    
    try:
        # yfinance 下载的文件有 3 行标题，需要跳过
        # Row 1: Price,Close,High,Low,Open,Volume
        # Row 2: Ticker,ES=F,ES=F,...
        # Row 3: Date,,,,,
        df = pd.read_csv(filepath, skiprows=3, names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        
        # 删除 NaN 行
        df = df.dropna()
        
        # 确保数值类型
        for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        # 计算收益率
        df['Returns'] = df['Close'].pct_change()
        
        # 计算波动率 (20日滚动标准差)
        df['Volatility_20d'] = df['Returns'].rolling(20).std()
        
        # 保存处理后的文件
        df.to_csv(output_path)
        
        processed += 1
        print(f"  ✅ {ticker}: {len(df)} rows")
        
    except Exception as e:
        failed.append(f"{ticker}: {str(e)[:50]}")
        print(f"  ❌ {ticker}: {str(e)[:50]}")

print(f"\n【处理结果】")
print(f"  成功: {processed}/{len(files)}")
print(f"  失败: {len(failed)}")

if failed:
    print(f"\n失败列表:")
    for f in failed:
        print(f"  {f}")

# 验证处理后的数据
print(f"\n【验证处理后的数据】")
sample_file = os.path.join(PROCESSED_DIR, 'ES=F.csv')
df_sample = pd.read_csv(sample_file, index_col=0, parse_dates=True)
print(f"  样本: ES=F")
print(f"  行数: {len(df_sample)}")
print(f"  列: {list(df_sample.columns)}")
print(f"  时间范围: {df_sample.index[0]} ~ {df_sample.index[-1]}")
print(f"  收益率范围: {df_sample['Returns'].min():.4f} ~ {df_sample['Returns'].max():.4f}")

print("\n" + "=" * 80)
print(f"✅ 预处理完成！处理后的数据保存在: {PROCESSED_DIR}/")
print("=" * 80)
