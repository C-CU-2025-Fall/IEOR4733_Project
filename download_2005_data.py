#!/usr/bin/env python3
"""
下载 2005-2019 期货数据

支持:
- 断点续传
- 强制重新下载
- 速率限制

用法:
    python3 download_2005_data.py              # 增量下载
    python3 download_2005_data.py --force      # 强制重新下载
"""

import os
import time
import random
import json
import pandas as pd
import yfinance as yf

# 配置
START_DATE = '2005-01-01'
END_DATE = '2019-12-31'
DATA_DIR = 'data/futures_processed'

# 速率限制
MIN_DELAY = 3.0
MAX_DELAY = 5.0
BATCH_SIZE = 5
BATCH_DELAY = 30

# 合约列表
CONTRACTS = [
    # Equity Index
    'ES=F', 'NQ=F', 'YM=F',
    # Commodities
    'CL=F', 'GC=F', 'SI=F', 'HG=F', 'NG=F', 'ZC=F', 'ZS=F', 'ZW=F', 'KC=F', 'CC=F', 'SB=F', 'CT=F', 'OJ=F',
    # Fixed Income
    'ZN=F', 'ZB=F', 'ZF=F', 'ZT=F', 'GE=F',
    # FX
    '6E=F', '6J=F', '6B=F', '6A=F', '6C=F', '6S=F', '6N=F', '6M=F', '6R=F'
]

def download_contract(ticker, force=False):
    """下载单个合约"""
    
    fname = f'{DATA_DIR}/{ticker.replace("=", "")}.csv'
    
    # 检查是否已存在
    if not force and os.path.exists(fname):
        try:
            df = pd.read_csv(fname)
            if len(df) > 3000:  # 2005-2019 大约 3750 行
                print(f"  ✅ {ticker}: 已下载 ({len(df)}行)")
                return True
        except:
            pass
    
    try:
        print(f"  📥 下载 {ticker}...")
        data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        
        if len(data) > 0:
            # 保存
            df = data.copy()
            df.to_csv(fname)
            
            first_date = df.index[0].strftime('%Y-%m-%d')
            last_date = df.index[-1].strftime('%Y-%m-%d')
            print(f"  ✅ {ticker}: {first_date} 至 {last_date} ({len(df)}行)")
            
            return True
        else:
            print(f"  ❌ {ticker}: 无数据")
            return False
            
    except Exception as e:
        print(f"  ❌ {ticker}: {e}")
        return False

def download_all(force=False):
    """下载所有合约"""
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("="*70)
    print("📥 下载 2005-2019 期货数据")
    print("="*70)
    print(f"合约数：{len(CONTRACTS)}")
    print(f"日期范围：{START_DATE} 至 {END_DATE}")
    print(f"输出目录：{DATA_DIR}/")
    if force:
        print("⚠️  强制重新下载模式")
    print("="*70)
    print()
    
    success = 0
    failed = 0
    
    for i, ticker in enumerate(CONTRACTS):
        if download_contract(ticker, force):
            success += 1
        else:
            failed += 1
        
        # 速率限制
        if (i + 1) % BATCH_SIZE == 0:
            delay = BATCH_DELAY
            print(f"\n  ⏳ 等待 {delay}秒 (速率限制)...")
            time.sleep(delay)
        else:
            delay = random.uniform(MIN_DELAY, MAX_DELAY)
            time.sleep(delay)
    
    print()
    print("="*70)
    print(f"✅ 成功：{success}/{len(CONTRACTS)}")
    if failed > 0:
        print(f"❌ 失败：{failed}/{len(CONTRACTS)}")
    print("="*70)
    
    # 保存清单
    manifest = {
        'contracts': CONTRACTS,
        'success': success,
        'failed': failed,
        'date_range': f"{START_DATE} to {END_DATE}",
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(f'{DATA_DIR}/manifest_2005.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n💾 清单：{DATA_DIR}/manifest_2005.json")

if __name__ == '__main__':
    import sys
    
    force = '--force' in sys.argv or '-f' in sys.argv
    download_all(force=force)
