#!/usr/bin/env python3
"""详细检查数据对齐情况"""

import os
import pandas as pd

# 论文 Appendix A - 50个合约
PAPER_CONTRACTS = {
    'Commodity': [
        'CL=F',   # Crude Oil
        'GC=F',   # Gold
        'SI=F',   # Silver
        'HG=F',   # Copper
        'NG=F',   # Natural Gas
        'ZC=F',   # Corn
        'ZS=F',   # Soybeans
        'ZW=F',   # Wheat
        'KC=F',   # Coffee
        'CC=F',   # Cocoa
        'SB=F',   # Sugar
        'CT=F',   # Cotton
        'LC=F',   # Live Cattle
        'LBS=F',  # Lumber
        'OJ=F',   # Orange Juice
    ],
    'Equity Index': [
        'ES=F',   # S&P 500
        'NQ=F',   # Nasdaq 100
        'YM=F',   # Dow Jones
        'RTY=F',  # Russell 2000
        'EMD=F',  # E-mini S&P MidCap 400
        'VA=F',   # Value Line
        'FDAX.F', # DAX
        'FTI.F',  # FTSE MIB
        'FCE.F',  # CAC 40
        'FBTP.F', # BTP
    ],
    'Fixed Income': [
        'ZN=F',   # 10-Year T-Note
        'ZB=F',   # 30-Year T-Bond
        'ZF=F',   # 5-Year T-Note
        'ZT=F',   # 2-Year T-Note
        'GE=F',   # Eurodollar
        'FGBL.F', # Bund
        'FGBM.F', # BOBL
        'FGBX.F', # Swiss
        'FBTP.F', # BTP (Italian)
        'FOAT.F', # OAT (French)
    ],
    'FX': [
        '6E=F',   # Euro
        '6J=F',   # Japanese Yen
        '6B=F',   # British Pound
        '6A=F',   # Australian Dollar
        '6C=F',   # Canadian Dollar
        '6S=F',   # Swiss Franc
        '6N=F',   # New Zealand Dollar
        '6M=F',   # Mexican Peso
        '6L=F',   # Brazilian Real
        '6R=F',   # Russian Ruble
        '6Z=F',   # South African Rand
        '6H=F',   # Czech Koruna
        '6O=F',   # Polish Zloty
        '6A=F',   # Australian Dollar (duplicate in paper?)
        '6C=F',   # Canadian Dollar (duplicate in paper?)
    ]
}

print("="*80)
print("📊 论文数据对齐检查")
print("="*80)
print(f"\n论文要求: 50个合约 (2005-2019, Pinnacle CLC)")
print(f"我们的数据: Yahoo Finance (下载到 data/futures_processed/)")
print()

total_paper = sum(len(v) for v in PAPER_CONTRACTS.values())
print(f"论文合约总数: {total_paper}")
print()

# 检查每个类别
available_by_class = {}
total_available = 0
total_missing = 0

for asset_class, tickers in PAPER_CONTRACTS.items():
    print(f"\n{'='*70}")
    print(f"【{asset_class}】({len(tickers)} 合约)")
    print('='*70)
    
    available = []
    missing = []
    
    for ticker in tickers:
        path = f'data/futures_processed/{ticker}.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Date'] = pd.to_datetime(df['Date'])
            start = df['Date'].min().strftime('%Y-%m-%d')
            end = df['Date'].max().strftime('%Y-%m-%d')
            years = len(df) / 252
            
            # 检查是否有足够数据（2011-2019测试期）
            test_start = pd.Timestamp('2011-01-01')
            test_end = pd.Timestamp('2019-12-31')
            train_start = pd.Timestamp('2011-01-03')
            train_end = pd.Timestamp('2015-12-31')
            
            has_train = len(df[(df['Date'] >= train_start) & (df['Date'] <= train_end)]) > 500
            has_test = len(df[(df['Date'] >= test_start) & (df['Date'] <= test_end)]) > 200
            
            status = '✅' if (has_train and has_test) else '⚠️'
            
            print(f'  {status} {ticker:<10} | {start} ~ {end} | {years:.1f}年', end='')
            
            if has_train and has_test:
                available.append(ticker)
                print()
            else:
                print(f' | 训练{"✓" if has_train else "✗"} 测试{"✓" if has_test else "✗"}')
        else:
            print(f'  ❌ {ticker:<10} | 文件不存在')
            missing.append(ticker)
    
    available_by_class[asset_class] = available
    total_available += len(available)
    total_missing += len(missing)
    
    print(f'\n  小计: {len(available)}/{len(tickers)} 可用')

# 汇总
print("\n" + "="*80)
print("📊 总体汇总")
print("="*80)
print(f"\n总合约数: {total_paper}")
print(f"可用: {total_available}")
print(f"缺失: {total_missing}")
print(f"对齐度: {total_available/total_paper*100:.1f}%")

print("\n按类别汇总:")
for asset_class, available in available_by_class.items():
    paper_count = len(PAPER_CONTRACTS[asset_class])
    print(f"  {asset_class:<15} | {len(available):>2}/{paper_count:<2} | {len(available)/paper_count*100:>5.1f}%")

# 列出缺失的合约
print("\n" + "="*80)
print("❌ 缺失的合约列表")
print("="*80)

all_missing = []
for asset_class, tickers in PAPER_CONTRACTS.items():
    available = set(available_by_class[asset_class])
    missing = [t for t in tickers if t not in available]
    if missing:
        print(f"\n【{asset_class}】({len(missing)} 个)")
        for t in missing:
            all_missing.append(t)
            print(f"  - {t}")

print(f"\n总计缺失: {len(all_missing)} 个合约")

# 分析原因
print("\n" + "="*80)
print("🔍 缺失原因分析")
print("="*80)
print("""
1. 欧洲期货 (.F 后缀)
   - FDAX.F, FTI.F, FCE.F, FBTP.F (欧洲股指)
   - FGBL.F, FGBM.F, FGBX.F, FOAT.F (欧洲国债)
   → Yahoo Finance 不提供或ticker不同

2. 小众商品
   - LC=F (Live Cattle)
   - LBS=F (Lumber)
   → Yahoo Finance 数据不全

3. 新兴市场货币
   - 6L=F (Brazilian Real)
   - 6Z=F (South African Rand)
   - 6H=F (Czech Koruna)
   - 6O=F (Polish Zloty)
   → Yahoo Finance 不提供或流动性低

4. 其他
   - EMD=F, VA=F (美股指数)
   → 可能ticker不同或已停牌
""")

# 建议
print("="*80)
print("💡 建议")
print("="*80)
print(f"""
选项A: 使用现有的 {total_available} 个合约
  - 优点: 立即可用
  - 缺点: 覆盖率 {total_available/total_paper*100:.1f}%

选项B: 尝试获取缺失的 {total_missing} 个合约
  - 从其他数据源（Bloomberg, Quandl）
  - 或使用替代ticker

选项C: 说明数据限制
  - 在报告中注明数据源差异
  - 重点展示可复现的部分
""")
