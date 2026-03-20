#!/usr/bin/env python3
"""
Futures Data Quality Check
检查 42 个可用期货合约的数据质量
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json

# 42 个期货合约（基于 futures_coverage_results.json）
FUTURES_42 = {
    "Equity Indices": [
        "ES=F",   # S&P 500
        "NQ=F",   # Nasdaq 100
        "YM=F",   # Dow Jones
        "RTY=F",  # Russell 2000 (注意: 2017年开始)
        "NKD=F",  # Nikkei
    ],
    "Commodities - Energy": [
        "CL=F",   # Crude Oil
        "NG=F",   # Natural Gas
        "RB=F",   # RBOB Gasoline
        "HO=F",   # Heating Oil
        "BZ=F",   # Brent Crude
    ],
    "Commodities - Metals": [
        "GC=F",   # Gold
        "SI=F",   # Silver
        "HG=F",   # Copper
        "PL=F",   # Platinum
        "PA=F",   # Palladium
    ],
    "Commodities - Agriculture": [
        "ZC=F",   # Corn
        "ZS=F",   # Soybeans
        "ZW=F",   # Wheat
        "ZL=F",   # Soybean Oil
        "ZM=F",   # Soybean Meal
        "KC=F",   # Coffee
        "CT=F",   # Cotton
        "SB=F",   # Sugar
        "CC=F",   # Cocoa
        "OJ=F",   # Orange Juice
        "KE=F",   # Kansas Wheat (新增)
        "DC=F",   # Milk (新增)
    ],
    "Fixed Income": [
        "ZN=F",   # 10Y Treasury
        "ZB=F",   # 30Y Treasury
        "ZF=F",   # 5Y Treasury
        "ZT=F",   # 2Y Treasury
        "GE=F",   # Eurodollar
        "TN=F",   # 10Y Ultra
        "UB=F",   # 30Y Ultra
    ],
    "FX": [
        "6E=F",   # Euro
        "6J=F",   # Yen
        "6B=F",   # British Pound
        "6A=F",   # Australian Dollar
        "6C=F",   # Canadian Dollar
        "6S=F",   # Swiss Franc
        "6M=F",   # Mexican Peso
        "6N=F",   # New Zealand Dollar
        "6R=F",   # Russian Ruble
        "DX=F",   # US Dollar Index
    ],
}

START_DATE = '2011-01-01'
END_DATE = '2019-12-31'
EXPECTED_TRADING_DAYS = 2260  # 大约 2011-2019 的交易日数量

def check_data_quality(ticker):
    """检查单个合约的数据质量"""
    result = {
        'ticker': ticker,
        'available': False,
        'rows': 0,
        'start_date': None,
        'end_date': None,
        'missing_pct': 0,
        'price_issues': [],
        'coverage_pct': 0,
    }
    
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        
        if df is None or df.empty:
            return result
        
        result['available'] = True
        result['rows'] = len(df)
        result['start_date'] = df.index[0].strftime('%Y-%m-%d')
        result['end_date'] = df.index[-1].strftime('%Y-%m-%d')
        
        # 计算覆盖率
        result['coverage_pct'] = (result['rows'] / EXPECTED_TRADING_DAYS) * 100
        
        # 检查缺失值
        total_cells = len(df) * 5  # OHLCV
        missing_cells = df[['Open', 'High', 'Low', 'Close', 'Volume']].isna().sum().sum()
        result['missing_pct'] = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        # 检查价格异常
        # 1. High < Low
        high_low_issues = (df['High'] < df['Low']).sum()
        if high_low_issues > 0:
            result['price_issues'].append(f"High<Low: {high_low_issues}")
        
        # 2. 零价格
        zero_prices = ((df[['Open', 'High', 'Low', 'Close']] == 0).any(axis=1)).sum()
        if zero_prices > 0:
            result['price_issues'].append(f"Zero prices: {zero_prices}")
        
        # 3. 负价格
        negative_prices = ((df[['Open', 'High', 'Low', 'Close']] < 0).any(axis=1)).sum()
        if negative_prices > 0:
            result['price_issues'].append(f"Negative prices: {negative_prices}")
        
        # 4. 极端收益率 (>50% 单日)
        df['Returns'] = df['Close'].pct_change()
        extreme_returns = (abs(df['Returns']) > 0.5).sum()
        if extreme_returns > 0:
            result['price_issues'].append(f"Extreme returns (>50%): {extreme_returns}")
        
        # 5. 连续缺失天数
        df['is_missing'] = df['Close'].isna().astype(int)
        max_consecutive = df['is_missing'].groupby((df['is_missing'] != df['is_missing'].shift()).cumsum()).sum().max()
        if max_consecutive > 5:
            result['price_issues'].append(f"Max consecutive missing: {max_consecutive}")
        
    except Exception as e:
        result['error'] = str(e)[:100]
    
    return result

def main():
    print("=" * 80)
    print("📊 期货数据质量检查")
    print("=" * 80)
    print(f"时间范围: {START_DATE} 至 {END_DATE}")
    print(f"预期交易日: ~{EXPECTED_TRADING_DAYS}")
    print("=" * 80)
    
    all_results = []
    summary = {
        'total': 0,
        'available': 0,
        'high_quality': 0,  # coverage > 90%, missing < 1%
        'medium_quality': 0,  # coverage > 70%, missing < 5%
        'low_quality': 0,  # others
    }
    
    for asset_class, tickers in FUTURES_42.items():
        print(f"\n{'='*80}")
        print(f"📦 {asset_class} ({len(tickers)} 个合约)")
        print(f"{'='*80}")
        print(f"{'Ticker':<10} {'Rows':>6} {'Coverage':>10} {'Missing':>8} {'Start':>12} {'End':>12} {'Issues':<20}")
        print("-" * 80)
        
        for ticker in tickers:
            summary['total'] += 1
            result = check_data_quality(ticker)
            all_results.append(result)
            
            if result['available']:
                summary['available'] += 1
                
                # 质量评级
                if result['coverage_pct'] > 90 and result['missing_pct'] < 1:
                    quality = "🟢"
                    summary['high_quality'] += 1
                elif result['coverage_pct'] > 70 and result['missing_pct'] < 5:
                    quality = "🟡"
                    summary['medium_quality'] += 1
                else:
                    quality = "🔴"
                    summary['low_quality'] += 1
                
                issues_str = ", ".join(result['price_issues']) if result['price_issues'] else "✅"
                
                print(f"{ticker:<10} {result['rows']:>6} {result['coverage_pct']:>9.1f}% "
                      f"{result['missing_pct']:>7.2f}% {result['start_date']:>12} "
                      f"{result['end_date']:>12} {issues_str:<20} {quality}")
            else:
                print(f"{ticker:<10} {'N/A':>6} {'N/A':>10} {'N/A':>8} {'N/A':>12} {'N/A':>12} ❌ 不可用")
    
    # 汇总报告
    print("\n" + "=" * 80)
    print("📊 数据质量汇总")
    print("=" * 80)
    
    print(f"\n【总体统计】")
    print(f"  总合约数: {summary['total']}")
    print(f"  可用合约: {summary['available']} ({summary['available']/summary['total']*100:.1f}%)")
    
    print(f"\n【质量分级】")
    print(f"  🟢 高质量 (>90% 覆盖, <1% 缺失): {summary['high_quality']}")
    print(f"  🟡 中等 (>70% 覆盖, <5% 缺失): {summary['medium_quality']}")
    print(f"  🔴 低质量: {summary['low_quality']}")
    
    # 按资产类别统计
    print(f"\n【按资产类别统计】")
    asset_stats = {}
    for asset_class, tickers in FUTURES_42.items():
        asset_results = [r for r in all_results if r['ticker'] in tickers]
        available = sum(1 for r in asset_results if r['available'])
        avg_coverage = np.mean([r['coverage_pct'] for r in asset_results if r['available']]) if available > 0 else 0
        asset_stats[asset_class] = {
            'total': len(tickers),
            'available': available,
            'avg_coverage': avg_coverage
        }
        print(f"  {asset_class}: {available}/{len(tickers)} 可用, 平均覆盖率 {avg_coverage:.1f}%")
    
    # 问题合约
    print(f"\n【需要注意的合约】")
    issues_found = False
    for r in all_results:
        if r['available']:
            if r['coverage_pct'] < 80:
                print(f"  ⚠️ {r['ticker']}: 覆盖率仅 {r['coverage_pct']:.1f}%")
                issues_found = True
            if r['price_issues']:
                print(f"  ⚠️ {r['ticker']}: {', '.join(r['price_issues'])}")
                issues_found = True
    if not issues_found:
        print("  ✅ 所有合约数据质量良好")
    
    # 保存结果
    output = {
        'check_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'date_range': f"{START_DATE} to {END_DATE}",
        'summary': summary,
        'asset_stats': asset_stats,
        'details': all_results
    }
    
    with open('data_quality_report.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n💾 详细报告已保存到: data_quality_report.json")
    print("=" * 80)

if __name__ == "__main__":
    main()
