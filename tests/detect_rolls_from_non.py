"""
从 NON 数据检测换月日期

假设：换月时 NON 数据会有价格跳空（因为合约切换）
方法：检测 NON 价格的异常跳空，推断 roll dates
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'CLC'

TEST_START = pd.Timestamp('2011-01-01')
TEST_END = pd.Timestamp('2019-12-31')

D_CONTRACTS = ['NR', 'SB', 'SN', 'SP', 'TY', 'US', 'ZA', 'ZC', 'ZF', 'ZG', 'ZH', 'ZI', 'ZK', 'ZL', 'ZN', 'ZO', 'ZP', 'ZR', 'ZT', 'ZU', 'ZW', 'ZZ']


def detect_rolls_from_non(symbol):
    """从 NON 数据检测换月"""
    non_file = DATA_DIR / f'{symbol}_NON.CSV'
    rollover_file = DATA_DIR / f'{symbol}_rollovers.csv'
    
    if not non_file.exists():
        return None
    
    # 读取 NON
    non = pd.read_csv(non_file, names=['Date','O','H','L','C','V','OI'])
    non['Date'] = pd.to_datetime(non['Date'])
    non = non.sort_values('Date').reset_index(drop=True)
    
    # 计算价格变化
    non['ret'] = non['C'].pct_change()
    
    # 检测异常跳空（>5%）
    threshold = 0.05
    jumps = non[non['ret'].abs() > threshold].copy()
    
    # 读取实际 rollover 日期对比
    if rollover_file.exists():
        rollover = pd.read_csv(rollover_file)
        rollover['RollDate'] = pd.to_datetime(rollover['RollDate'])
        actual_rolls = set(rollover['RollDate'])
        
        # 过滤测试期
        test_rolls = [r for r in actual_rolls if TEST_START <= r <= TEST_END]
        
        # 检测到的跳空日期
        detected_jumps = set(jumps['Date'])
        test_jumps = [d for d in detected_jumps if TEST_START <= d <= TEST_END]
        
        # 匹配
        matched = len(set(test_jumps) & set(test_rolls))
        
        return {
            'Symbol': symbol,
            'Test_Rolls': len(test_rolls),
            'Test_Jumps': len(test_jumps),
            'Matched': matched,
            'Match_Rate': matched / len(test_rolls) * 100 if test_rolls else 0,
        }
    else:
        return {
            'Symbol': symbol,
            'Test_Jumps': len(jumps[(jumps['Date'] >= TEST_START) & (jumps['Date'] <= TEST_END)]),
        }


def detect_rolls_from_rad_non_ratio(symbol):
    """从 RAD/NON ratio 检测换月"""
    non_file = DATA_DIR / f'{symbol}_NON.CSV'
    rad_file = DATA_DIR / f'{symbol}_RAD.CSV'
    rollover_file = DATA_DIR / f'{symbol}_rollovers.csv'
    
    if not all([non_file.exists(), rad_file.exists()]):
        return None
    
    # 读取数据
    non = pd.read_csv(non_file, names=['Date','O','H','L','C','V','OI'])
    non['Date'] = pd.to_datetime(non['Date'])
    non['C'] = pd.to_numeric(non['C'])
    
    rad = pd.read_csv(rad_file, names=['Date','O','H','L','C','V','OI'])
    rad['Date'] = pd.to_datetime(rad['Date'])
    rad['C'] = pd.to_numeric(rad['C'])
    
    # 合并计算 ratio
    merged = pd.merge(non, rad, on='Date', suffixes=('_non', '_rad'))
    merged['ratio'] = merged['C_rad'] / merged['C_non']
    
    # 检测 ratio 跳变
    merged['ratio_diff'] = merged['ratio'].diff().abs()
    
    # 换月时 ratio 会跳变
    rolls_detected = merged[merged['ratio_diff'] > 0.01].copy()
    
    # 与实际 rollover 对比
    if rollover_file.exists():
        rollover = pd.read_csv(rollover_file)
        rollover['RollDate'] = pd.to_datetime(rollover['RollDate'])
        actual_rolls = set(rollover['RollDate'])
        
        test_rolls = [r for r in actual_rolls if TEST_START <= r <= TEST_END]
        detected_rolls = set(rolls_detected['Date'])
        test_detected = [d for d in detected_rolls if TEST_START <= d <= TEST_END]
        
        matched = len(set(test_detected) & set(test_rolls))
        
        return {
            'Symbol': symbol,
            'Actual_Rolls': len(test_rolls),
            'Detected_Rolls': len(test_detected),
            'Matched': matched,
            'Match_Rate': matched / len(test_rolls) * 100 if test_rolls else 0,
        }
    
    return None


def main():
    print("="*100)
    print("从 NON/RAD 数据检测换月日期")
    print("="*100)
    
    print("\n方法 1: NON 价格跳空检测")
    print("-"*60)
    
    results1 = []
    for symbol in D_CONTRACTS:
        result = detect_rolls_from_non(symbol)
        if result:
            results1.append(result)
            if 'Match_Rate' in result:
                print(f"{symbol}: 实际={result['Test_Rolls']}, 检测到={result['Test_Jumps']}, 匹配={result['Matched']} ({result['Match_Rate']:.1f}%)")
            else:
                print(f"{symbol}: 检测到={result['Test_Jumps']} 跳空")
    
    print("\n方法 2: RAD/NON ratio 跳变检测")
    print("-"*60)
    
    results2 = []
    for symbol in D_CONTRACTS:
        result = detect_rolls_from_rad_non_ratio(symbol)
        if result:
            results2.append(result)
            print(f"{symbol}: 实际={result['Actual_Rolls']}, 检测到={result['Detected_Rolls']}, 匹配={result['Matched']} ({result['Match_Rate']:.1f}%)")
    
    # 详细分析 SP（高相关性）
    print("\n" + "="*100)
    print("SP 合约详细分析")
    print("="*100)
    
    non = pd.read_csv(DATA_DIR / 'SP_NON.CSV', names=['Date','O','H','L','C','V','OI'])
    non['Date'] = pd.to_datetime(non['Date'])
    non['C'] = pd.to_numeric(non['C'])
    
    rad = pd.read_csv(DATA_DIR / 'SP_RAD.CSV', names=['Date','O','H','L','C','V','OI'])
    rad['Date'] = pd.to_datetime(rad['Date'])
    rad['C'] = pd.to_numeric(rad['C'])
    
    merged = pd.merge(non, rad, on='Date', suffixes=('_non', '_rad'))
    merged['ratio'] = merged['C_rad'] / merged['C_non']
    merged['ratio_diff'] = merged['ratio'].diff().abs()
    
    # 过滤测试期
    test = merged[(merged['Date'] >= TEST_START) & (merged['Date'] <= TEST_END)]
    
    # 显示前 10 次换月
    rolls = test[test['ratio_diff'] > 0.01].head(10)
    print("\n前 10 次 ratio 跳变（换月）:")
    print(rolls[['Date', 'C_non', 'C_rad', 'ratio', 'ratio_diff']].to_string())


if __name__ == '__main__':
    main()
