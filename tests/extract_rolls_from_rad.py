"""
从 RAD/NON ratio 跳变反推 roll dates

方法：
1. 计算 RAD/NON ratio
2. 检测 ratio 跳变点（换月日）
3. 用这些日期生成 RAD_v2
4. 与 vendor RAD 对比验证

如果成功 → 说明可以只用 NON+RAD 反推 roll dates，不需要 ASC
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'CLC'

TEST_START = pd.Timestamp('2011-01-01')
TEST_END = pd.Timestamp('2019-12-31')

D_CONTRACTS = ['NR', 'SB', 'SN', 'SP', 'TY', 'US', 'ZA', 'ZC', 'ZF', 'ZG', 'ZH', 'ZI', 'ZK', 'ZL', 'ZN', 'ZO', 'ZP', 'ZR', 'ZT', 'ZU', 'ZW', 'ZZ']


def extract_rolls_and_generate(symbol):
    """从 RAD/NON 提取 roll dates 并生成 RAD_v2"""
    non_file = DATA_DIR / f'{symbol}_NON.CSV'
    rad_file = DATA_DIR / f'{symbol}_RAD.CSV'
    
    if not all([non_file.exists(), rad_file.exists()]):
        return None
    
    # 读取数据
    non = pd.read_csv(non_file, names=['Date','O','H','L','C','V','OI'])
    non['Date'] = pd.to_datetime(non['Date'])
    non['C'] = pd.to_numeric(non['C'])
    non = non.sort_values('Date').reset_index(drop=True)
    
    rad = pd.read_csv(rad_file, names=['Date','O','H','L','C','V','OI'])
    rad['Date'] = pd.to_datetime(rad['Date'])
    rad['C'] = pd.to_numeric(rad['C'])
    
    # 合并计算 ratio
    merged = pd.merge(non, rad, on='Date', suffixes=('_non', '_rad'))
    merged = merged.sort_values('Date').reset_index(drop=True)
    merged['ratio'] = merged['C_rad'] / merged['C_non']
    
    # 检测 ratio 跳变
    merged['ratio_diff'] = merged['ratio'].diff().abs()
    
    # 换月检测阈值（>1% 变化）
    threshold = 0.01
    rolls = merged[merged['ratio_diff'] > threshold].copy()
    
    if len(rolls) == 0:
        return {'Symbol': symbol, 'Status': 'NO_ROLLS_DETECTED'}
    
    # 提取 roll dates 和 ratio 变化
    roll_dates = []
    for _, row in rolls.iterrows():
        roll_dates.append({
            'date': row['Date'],
            'idx': row.name,
            'ratio_change': row['ratio'] / (row['ratio'] / row['ratio_diff'] * (row['ratio'] - row['ratio_diff'])) if row['ratio_diff'] > 0 else 1.0,
            'new_ratio': row['ratio'],
        })
    
    # 过滤测试期
    test_rolls = [r for r in roll_dates if TEST_START <= r['date'] <= TEST_END]
    
    if len(test_rolls) == 0:
        return {'Symbol': symbol, 'Status': 'NO_ROLLS_IN_TEST'}
    
    # 用提取的 roll dates 生成 RAD_v2
    non['rad_v2'] = non['C']
    current_ratio = 1.0
    
    # 找到测试期开始前的 ratio
    pre_test = merged[merged['Date'] < TEST_START]
    if len(pre_test) > 0:
        current_ratio = pre_test['ratio'].iloc[-1]
    
    for i, roll in enumerate(test_rolls):
        idx = roll['idx']
        # 从当前 ratio 更新到新的 ratio
        new_ratio = roll['new_ratio']
        
        next_idx = test_rolls[i + 1]['idx'] if i + 1 < len(test_rolls) else len(non)
        non.loc[idx:next_idx - 1, 'rad_v2'] = non.loc[idx:next_idx - 1, 'C'] * new_ratio
    
    # 对比 vendor RAD
    test_data = non[(non['Date'] >= TEST_START) & (non['Date'] <= TEST_END)]
    rad_test = rad[(rad['Date'] >= TEST_START) & (rad['Date'] <= TEST_END)]
    
    merged_test = pd.merge(test_data, rad_test, on='Date', suffixes=('_v2', '_vendor'))
    
    if len(merged_test) < 100:
        return {'Symbol': symbol, 'Status': 'INSUFFICIENT_DATA'}
    
    corr = merged_test['rad_v2'].corr(merged_test['C_vendor'])
    
    return {
        'Symbol': symbol,
        'Status': 'OK',
        'Total_Rolls': len(roll_dates),
        'Test_Rolls': len(test_rolls),
        'Test_Days': len(merged_test),
        'Correlation': corr,
    }


def main():
    print("="*100)
    print("从 RAD/NON ratio 反推 roll dates 并生成 RAD_v2")
    print("="*100)
    
    results = []
    
    for symbol in D_CONTRACTS:
        result = extract_rolls_and_generate(symbol)
        if result:
            results.append(result)
            
            if result['Status'] == 'OK':
                corr_str = f"{result['Correlation']:.4f}"
                status = "✅" if result['Correlation'] >= 0.99 else "⚠️" if result['Correlation'] >= 0.90 else "❌"
                print(f"{symbol}: {status} 换月={result['Test_Rolls']}, 天数={result['Test_Days']}, corr={corr_str}")
            else:
                print(f"{symbol}: {result['Status']}")
    
    # 保存
    df = pd.DataFrame(results)
    output_file = PROJECT_ROOT / 'tests' / 'results' / 'rolls_from_rad.csv'
    df.to_csv(output_file, index=False)
    print(f"\n详细结果：{output_file}")
    
    # 统计
    print("\n" + "="*100)
    print("统计摘要")
    print("="*100)
    
    ok_df = df[df['Status'] == 'OK']
    
    if len(ok_df) > 0:
        print(f"成功生成：{len(ok_df)}/{len(df)}")
        valid_corr = ok_df[ok_df['Correlation'].notna()]
        if len(valid_corr) > 0:
            print(f"\n相关性分布 (n={len(valid_corr)}):")
            print(f"  =1.0000: {len(valid_corr[valid_corr['Correlation']>=0.9999])}")
            print(f"  ≥0.99: {len(valid_corr[valid_corr['Correlation']>=0.99])}")
            print(f"  0.95-0.99: {len(valid_corr[(valid_corr['Correlation']>=0.95) & (valid_corr['Correlation']<0.99)])}")
            print(f"  <0.95: {len(valid_corr[valid_corr['Correlation']<0.95])}")
            print(f"  中位数：{valid_corr['Correlation'].median():.4f}")
    
    # 显示 SP 详情
    print(f"\n=== SP 合约详情 ===")
    sp = results[[i for i, r in enumerate(results) if r['Symbol'] == 'SP'][0]] if any(r['Symbol'] == 'SP' for r in results) else None
    if sp and sp['Status'] == 'OK':
        print(f"换月数：{sp['Test_Rolls']}")
        print(f"天数：{sp['Test_Days']}")
        print(f"相关性：{sp['Correlation']:.6f}")


if __name__ == '__main__':
    main()
