"""
使用 ASC 提取的实际换月日期生成 RAD_v2 (修复版)

修复点：换月日当天仍用旧 ratio，次日开始用新 ratio
这与 CLC 官方实现一致
"""

import pandas as pd
from pathlib import Path

TEMP_DIR = Path('config/TEMP')
DATA_DIR = Path('data/CLC')
TEST_START = pd.Timestamp('2011-01-01')
TEST_END = pd.Timestamp('2019-12-31')


def generate_rad_v2_fixed(symbol):
    """生成 RAD_v2_fixed"""
    asc_file = TEMP_DIR / f'{symbol}_CLC.ASC'
    non_file = DATA_DIR / f'{symbol}_NON.CSV'
    rollover_file = DATA_DIR / f'{symbol}_rollovers.csv'
    output_file = DATA_DIR / f'{symbol}_RAD_v2_fixed.CSV'
    
    if not all(f.exists() for f in [asc_file, non_file, rollover_file]):
        return {'status': 'MISSING_FILES'}
    
    # 读取换月数据
    rollover_df = pd.read_csv(rollover_file)
    rollover_df['RollDate'] = pd.to_datetime(rollover_df['RollDate'])
    
    # 读取 NON
    non = pd.read_csv(non_file, names=['Date', 'O', 'H', 'L', 'C', 'V', 'OI'])
    non['Date'] = pd.to_datetime(non['Date'])
    non = non.sort_values('Date').reset_index(drop=True)
    
    if len(non) == 0:
        return {'status': 'EMPTY_NON'}
    
    # 找到测试期开始索引
    test_start_mask = non['Date'] >= TEST_START
    if not test_start_mask.any():
        return {'status': 'NO_TEST_DATA'}
    test_start_idx = test_start_mask.idxmax()
    
    # 匹配换月日期到 NON 数据
    rolls = []
    for _, row in rollover_df.iterrows():
        roll_date = row['RollDate']
        idx = non[non['Date'] == roll_date].index
        if len(idx) > 0:
            rolls.append({'idx': idx[0], 'c': row['PrevClose_c'], 'C': row['NewClose_C']})
        else:
            diffs = (non['Date'] - roll_date).abs()
            if diffs.min().days <= 5:
                rolls.append({'idx': diffs.idxmin(), 'c': row['PrevClose_c'], 'C': row['NewClose_C']})
    
    if len(rolls) == 0:
        return {'status': 'NO_ROLLS_MATCHED'}
    
    # 过滤测试期内的换月
    rolls_after_test = [r for r in rolls if r['idx'] >= test_start_idx]
    if len(rolls_after_test) == 0:
        return {'status': 'NO_ROLLS_IN_TEST'}
    
    # 生成 RAD - 换月日 +1 开始应用新 ratio
    non['ratio'] = 1.0
    current_ratio = 1.0
    for i, roll in enumerate(rolls_after_test):
        idx = roll['idx']
        c, C = roll['c'], roll['C']
        if C > 0 and c > 0:
            current_ratio *= (c / C)
            next_idx = rolls_after_test[i + 1]['idx'] if i + 1 < len(rolls_after_test) else len(non)
            non.loc[idx + 1:next_idx, 'ratio'] = current_ratio
    
    non['RAD_Open'] = non['O'] * non['ratio']
    non['RAD_High'] = non['H'] * non['ratio']
    non['RAD_Low'] = non['L'] * non['ratio']
    non['RAD_Close'] = non['C'] * non['ratio']
    
    # 输出测试期数据
    test_data = non[(non['Date'] >= TEST_START) & (non['Date'] <= TEST_END)].copy()
    output_df = test_data[['Date', 'RAD_Open', 'RAD_High', 'RAD_Low', 'RAD_Close', 'V', 'OI']].copy()
    output_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInterest']
    output_df['Date'] = output_df['Date'].dt.strftime('%m/%d/%Y')
    output_df.to_csv(output_file, index=False, header=False)
    
    # 验证与 CLC RAD 对比
    rad_file = DATA_DIR / f'{symbol}_RAD.CSV'
    if rad_file.exists():
        rad = pd.read_csv(rad_file, names=['Date', 'O', 'H', 'L', 'C', 'V', 'OI'])
        rad['Date'] = pd.to_datetime(rad['Date'])
        rad_test = rad[(rad['Date'] >= TEST_START) & (rad['Date'] <= TEST_END)].reset_index(drop=True)
        
        if len(rad_test) > 0 and len(test_data) > 0:
            min_len = min(len(rad_test), len(test_data))
            corr = test_data['RAD_Close'].iloc[:min_len].corr(rad_test['C'].iloc[:min_len])
            
            # 对齐因子
            scale = rad_test.iloc[0]['C'] / test_data['RAD_Close'].iloc[0]
            
            return {
                'status': 'OK',
                'rolls': len(rolls_after_test),
                'rows': len(test_data),
                'corr': corr,
                'scale': scale,
                'first_date': test_data['Date'].iloc[0],
                'first_rad': rad_test.iloc[0]['C'],
                'first_v2': test_data['RAD_Close'].iloc[0]
            }
    
    return {'status': 'NO_RAD_FILE', 'rolls': len(rolls_after_test)}


def main():
    print("=" * 70)
    print("生成 RAD_v2_fixed (修复换月日 ratio 应用时机)")
    print("=" * 70)
    
    asc_files = list(TEMP_DIR.glob('*_CLC.ASC'))
    print(f"\n找到 {len(asc_files)} 个 ASC 文件")
    
    results = []
    for asc_file in asc_files:
        symbol = asc_file.stem.replace('_CLC', '')
        result = generate_rad_v2_fixed(symbol)
        if result:
            results.append({'Symbol': symbol, **result})
            if result['status'] == 'OK':
                print(f"  {symbol}: {result['rolls']} rolls, corr={result['corr']:.6f} ✓")
            else:
                print(f"  {symbol}: {result['status']} ✗")
    
    # 保存
    results_df = pd.DataFrame(results)
    results_df.to_csv(DATA_DIR / 'rad_v2_fixed_summary.csv', index=False)
    print(f"\n已保存到：{DATA_DIR / 'rad_v2_fixed_summary.csv'}")
    
    # 统计
    ok_df = results_df[results_df['status'] == 'OK']
    print(f"\n=== 统计 ===")
    print(f"成功生成：{len(ok_df)}/{len(results_df)}")
    
    if len(ok_df) > 0:
        valid_corr = ok_df[ok_df['corr'].notna()]
        if len(valid_corr) > 0:
            print(f"\n相关性 ≥0.9999: {len(valid_corr[valid_corr['corr']>=0.9999])} ({len(valid_corr[valid_corr['corr']>=0.9999])/len(valid_corr)*100:.1f}%)")
            print(f"相关性 ≥0.999: {len(valid_corr[valid_corr['corr']>=0.999])} ({len(valid_corr[valid_corr['corr']>=0.999])/len(valid_corr)*100:.1f}%)")
            print(f"相关性中位数：{valid_corr['corr'].median():.6f}")


if __name__ == '__main__':
    main()
