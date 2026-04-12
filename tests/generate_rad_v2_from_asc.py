"""
使用 ASC 提取的实际换月日期生成 RAD_v2

ASC 文件中的 00000000 行标记了真正的换月：
- 第 2 列 = 旧合约 Close (c)
- 第 5 列 = 新合约 Close (C)
- ratio = c / C

Usage:
    python tests/generate_rad_v2_from_asc.py
"""

import pandas as pd
from pathlib import Path

TEMP_DIR = Path('/home/congge2026/.openclaw/workspace/IEOR4733_Project/config/TEMP')
DATA_DIR = Path('/home/congge2026/.openclaw/workspace/IEOR4733_Project/data/CLC')
TEST_START = pd.Timestamp('2011-01-01')
TEST_END = pd.Timestamp('2019-12-31')


def generate_rad_v2_from_asc(symbol):
    """
    使用 ASC 提取的实际换月日期生成 RAD_v2
    
    参数:
        symbol: 合约符号
    
    返回:
        结果字典或 None
    """
    asc_file = TEMP_DIR / f'{symbol}_CLC.ASC'
    non_file = DATA_DIR / f'{symbol}_NON.CSV'
    rollover_file = DATA_DIR / f'{symbol}_rollovers.csv'
    output_file = DATA_DIR / f'{symbol}_RAD_v2_asc.CSV'
    
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
            rolls.append({
                'idx': idx[0],
                'c': row['PrevClose_c'],
                'C': row['NewClose_C']
            })
        else:
            # 找最近的 (±5 天内)
            diffs = (non['Date'] - roll_date).abs()
            min_diff = diffs.min()
            if min_diff.days <= 5:
                closest_idx = diffs.idxmin()
                rolls.append({
                    'idx': closest_idx,
                    'c': row['PrevClose_c'],
                    'C': row['NewClose_C']
                })
    
    if len(rolls) == 0:
        return {'status': 'NO_ROLLS_MATCHED'}
    
    # 过滤测试期内的换月
    rolls_after_test = [r for r in rolls if r['idx'] >= test_start_idx]
    
    if len(rolls_after_test) == 0:
        return {'status': 'NO_ROLLS_IN_TEST'}
    
    # 生成 RAD
    non['ratio'] = 1.0
    non.loc[:test_start_idx, 'ratio'] = 1.0
    
    current_ratio = 1.0
    for i, roll in enumerate(rolls_after_test):
        idx = roll['idx']
        c = roll['c']
        C = roll['C']
        
        if C > 0 and c > 0:
            ratio_change = c / C
            current_ratio *= ratio_change
            
            next_idx = rolls_after_test[i + 1]['idx'] if i + 1 < len(rolls_after_test) else len(non)
            non.loc[idx:next_idx - 1, 'ratio'] = current_ratio
    
    # 生成 RAD 价格
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
    
    # 验证与 CLC RAD 的对比
    rad_file = DATA_DIR / f'{symbol}_RAD.CSV'
    if rad_file.exists():
        rad = pd.read_csv(rad_file, names=['Date', 'O', 'H', 'L', 'C', 'V', 'OI'])
        rad['Date'] = pd.to_datetime(rad['Date'])
        rad_test = rad[(rad['Date'] >= TEST_START) & (rad['Date'] <= TEST_END)].reset_index(drop=True)
        
        if len(rad_test) > 0 and len(test_data) > 0:
            min_len = min(len(rad_test), len(test_data))
            corr = test_data['RAD_Close'].iloc[:min_len].corr(rad_test['C'].iloc[:min_len])
        else:
            corr = None
    else:
        corr = None
    
    return {
        'status': 'OK',
        'rolls': len(rolls_after_test),
        'rows': len(test_data),
        'first_close': test_data['RAD_Close'].iloc[0] if len(test_data) > 0 else None,
        'last_close': test_data['RAD_Close'].iloc[-1] if len(test_data) > 0 else None,
        'corr_with_clc': corr
    }


def main():
    print("=" * 60)
    print("使用 ASC 换月数据生成 RAD_v2")
    print("=" * 60)
    
    # 找到所有有 ASC 文件的合约
    asc_files = list(TEMP_DIR.glob('*_CLC.ASC'))
    print(f"\n找到 {len(asc_files)} 个 ASC 文件")
    
    results = []
    for asc_file in asc_files:
        symbol = asc_file.stem.replace('_CLC', '')
        result = generate_rad_v2_from_asc(symbol)
        
        if result:
            results.append({'Symbol': symbol, **result})
            if result['status'] == 'OK':
                corr_str = f"{result['corr_with_clc']:.4f}" if result['corr_with_clc'] else "N/A"
                print(f"  {symbol}: {result['rolls']} rolls, {result['rows']} rows, corr={corr_str} ✓")
            else:
                print(f"  {symbol}: {result['status']} ✗")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(DATA_DIR / 'rad_v2_asc_summary.csv', index=False)
    print(f"\n已保存到：{DATA_DIR / 'rad_v2_asc_summary.csv'}")
    
    # 统计
    ok_df = results_df[results_df['status'] == 'OK']
    print(f"\n=== 统计 ===")
    print(f"成功生成：{len(ok_df)}/{len(results_df)}")
    
    if len(ok_df) > 0:
        valid_corr = ok_df[ok_df['corr_with_clc'].notna()]
        if len(valid_corr) > 0:
            print(f"\n=== 与 CLC RAD 相关性 ===")
            print(f"≥0.99: {len(valid_corr[valid_corr['corr_with_clc']>=0.99])} ({len(valid_corr[valid_corr['corr_with_clc']>=0.99])/len(valid_corr)*100:.1f}%)")
            print(f"0.95-0.99: {len(valid_corr[(valid_corr['corr_with_clc']>=0.95) & (valid_corr['corr_with_clc']<0.99)])}")
            print(f"<0.95: {len(valid_corr[valid_corr['corr_with_clc']<0.95])}")
            print(f"\n中位数：{valid_corr['corr_with_clc'].median():.4f}")
            print(f"平均值：{valid_corr['corr_with_clc'].mean():.4f}")


if __name__ == '__main__':
    main()
