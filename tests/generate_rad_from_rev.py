"""
用 REV + NON 为 ZN 和 US 生成确定性 RAD

方法：
  1. 从 REV 检测所有 roll events（adj_change ≠ 0，确定性）
  2. 计算 cumulative ratio = Π(prev_close / new_close)
  3. RAD = NON × cumulative ratio

Usage:
  cd IEOR4733_Project && PYTHONPATH=. python3 tests/generate_rad_from_rev.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'CLC'
sys.path.insert(0, str(PROJECT_ROOT))

TEST_START = pd.Timestamp('2011-01-01')
TEST_END = pd.Timestamp('2019-12-31')


def load(filepath):
    df = pd.read_csv(filepath, header=None,
                     names=['Date', 'O', 'H', 'L', 'C', 'V', 'OI'])
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df = df[df['C'].notna()].sort_values('Date').reset_index(drop=True)
    return df


def detect_rev_rolls(non_df, rev_df):
    merged = non_df[['Date', 'C']].merge(
        rev_df[['Date', 'C']], on='Date', suffixes=('_non', '_rev'))
    merged = merged.sort_values('Date').reset_index(drop=True)
    merged['adj'] = merged['C_rev'] - merged['C_non']
    merged['adj_change'] = merged['adj'] - merged['adj'].shift(1)

    rolls = []
    for i in range(1, len(merged)):
        ac = merged.loc[i, 'adj_change']
        if abs(ac) < 1e-10:
            continue
        roll_date = merged.loc[i - 1, 'Date']
        prev_close = merged.loc[i - 1, 'C_non']
        new_close = prev_close - ac
        rolls.append({'idx': i - 1, 'date': roll_date, 'prev': prev_close,
                      'new': new_close, 'ratio': prev_close / new_close if new_close != 0 else 1.0})
    return rolls, merged


def generate_rad(symbol):
    non_f = DATA_DIR / f'{symbol}_NON.CSV'
    rev_f = DATA_DIR / f'{symbol}_REV.CSV'
    output_f = DATA_DIR / f'{symbol}_RAD_v2.CSV'

    non_df = load(non_f)
    rev_df = load(rev_f)

    rolls, merged = detect_rev_rolls(non_df, rev_df)

    # 只用测试期内的 rolls（但输出从 NON 最早日期开始，保证 warmup）
    test_rolls = [r for r in rolls if r['date'] >= pd.Timestamp('2009-01-01')]
    print(f'\n{symbol}: {len(rolls)} total rolls, {len(test_rolls)} from 2009+')

    # 累积 ratio：从 2009 年第一个 roll 开始
    # 2009 年之前的 cum_ratio = 1.0（不调整）
    rad_c = merged['C_non'].values.astype(float).copy()
    cum_ratio = 1.0

    # 按 idx 排序
    test_rolls.sort(key=lambda r: r['idx'])

    roll_idx = 0
    for t in range(len(rad_c)):
        while roll_idx < len(test_rolls) and test_rolls[roll_idx]['idx'] <= t:
            cum_ratio *= test_rolls[roll_idx]['ratio']
            roll_idx += 1
        rad_c[t] *= cum_ratio

    # 构建 RAD DataFrame
    rad_df = merged.copy()
    rad_df['RAD_C'] = rad_c

    # 输出从 2009 开始（warmup 需要）
    out_mask = rad_df['Date'] >= pd.Timestamp('2009-01-01')
    out_df = rad_df[out_mask].copy()

    out = out_df[['Date', 'C_non', 'C_non', 'C_non', 'RAD_C', 'C_rev', 'C_non']].copy()
    out.columns = ['Date', 'O', 'H', 'L', 'C', 'V', 'OI']
    out['Date'] = out['Date'].dt.strftime('%m/%d/%Y')
    out.to_csv(output_f, index=False, header=False)

    # 验证（测试期 2011-2019）
    test_mask = (rad_df['Date'] >= TEST_START) & (rad_df['Date'] <= TEST_END)
    test_rad = rad_df.loc[test_mask, 'RAD_C'].values
    test_non = rad_df.loc[test_mask, 'C_non'].values
    test_rev = rad_df.loc[test_mask, 'C_rev'].values

    if len(test_rad) > 0:
        corr_non = np.corrcoef(test_rad, test_non)[0, 1]
        corr_rev = np.corrcoef(test_rad, test_rev)[0, 1]
        print(f'  测试期: {len(test_rad)} 行')
        print(f'  RAD 价格范围: {test_rad.min():.4f} - {test_rad.max():.4f}')
        print(f'  NON 价格范围: {test_non.min():.4f} - {test_non.max():.4f}')
        print(f'  REV 价格范围: {test_rev.min():.4f} - {test_rev.max():.4f}')
        print(f'  RAD vs NON corr: {corr_non:.6f}')
        print(f'  RAD vs REV corr: {corr_rev:.6f}')
        print(f'  → 已保存到 {output_f}')
        return {'symbol': symbol, 'rolls': len(test_rolls), 'rows': len(test_rad),
                'corr_non': corr_non, 'corr_rev': corr_rev}

    return {'symbol': symbol, 'rolls': len(test_rolls), 'rows': 0}


def main():
    print('=' * 80)
    print('用 REV + NON 生成确定性 RAD (ZN, US)')
    print('=' * 80)

    for sym in ['ZN', 'US']:
        r = generate_rad(sym)
        print(f'  结果: {r}')

    # 更新 data_loader.py
    print('\n更新 data_loader.py...')
    loader_path = PROJECT_ROOT / 'data_loader.py'
    with open(loader_path) as f:
        content = f.read()

    # 检查当前硬编码
    if "if ticker in ['ZH', 'ZN', 'ZU', 'US']:" in content:
        print('  ZN/US 已在硬编码列表中，将使用 RAD_v2')
    else:
        print('  ⚠️ ZN/US 不在硬编码列表中！')

    print('\n✅ 完成')


if __name__ == '__main__':
    main()
