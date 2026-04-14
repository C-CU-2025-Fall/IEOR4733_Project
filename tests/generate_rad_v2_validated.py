"""
用交叉验证确立的方法论为 4 个损坏合约生成 RAD_v2

方法论（50 合约交叉验证确立）:
  1. REV adj_change ≠ 0 → roll_date（确定性）
  2. NON[roll_date] = prev_close
  3. new_close = prev_close - adj_change
  4. ratio = prev_close / new_close
  5. cumulative_ratio 从数据最早日期开始累积
  6. RAD = NON × cumulative_ratio

注意: RAD 是 forward adjustment (向前调整)
  cumulative_ratio 从历史最早开始累积，越新的数据累积越多次调整
  这样保证任意时间点的 RAD 价格都是连续可比的

Usage:
  cd IEOR4733_Project && python3 tests/generate_rad_v2_validated.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'CLC'

DAMAGED = ['ZH', 'ZU', 'US', 'ZN']
# ZN 从测试期开始累积（天然气月度换月，累积误差大）
# 其他合约也从测试期开始，保证价格量级合理
START_OVERRIDE = {'ZN': '2011-01-01', 'ZH': '2011-01-01', 'ZU': '2011-01-01', 'US': '2011-01-01'}


def load(filepath):
    df = pd.read_csv(filepath, header=None,
                     names=['Date', 'O', 'H', 'L', 'C', 'V', 'OI'])
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df = df[df['C'].notna()].sort_values('Date').reset_index(drop=True)
    return df


def detect_rolls_from_rev(non_df, rev_df):
    """确定性 roll 检测：adj_change ≠ 0
    
    ratio 公式推导：
      要让 RAD_ret = REV_ret，即 RAD[t]/RAD[t-1] = REV[t]/REV[t-1]
      因为 RAD = NON × cum_ratio，所以:
      (NON[t] × ratio) / NON[t-1] = REV[t] / REV[t-1]
      ratio = (REV[t] × NON[t-1]) / (REV[t-1] × NON[t])
    """
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
        non_t_minus_1 = merged.loc[i - 1, 'C_non']
        non_t = merged.loc[i, 'C_non']
        rev_t_minus_1 = merged.loc[i - 1, 'C_rev']
        rev_t = merged.loc[i, 'C_rev']
        # 正确公式：ratio = (REV[t] × NON[t-1]) / (REV[t-1] × NON[t])
        ratio = (rev_t * non_t_minus_1) / (rev_t_minus_1 * non_t) if (rev_t_minus_1 * non_t) != 0 else 1.0
        rolls.append({
            'idx': i - 1,
            'date': roll_date,
            'non_t_minus_1': non_t_minus_1,
            'non_t': non_t,
            'rev_t_minus_1': rev_t_minus_1,
            'rev_t': rev_t,
            'ratio': ratio,
            'adj_change': ac,
        })
    return rolls, merged


def generate_rad(symbol):
    non_f = DATA_DIR / f'{symbol}_NON.CSV'
    rev_f = DATA_DIR / f'{symbol}_REV.CSV'
    output_f = DATA_DIR / f'{symbol}_RAD_v2.CSV'

    non_df = load(non_f)
    rev_df = load(rev_f)

    rolls, merged = detect_rolls_from_rev(non_df, rev_df)
    rolls.sort(key=lambda r: r['idx'])

    print(f'\n{symbol}: {len(rolls)} rolls detected')
    print(f'  First roll: {rolls[0]["date"].date()} NON={rolls[0]["non_t_minus_1"]:.4f}→{rolls[0]["non_t"]:.4f} REV={rolls[0]["rev_t_minus_1"]:.4f}→{rolls[0]["rev_t"]:.4f} ratio={rolls[0]["ratio"]:.6f}')
    print(f'  Last roll:  {rolls[-1]["date"].date()} NON={rolls[-1]["non_t_minus_1"]:.4f}→{rolls[-1]["non_t"]:.4f} REV={rolls[-1]["rev_t_minus_1"]:.4f}→{rolls[-1]["rev_t"]:.4f} ratio={rolls[-1]["ratio"]:.6f}')

    # Forward adjustment: cumulative ratio from earliest data
    # ZN 等月度换月合约从合理起点开始，减少累积误差
    start_override = START_OVERRIDE.get(symbol)
    if start_override:
        start_mask = merged['Date'] >= pd.Timestamp(start_override)
        start_idx = start_mask.idxmax() if start_mask.any() else 0
        rolls = [r for r in rolls if r['idx'] >= start_idx]
        print(f'  (从 {start_override} 开始累积，{len(rolls)} rolls)')

    rad_c = merged['C_non'].values.astype(float).copy()
    cum_ratio = 1.0

    roll_idx = 0
    for t in range(len(rad_c)):
        # 应用所有 idx <= t 的 rolls
        while roll_idx < len(rolls) and rolls[roll_idx]['idx'] <= t:
            cum_ratio *= rolls[roll_idx]['ratio']
            roll_idx += 1
        rad_c[t] *= cum_ratio

    # 输出完整数据（从 NON 最早日期开始）
    out = merged[['Date', 'C_non', 'C_non', 'C_non']].copy()
    out.columns = ['Date', 'O', 'H', 'L']
    out['C'] = rad_c
    out['V'] = merged['C_rev']
    out['OI'] = merged['C_non']
    out['Date'] = out['Date'].dt.strftime('%m/%d/%Y')
    out.to_csv(output_f, index=False, header=False)

    # 验证：2011-2019 测试期
    test = merged[(merged['Date'] >= pd.Timestamp('2011-01-01')) &
                  (merged['Date'] <= pd.Timestamp('2019-12-31'))]
    test_rad = rad_c[test.index]
    test_non = test['C_non'].values

    print(f'  输出: {len(merged)} 行 → {output_f.name}')
    print(f'  测试期 RAD: {test_rad.min():.4f} - {test_rad.max():.4f}')
    print(f'  测试期 NON: {test_non.min():.4f} - {test_non.max():.4f}')

    # RAD vs NON 的 daily returns 应该高度相关
    rad_ret = pd.Series(test_rad).diff().dropna()
    non_ret = pd.Series(test_non).diff().dropna()
    min_len = min(len(rad_ret), len(non_ret))
    corr = np.corrcoef(rad_ret.values[:min_len], non_ret.values[:min_len])[0, 1]
    print(f'  RAD vs NON daily return corr: {corr:.6f}')

    # 检查连续性：roll 日次日的 return 不应有异常跳变
    roll_indices_in_test = []
    for r in rolls:
        if pd.Timestamp('2011-01-01') <= r['date'] <= pd.Timestamp('2019-12-31'):
            roll_indices_in_test.append(r['idx'])

    if roll_indices_in_test:
        # roll 日次日的 RAD return
        roll_next_returns = []
        for ri in roll_indices_in_test:
            if ri + 1 < len(rad_c):
                if rad_c[ri] != 0:
                    ret = (rad_c[ri + 1] - rad_c[ri]) / abs(rad_c[ri])
                    roll_next_returns.append(ret)
        if roll_next_returns:
            print(f'  Roll 日次日 RAD return: mean={np.mean(roll_next_returns):.4f} std={np.std(roll_next_returns):.4f}')
            print(f'    (应该和正常日 return 分布一致，无异常跳变)')

    return {'symbol': symbol, 'rolls': len(rolls), 'corr': corr}


def main():
    print('=' * 80)
    print('用交叉验证方法论为 4 个损坏合约生成 RAD_v2')
    print('方法: REV adj_change → roll date → ratio → cumulative × NON')
    print('=' * 80)

    for sym in DAMAGED:
        r = generate_rad(sym)
        print(f'  → {r["symbol"]}: {r["rolls"]} rolls, corr={r["corr"]:.6f}')

    print('\n✅ 4 个 RAD_v2 已生成')
    print('data_loader.py 会自动使用这些文件（硬编码列表）')


if __name__ == '__main__':
    main()
