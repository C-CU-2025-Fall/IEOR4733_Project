"""
生成全部 50 个合约的 RAD_v2，使用所有可用的 rollover dates

关键修改：使用全部历史换月累积 ratio，不再只看测试期内换月。
即使换月发生在 2011 年之前，ratio 仍会影响测试期内的价格水平。
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 项目根目录（相对于当前脚本）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / 'config'
DATA_DIR = PROJECT_ROOT / 'data' / 'CLC'
TEMP_DIR = CONFIG_DIR / 'TEMP'
OUTPUT_DIR = DATA_DIR

TEST_START = pd.Timestamp('2011-01-01')
TEST_END = pd.Timestamp('2019-12-31')

# 论文 50 合约
PAPER_50 = [
    # Commodity (25)
    'CC', 'DA', 'GI', 'JO', 'KC', 'KW', 'LB', 'NR', 'SB',
    'ZA', 'ZC', 'ZF', 'ZG', 'ZH', 'ZI', 'ZK', 'ZL',
    'ZN', 'ZO', 'ZP', 'ZR', 'ZT', 'ZU', 'ZW', 'ZZ',
    # Equity Index (11)
    'CA', 'EN', 'ER', 'ES', 'LX', 'MD', 'SC', 'SP', 'XU', 'XX', 'YM',
    # Fixed Income (5)
    'DT', 'FB', 'TY', 'UB', 'US',
    # Forex (9)
    'AN', 'BN', 'CN', 'DX', 'FN', 'JN', 'MP', 'NK', 'SN',
]


def generate_rad_v2(symbol):
    """使用全部历史换月数据生成 RAD_v2"""
    
    asc_file = TEMP_DIR / f'{symbol}_CLC.ASC'
    non_file = DATA_DIR / f'{symbol}_NON.CSV'
    rollover_file = DATA_DIR / f'{symbol}_rollovers.csv'
    rad_file = DATA_DIR / f'{symbol}_RAD.CSV'
    output_file = DATA_DIR / f'{symbol}_RAD_v2_fixed.CSV'
    
    # 检查必要文件
    missing = []
    if not non_file.exists():
        missing.append('NON')
    if not asc_file.exists():
        missing.append('ASC')
    if not rollover_file.exists():
        missing.append('Rollover')
    if missing:
        return {'status': f'MISSING: {",".join(missing)}'}
    
    # 读取 NON 数据
    non = pd.read_csv(non_file, names=['Date', 'O', 'H', 'L', 'C', 'V', 'OI'])
    non['Date'] = pd.to_datetime(non['Date'])
    non = non.sort_values('Date').reset_index(drop=True)
    
    if len(non) == 0:
        return {'status': 'EMPTY_NON'}
    
    # 读取换月数据
    rollover_df = pd.read_csv(rollover_file)
    rollover_df['RollDate'] = pd.to_datetime(rollover_df['RollDate'])
    
    if len(rollover_df) == 0:
        return {'status': 'NO_ROLLOVERS'}
    
    # 匹配换月日期到 NON 数据的索引
    # 允许 ±3 天误差匹配
    rolls = []
    for _, row in rollover_df.iterrows():
        roll_date = row['RollDate']
        
        # 精确匹配
        idx = non[non['Date'] == roll_date].index
        if len(idx) > 0:
            rolls.append({
                'idx': idx[0],
                'date': roll_date,
                'c': row['PrevClose_c'],
                'C': row['NewClose_C'],
            })
        else:
            # ±3 天模糊匹配
            diffs = (non['Date'] - roll_date).abs()
            if diffs.min().days <= 3:
                best_idx = diffs.idxmin()
                rolls.append({
                    'idx': best_idx,
                    'date': roll_date,
                    'c': row['PrevClose_c'],
                    'C': row['NewClose_C'],
                })
    
    if len(rolls) == 0:
        return {'status': 'NO_ROLLS_MATCHED', 'total_rolls': len(rollover_df)}
    
    # 按 idx 排序
    rolls.sort(key=lambda x: x['idx'])
    
    # === 核心：使用全部历史换月累积 ratio ===
    # 从最早的数据开始，每次换月累积一个 ratio
    # ratio = cumprod(old_close / new_close)
    # 换月日当天仍用旧 ratio，次日开始用新 ratio
    
    non['ratio'] = 1.0
    current_ratio = 1.0
    
    for i, roll in enumerate(rolls):
        idx = roll['idx']
        c, C = roll['c'], roll['C']
        
        if C > 0 and c > 0:
            current_ratio *= (c / C)
            # 下一个换月的 idx 或数据末尾
            next_idx = rolls[i + 1]['idx'] if i + 1 < len(rolls) else len(non)
            # 换月日 +1 开始应用新 ratio
            apply_start = idx + 1
            if apply_start < len(non):
                non.loc[apply_start:next_idx - 1 if next_idx <= len(non) else len(non) - 1, 'ratio'] = current_ratio
    
    # 生成 RAD 价格
    non['RAD_Open'] = non['O'] * non['ratio']
    non['RAD_High'] = non['H'] * non['ratio']
    non['RAD_Low'] = non['L'] * non['ratio']
    non['RAD_Close'] = non['C'] * non['ratio']
    non['RAD_Volume'] = non['V']
    non['RAD_OI'] = non['OI']
    
    # 保存全量数据（不只是测试期）
    output_cols = ['Date', 'RAD_Open', 'RAD_High', 'RAD_Low', 'RAD_Close', 'RAD_Volume', 'RAD_OI']
    non[output_cols].to_csv(output_file, index=False, header=False)
    
    # === 与 CLC RAD 对比（测试期）===
    test_mask = (non['Date'] >= TEST_START) & (non['Date'] <= TEST_END)
    test_data = non[test_mask].copy().reset_index(drop=True)
    
    if len(test_data) == 0:
        return {'status': 'NO_TEST_DATA', 'rolls_total': len(rolls)}
    
    # 读取 CLC RAD
    if not rad_file.exists():
        return {
            'status': 'OK_NO_RAD_COMPARE',
            'rolls_total': len(rolls),
            'rolls_in_test': len([r for r in rolls if TEST_START <= non.loc[r['idx'], 'Date'] <= TEST_END]),
            'test_rows': len(test_data),
            'first_date': test_data['Date'].iloc[0].strftime('%Y-%m-%d'),
            'last_date': test_data['Date'].iloc[-1].strftime('%Y-%m-%d'),
        }
    
    rad = pd.read_csv(rad_file, names=['Date', 'O', 'H', 'L', 'C', 'V', 'OI'])
    rad['Date'] = pd.to_datetime(rad['Date'])
    rad_test = rad[(rad['Date'] >= TEST_START) & (rad['Date'] <= TEST_END)].copy().reset_index(drop=True)
    
    if len(rad_test) == 0:
        return {
            'status': 'OK_NO_RAD_TEST_DATA',
            'rolls_total': len(rolls),
            'test_rows': len(test_data),
        }
    
    # 对齐并比较
    min_len = min(len(rad_test), len(test_data))
    if min_len == 0:
        return {
            'status': 'OK_NO_OVERLAP',
            'rolls_total': len(rolls),
            'rad_rows': len(rad_test),
            'v2_rows': len(test_data),
        }
    
    rad_prices = rad_test['C'].values[:min_len].astype(float)
    v2_prices = test_data['RAD_Close'].values[:min_len].astype(float)
    
    # 过滤无效值
    valid = np.isfinite(rad_prices) & np.isfinite(v2_prices) & (rad_prices > 0) & (v2_prices > 0)
    valid_count = valid.sum()
    
    if valid_count < 10:
        return {
            'status': 'OK_INSUFFICIENT_VALID',
            'rolls_total': len(rolls),
            'valid_points': valid_count,
        }
    
    rad_v = rad_prices[valid]
    v2_v = v2_prices[valid]
    
    # 缩放对齐第一天
    scale = rad_v[0] / v2_v[0]
    v2_scaled = v2_v * scale
    
    # 相关性
    corr = np.corrcoef(rad_v, v2_scaled)[0, 1]
    
    # 收益率相关性
    rad_ret = np.diff(rad_v) / rad_v[:-1]
    v2_ret = np.diff(v2_scaled) / v2_scaled[:-1]
    valid_ret = np.isfinite(rad_ret) & np.isfinite(v2_ret)
    ret_corr = np.corrcoef(rad_ret[valid_ret], v2_ret[valid_ret])[0, 1] if valid_ret.sum() > 10 else np.nan
    
    # 差异
    diff_pct = (v2_scaled - rad_v) / rad_v * 100
    first_diff = diff_pct[0]
    last_diff = diff_pct[-1]
    max_diff = np.max(np.abs(diff_pct))
    avg_diff = np.mean(np.abs(diff_pct))
    
    rolls_in_test = len([r for r in rolls if TEST_START <= non.loc[r['idx'], 'Date'] <= TEST_END])
    
    return {
        'status': 'OK',
        'rolls_total': len(rolls),
        'rolls_in_test': rolls_in_test,
        'test_rows': min_len,
        'first_date': test_data['Date'].iloc[0].strftime('%Y-%m-%d'),
        'last_date': test_data['Date'].iloc[-1].strftime('%Y-%m-%d'),
        'scale': scale,
        'corr': corr,
        'ret_corr': ret_corr,
        'first_diff%': first_diff,
        'last_diff%': last_diff,
        'max_diff%': max_diff,
        'avg_diff%': avg_diff,
        'rad_first': rad_v[0],
        'rad_last': rad_v[-1],
        'v2_first': v2_v[0],
        'v2_last': v2_v[-1],
    }


def main():
    print("=" * 80)
    print("生成全部 50 个合约的 RAD_v2_fixed（使用全部历史换月）")
    print("=" * 80)
    
    results = []
    failed = []
    
    for sym in PAPER_50:
        result = generate_rad_v2(sym)
        result['Symbol'] = sym
        results.append(result)
        
        status = result['status']
        if status == 'OK':
            print(f"  ✅ {sym:<4} corr={result['corr']:.6f}  last_diff={result['last_diff%']:+.4f}%  "
                  f"rolls={result['rolls_total']}(test:{result['rolls_in_test']})  "
                  f"scale={result['scale']:.6f}")
        else:
            print(f"  ❌ {sym:<4} {status}")
            failed.append(sym)
    
    # 保存结果
    df = pd.DataFrame(results)
    # 调整列顺序
    cols = ['Symbol', 'status', 'corr', 'ret_corr', 'last_diff%', 'max_diff%', 'avg_diff%',
            'rolls_total', 'rolls_in_test', 'test_rows', 'scale', 'first_date', 'last_date']
    cols = [c for c in cols if c in df.columns]
    df = df[[c for c in cols] + [c for c in df.columns if c not in cols]]
    df.to_csv(DATA_DIR / 'rad_v2_all_50_comparison.csv', index=False)
    
    # 统计
    print(f"\n{'=' * 80}")
    print("统计汇总")
    print(f"{'=' * 80}")
    
    ok = df[df['status'] == 'OK']
    not_ok = df[df['status'] != 'OK']
    
    print(f"\n成功生成并可对比：{len(ok)} / {len(PAPER_50)}")
    print(f"无法生成或对比：{len(not_ok)}")
    
    if len(not_ok) > 0:
        print(f"\n失败合约：")
        for _, row in not_ok.iterrows():
            print(f"  {row['Symbol']}: {row['status']}")
    
    if len(ok) > 0:
        valid_corr = ok[ok['corr'].notna()]
        if len(valid_corr) > 0:
            print(f"\n=== 相关性分布 ===")
            print(f"  ≥0.9999: {len(valid_corr[valid_corr['corr'] >= 0.9999])}")
            print(f"  ≥0.999:  {len(valid_corr[valid_corr['corr'] >= 0.999])}")
            print(f"  ≥0.99:   {len(valid_corr[valid_corr['corr'] >= 0.99])}")
            print(f"  ≥0.95:   {len(valid_corr[valid_corr['corr'] >= 0.95])}")
            print(f"  <0.95:   {len(valid_corr[valid_corr['corr'] < 0.95])}")
            print(f"  中位数:   {valid_corr['corr'].median():.6f}")
            
            print(f"\n=== 最后一天差异分布 ===")
            vd = valid_corr['last_diff%'].abs()
            print(f"  <0.01%:  {len(vd[vd < 0.01])}")
            print(f"  <0.1%:   {len(vd[vd < 0.1])}")
            print(f"  <1%:     {len(vd[vd < 1])}")
            print(f"  <5%:     {len(vd[vd < 5])}")
            print(f"  ≥5%:     {len(vd[vd >= 5])}")
            print(f"  中位数:   {vd.median():.4f}%")


if __name__ == '__main__':
    main()
