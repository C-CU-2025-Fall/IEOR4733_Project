"""
验证 D 类合约的换月规则并用 NON 生成 RAD_v2

方法：
1. 用 2011 年前的 ASC+rollover 验证换月规则代码
2. 验证通过后，用测试期 NON + 规则生成 RAD_v2
3. 与 vendor RAD 对比验证

D 类合约 (22 个)：NR, SB, SN, SP, TY, US, ZA, ZC, ZF, ZG, ZH, ZI, ZK, ZL, ZN, ZO, ZP, ZR, ZT, ZU, ZW, ZZ
"""

import pandas as pd
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'CLC'
CONFIG_DIR = PROJECT_ROOT / 'config'
TEMP_DIR = CONFIG_DIR / 'TEMP'

TEST_START = pd.Timestamp('2011-01-01')
TEST_END = pd.Timestamp('2019-12-31')

# D 类合约
D_CONTRACTS = ['NR', 'SB', 'SN', 'SP', 'TY', 'US', 'ZA', 'ZC', 'ZF', 'ZG', 'ZH', 'ZI', 'ZK', 'ZL', 'ZN', 'ZO', 'ZP', 'ZR', 'ZT', 'ZU', 'ZW', 'ZZ']

# 加载换月规则
with open(CONFIG_DIR / 'roll_rules_corrected.json') as f:
    roll_rules = json.load(f)


def load_asc_prices(symbol):
    """读取 ASC 文件价格"""
    asc_file = TEMP_DIR / f'{symbol}_CLC.ASC'
    if not asc_file.exists():
        return None
    
    rows = []
    with open(asc_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and len(parts[0]) == 8 and parts[0].isdigit():
                year = int(parts[0][:4])
                if 1900 <= year <= 2099:
                    rows.append(parts[:8])
    
    if not rows:
        return None
    
    df = pd.DataFrame(rows, columns=['Date','O','H','L','C','V','OI','_'])
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    df['C'] = pd.to_numeric(df['C'])
    return df


def load_rollover(symbol):
    """读取 rollover 文件"""
    rollover_file = DATA_DIR / f'{symbol}_rollovers.csv'
    if not rollover_file.exists():
        return None
    
    df = pd.read_csv(rollover_file)
    df['RollDate'] = pd.to_datetime(df['RollDate'])
    return df


def load_non(symbol):
    """读取 NON 文件"""
    non_file = DATA_DIR / f'{symbol}_NON.CSV'
    if not non_file.exists():
        return None
    
    df = pd.read_csv(non_file, names=['Date','O','H','L','C','V','OI'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['C'] = pd.to_numeric(df['C'])
    return df


def load_rad(symbol):
    """读取 RAD 文件"""
    rad_file = DATA_DIR / f'{symbol}_RAD.CSV'
    if not rad_file.exists():
        return None
    
    df = pd.read_csv(rad_file, names=['Date','O','H','L','C','V','OI'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['C'] = pd.to_numeric(df['C'])
    return df


def generate_rad_v2(non_df, rollover_df, test_start):
    """用 NON + rollover 生成 RAD_v2"""
    non = non_df.copy().sort_values('Date').reset_index(drop=True)
    
    # 找到测试期开始索引
    test_start_mask = non['Date'] >= test_start
    if not test_start_mask.any():
        return None
    test_start_idx = test_start_mask.idxmax()
    
    # 匹配换月日期
    rolls = []
    for _, row in rollover_df.iterrows():
        roll_date = row['RollDate']
        idx = non[non['Date'] == roll_date].index
        
        if len(idx) == 0:
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
        else:
            rolls.append({
                'idx': idx[0],
                'c': row['PrevClose_c'],
                'C': row['NewClose_C']
            })
    
    if len(rolls) == 0:
        return None
    
    # 过滤测试期内的换月
    rolls_after_test = [r for r in rolls if r['idx'] >= test_start_idx]
    
    # 生成 back-adjusted ratio
    non['ratio'] = 1.0
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
    
    # 生成 RAD_v2
    non['RAD_Close'] = non['C'] * non['ratio']
    return non


def verify_pre2011(symbol):
    """用 2011 年前数据验证换月方法"""
    asc = load_asc_prices(symbol)
    rollover = load_rollover(symbol)
    non = load_non(symbol)
    rad = load_rad(symbol)
    
    if asc is None or rollover is None or non is None or rad is None:
        return {'status': 'MISSING_DATA'}
    
    # 过滤 2011 年前数据
    pre2011_mask = asc['Date'] < TEST_START
    asc_pre = asc[pre2011_mask].reset_index(drop=True)
    
    if len(asc_pre) < 100:
        return {'status': 'INSUFFICIENT_PRE2011_DATA'}
    
    # 用 ASC 生成 RAD_v2（2011 年前）
    rolls = []
    for _, row in rollover.iterrows():
        roll_date = row['RollDate']
        if roll_date >= TEST_START:
            continue
        
        idx = asc[asc['Date'] == roll_date].index
        if len(idx) > 0:
            rolls.append({'idx': idx[0], 'c': row['PrevClose_c'], 'C': row['NewClose_C']})
    
    if len(rolls) < 10:
        return {'status': 'INSUFFICIENT_ROLLS'}
    
    # 计算 ratio
    asc_pre['ratio'] = 1.0
    current_ratio = 1.0
    
    for i, roll in enumerate(rolls):
        idx = roll['idx']
        c = roll['c']
        C = roll['C']
        
        if C > 0 and c > 0:
            ratio_change = c / C
            current_ratio *= ratio_change
            
            next_idx = rolls[i + 1]['idx'] if i + 1 < len(rolls) else len(asc_pre)
            asc_pre.loc[idx:next_idx - 1, 'ratio'] = current_ratio
    
    asc_pre['RAD_v2'] = asc_pre['C'] * asc_pre['ratio']
    
    # 与 vendor RAD 对比
    rad_pre = rad[rad['Date'] < TEST_START].reset_index(drop=True)
    
    # 合并对比
    merged = pd.merge(asc_pre, rad_pre, on='Date', suffixes=('_v2', '_vendor'))
    
    if len(merged) < 100:
        return {'status': 'INSUFFICIENT_OVERLAP'}
    
    # 计算相关性
    corr = merged['RAD_v2'].corr(merged['C_vendor'])
    
    # 检查第一天价格对齐
    first_v2 = merged['RAD_v2'].iloc[0]
    first_vendor = merged['C_vendor'].iloc[0]
    first_ratio = first_v2 / first_vendor
    
    return {
        'status': 'OK',
        'pre2011_days': len(merged),
        'rolls_used': len(rolls),
        'corr': corr,
        'first_ratio': first_ratio,
    }


def generate_test_rad_v2(symbol):
    """生成测试期 RAD_v2"""
    non = load_non(symbol)
    rollover = load_rollover(symbol)
    rad = load_rad(symbol)
    
    if non is None or rollover is None or rad is None:
        return {'status': 'MISSING_DATA'}
    
    # 生成 RAD_v2
    rad_v2 = generate_rad_v2(non, rollover, TEST_START)
    
    if rad_v2 is None:
        return {'status': 'GENERATION_FAILED'}
    
    # 过滤测试期
    test_mask = (rad_v2['Date'] >= TEST_START) & (rad_v2['Date'] <= TEST_END)
    rad_v2_test = rad_v2[test_mask].reset_index(drop=True)
    
    # 与 vendor RAD 对比
    rad_test = rad[(rad['Date'] >= TEST_START) & (rad['Date'] <= TEST_END)].reset_index(drop=True)
    
    # 合并对比
    merged = pd.merge(rad_v2_test, rad_test, on='Date', suffixes=('_v2', '_vendor'))
    
    if len(merged) < 100:
        return {'status': 'INSUFFICIENT_TEST_DATA'}
    
    # 计算相关性
    corr = merged['RAD_Close'].corr(merged['C_vendor'])
    
    # 检查价格水平
    v2_mean = merged['RAD_Close'].mean()
    vendor_mean = merged['C_vendor'].mean()
    level_ratio = v2_mean / vendor_mean
    
    return {
        'status': 'OK',
        'test_days': len(merged),
        'rolls_in_test': len([r for r in rad_v2_test['ratio'].diff().dropna() if abs(r - 1) > 0.001]),
        'corr': corr,
        'level_ratio': level_ratio,
    }


def main():
    print("="*100)
    print("D 类合约换月规则验证 + RAD_v2 生成")
    print("="*100)
    
    results = []
    
    for symbol in D_CONTRACTS:
        print(f"\n{symbol}:")
        
        # 步骤 1: 用 2011 年前数据验证
        pre_result = verify_pre2011(symbol)
        
        if pre_result['status'] == 'OK':
            print(f"  2011 年前验证：corr={pre_result['corr']:.4f}, 天数={pre_result['pre2011_days']}, 换月={pre_result['rolls_used']}")
        else:
            print(f"  2011 年前验证：{pre_result['status']}")
        
        # 步骤 2: 生成测试期 RAD_v2
        test_result = generate_test_rad_v2(symbol)
        
        if test_result['status'] == 'OK':
            print(f"  测试期 RAD_v2: corr={test_result['corr']:.4f}, 天数={test_result['test_days']}, 换月={test_result['rolls_in_test']}")
        else:
            print(f"  测试期 RAD_v2: {test_result['status']}")
        
        results.append({
            'Symbol': symbol,
            'Pre2011_Status': pre_result['status'],
            'Pre2011_Corr': pre_result.get('corr'),
            'Pre2011_Days': pre_result.get('pre2011_days'),
            'Test_Status': test_result['status'],
            'Test_Corr': test_result.get('corr'),
            'Test_Days': test_result.get('test_days'),
            'Level_Ratio': test_result.get('level_ratio'),
        })
    
    # 保存结果
    df = pd.DataFrame(results)
    output_file = PROJECT_ROOT / 'tests' / 'results' / 'd_contracts_verification.csv'
    df.to_csv(output_file, index=False)
    print(f"\n\n详细结果：{output_file}")
    
    # 统计
    print("\n" + "="*100)
    print("统计摘要")
    print("="*100)
    
    ok_pre = df[df['Pre2011_Status'] == 'OK']
    ok_test = df[df['Test_Status'] == 'OK']
    
    print(f"2011 年前验证成功：{len(ok_pre)}/{len(df)} ({len(ok_pre)/len(df)*100:.1f}%)")
    print(f"测试期生成成功：{len(ok_test)}/{len(df)} ({len(ok_test)/len(df)*100:.1f}%)")
    
    if len(ok_test) > 0 and ok_test['Test_Corr'].notna().any():
        valid_corr = ok_test[ok_test['Test_Corr'].notna()]
        print(f"\n测试期相关性统计 (n={len(valid_corr)}):")
        print(f"  ≥0.99: {len(valid_corr[valid_corr['Test_Corr']>=0.99])} ({len(valid_corr[valid_corr['Test_Corr']>=0.99])/len(valid_corr)*100:.1f}%)")
        print(f"  0.95-0.99: {len(valid_corr[(valid_corr['Test_Corr']>=0.95) & (valid_corr['Test_Corr']<0.99)])}")
        print(f"  <0.95: {len(valid_corr[valid_corr['Test_Corr']<0.95])}")
        print(f"  中位数：{valid_corr['Test_Corr'].median():.4f}")


if __name__ == '__main__':
    main()
