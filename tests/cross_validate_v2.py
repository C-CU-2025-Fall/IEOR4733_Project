"""
CLC 数据交叉验证 v2

逻辑链:
1. ASC (测试期外) vs 理论规则 → 验证理论规则可靠性
2. 理论规则 (测试期内) → 推导 Roll Dates
3. 理论 Roll Dates vs RAD 跳变 → 验证 RAD 可靠性

这样即使 ASC 不覆盖测试期，也能通过理论规则验证 RAD
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import calendar

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'CLC'
CONFIG_DIR = PROJECT_ROOT / 'config'
TEMP_DIR = CONFIG_DIR / 'TEMP'
RESULTS_DIR = PROJECT_ROOT / 'tests' / 'results'

# 50 个合约
ALL_CONTRACTS = ['NR', 'SB', 'SN', 'SP', 'TY', 'US', 'ZA', 'ZC', 'ZF', 'ZG', 'ZH', 'ZI', 'ZK', 'ZL', 'ZN', 'ZO', 'ZP', 'ZR', 'ZT', 'ZU', 'ZW', 'ZZ',
                 'CC', 'DA', 'GI', 'JO', 'KC', 'KW', 'LB', 'CA', 'EN', 'ER', 'ES', 'LX', 'MD', 'SC', 'XU', 'XX', 'YM',
                 'DT', 'FB', 'UB', 'AN', 'BN', 'CN', 'DX', 'FN', 'JN', 'MP', 'NK']

# 加载 roll rules
with open(CONFIG_DIR / 'roll_rules_corrected.json') as f:
    ROLL_RULES = json.load(f)

# 加载合约月份
with open(CONFIG_DIR / 'contract_months.json') as f:
    CONTRACT_MONTHS = json.load(f)

months_map = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
              'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}


def load_non(symbol):
    """加载 NON 数据"""
    fpath = DATA_DIR / f'{symbol}_NON.CSV'
    if not fpath.exists():
        return None
    
    df = pd.read_csv(fpath, names=['Date','O','H','L','C','V','OI'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['C'] = pd.to_numeric(df['C'])
    return df.sort_values('Date').reset_index(drop=True)


def load_rad(symbol):
    """加载 RAD 数据"""
    if symbol in ['ZH', 'ZN', 'ZU', 'US']:
        fpath = DATA_DIR / f'{symbol}_RAD_v2.CSV'
    else:
        fpath = DATA_DIR / f'{symbol}_RAD.CSV'
    
    if not fpath.exists():
        return None
    
    df = pd.read_csv(fpath, names=['Date','O','H','L','C','V','OI'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['C'] = pd.to_numeric(df['C'])
    return df.sort_values('Date').reset_index(drop=True)


def load_asc(symbol):
    """加载 ASC 数据 (如果存在)"""
    fpath = TEMP_DIR / f'{symbol}_CLC.ASC'
    if not fpath.exists():
        return None
    
    try:
        df = pd.read_csv(fpath, sep=r'\s+', header=None, engine='python',
                         names=['Date','O','H','L','C','V','OI','Adj'],
                         on_bad_lines='skip')
        df['Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m%d', errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Date'] = df['Date'].dt.normalize()
        df['Adj'] = pd.to_numeric(df['Adj'], errors='coerce')
        return df.sort_values('Date').reset_index(drop=True)
    except Exception as e:
        print(f"  ⚠️  {symbol} ASC 加载失败：{e}")
        return None


def get_theoretical_roll_dates(symbol, start_year=1970, end_year=2020):
    """根据理论规则计算 roll dates"""
    rule_name = None
    for rname, rdata in ROLL_RULES.items():
        if symbol in rdata.get('symbols', []):
            rule_name = rname
            break
    
    if rule_name is None:
        return []
    
    parts = rule_name.split('_')
    rule_type = None
    day = None
    
    if parts[0] == 'MPDM':
        rule_type = 'MPDM'
        day = int(parts[1])
    elif parts[0] == 'DM':
        rule_type = 'DM'
        day = int(parts[1])
    else:
        return []
    
    months_str = CONTRACT_MONTHS.get(symbol, 'H,M,U,Z')
    delivery_months = [months_map[m.strip()] for m in months_str.split(',') 
                       if m.strip() in months_map]
    
    roll_dates = []
    for year in range(start_year, end_year + 1):
        for dm in delivery_months:
            if rule_type == 'MPDM':
                if dm == 1:
                    pm_year = year - 1
                    pm_month = 12
                else:
                    pm_year = year
                    pm_month = dm - 1
                
                max_day = calendar.monthrange(pm_year, pm_month)[1]
                actual_day = min(day, max_day)
                
                try:
                    rd = pd.Timestamp(year=pm_year, month=pm_month, day=actual_day)
                    roll_dates.append(rd)
                except:
                    pass
            elif rule_type == 'DM':
                max_day = calendar.monthrange(year, dm)[1]
                actual_day = min(day, max_day)
                
                try:
                    rd = pd.Timestamp(year=year, month=dm, day=actual_day)
                    roll_dates.append(rd)
                except:
                    pass
    
    return sorted(roll_dates)


def detect_asc_roll_dates(asc_df, threshold=100):
    """从 ASC 检测 Roll Dates (Adj 跳变点)"""
    if asc_df is None or len(asc_df) == 0:
        return []
    
    asc_df = asc_df.copy()
    asc_df['Adj_change'] = asc_df['Adj'].diff().abs()
    rolls = asc_df[asc_df['Adj_change'] > threshold]
    return rolls['Date'].tolist()


def detect_rad_roll_dates(rad_df, non_df, threshold=0.01):
    """从 RAD 检测 Roll Dates (ratio 跳变点)"""
    if rad_df is None or non_df is None:
        return []
    
    merged = pd.merge(rad_df, non_df, on='Date', suffixes=('_rad', '_non'))
    merged['ratio'] = merged['C_rad'] / merged['C_non']
    merged['ratio_change'] = merged['ratio'].diff().abs()
    
    jumps = merged[merged['ratio_change'] > threshold]
    return jumps['Date'].tolist()


def match_dates(dates_a, dates_b, max_diff_days=3):
    """匹配两个日期列表，返回匹配率和平均差异"""
    if len(dates_a) == 0 or len(dates_b) == 0:
        return 0, np.nan, 0, 0
    
    matches = 0
    diffs = []
    
    for da in dates_a:
        for db in dates_b:
            diff = abs((da - db).days)
            if diff <= max_diff_days:
                matches += 1
                diffs.append(diff)
                break
    
    match_rate = matches / max(len(dates_a), len(dates_b))
    avg_diff = np.mean(diffs) if diffs else np.nan
    
    return match_rate, avg_diff, matches, max(len(dates_a), len(dates_b))


def validate_symbol(symbol):
    """验证单个合约"""
    result = {
        'Symbol': symbol,
        'Has_ASC': False,
        'Has_RAD': False,
        'Has_Roll_Rule': False,
        
        # ASC 覆盖范围
        'ASC_Start': None,
        'ASC_End': None,
        'ASC_In_Test_Period': False,
        
        # Roll Dates 数量
        'Theory_Rolls_All': 0,
        'Theory_Rolls_Test': 0,
        'ASC_Rolls': 0,
        'RAD_Rolls_Test': 0,
        
        # 验证结果 (ASC 覆盖期)
        'Theory_vs_ASC_Match_Rate': np.nan,
        'Theory_vs_ASC_Avg_Diff': np.nan,
        
        # 验证结果 (测试期)
        'Theory_vs_RAD_Match_Rate': np.nan,
        'Theory_vs_RAD_Avg_Diff': np.nan,
        
        # 数据质量
        'RAD_NON_Corr': np.nan,
        'Max_Daily_Gap': np.nan,
        
        # 综合评分
        'Theory_Reliability': 'Unknown',  # Verified / Assumed / Unknown
        'RAD_Trustworthiness': 'Unknown',  # High / Medium / Low / Unknown
        'Quality_Score': 'F',
    }
    
    # 加载数据
    asc = load_asc(symbol)
    rad = load_rad(symbol)
    non = load_non(symbol)
    
    # 检查 ASC
    if asc is not None:
        result['Has_ASC'] = True
        result['ASC_Start'] = asc['Date'].min()
        result['ASC_End'] = asc['Date'].max()
        result['ASC_In_Test_Period'] = result['ASC_End'] >= pd.Timestamp('2011-01-01')
        
        # 检测 ASC Roll Dates
        asc_rolls = detect_asc_roll_dates(asc)
        result['ASC_Rolls'] = len(asc_rolls)
        
        # 理论 Roll Dates (ASC 覆盖期)
        if len(asc) > 0:
            start_year = asc['Date'].dt.year.min()
            end_year = asc['Date'].dt.year.max()
            theory_rolls_asc_period = get_theoretical_roll_dates(symbol, start_year, end_year)
            
            # 验证理论 vs ASC
            match_rate, avg_diff, matches, total = match_dates(
                pd.to_datetime(theory_rolls_asc_period), 
                pd.to_datetime(asc_rolls)
            )
            result['Theory_vs_ASC_Match_Rate'] = match_rate
            result['Theory_vs_ASC_Avg_Diff'] = avg_diff
            
            # 理论规则可靠性
            if match_rate >= 0.90:
                result['Theory_Reliability'] = 'Verified'
            elif match_rate >= 0.70:
                result['Theory_Reliability'] = 'Partial'
            else:
                result['Theory_Reliability'] = 'Unverified'
    
    # 检查 RAD
    if rad is not None:
        result['Has_RAD'] = True
        
        # 测试期 RAD Roll Dates
        test_start = pd.Timestamp('2011-01-01')
        test_end = pd.Timestamp('2019-12-31')
        rad_test = rad[(rad['Date'] >= test_start) & (rad['Date'] <= test_end)]
        
        if non is not None:
            non_test = non[(non['Date'] >= test_start) & (non['Date'] <= test_end)]
            
            # RAD Roll Dates
            rad_rolls_test = detect_rad_roll_dates(rad_test, non_test)
            result['RAD_Rolls_Test'] = len(rad_rolls_test)
            
            # 理论 Roll Dates (测试期)
            theory_rolls_test = get_theoretical_roll_dates(symbol, 2011, 2019)
            result['Theory_Rolls_Test'] = len(theory_rolls_test)
            result['Theory_Rolls_All'] = len(get_theoretical_roll_dates(symbol, 1970, 2020))
            
            # 验证理论 vs RAD
            if theory_rolls_test and rad_rolls_test:
                match_rate, avg_diff, matches, total = match_dates(
                    pd.to_datetime(theory_rolls_test),
                    pd.to_datetime(rad_rolls_test)
                )
                result['Theory_vs_RAD_Match_Rate'] = match_rate
                result['Theory_vs_RAD_Avg_Diff'] = avg_diff
            
            # RAD/NON 相关性
            merged = pd.merge(rad_test, non_test, on='Date', suffixes=('_rad', '_non'))
            if len(merged) > 0:
                result['RAD_NON_Corr'] = merged['C_rad'].corr(merged['C_non'])
                result['Max_Daily_Gap'] = merged['C_rad'].pct_change().abs().max()
    
    # 检查 Roll Rule
    for rname, rdata in ROLL_RULES.items():
        if symbol in rdata.get('symbols', []):
            result['Has_Roll_Rule'] = True
            break
    
    # 综合评分
    if not result['Has_RAD'] or not result['Has_Roll_Rule']:
        result['Quality_Score'] = 'F'
        result['RAD_Trustworthiness'] = 'Unknown'
    elif result['Theory_Reliability'] == 'Verified' and result['Theory_vs_RAD_Match_Rate'] >= 0.80:
        result['RAD_Trustworthiness'] = 'High'
        result['Quality_Score'] = 'A'
    elif result['Theory_Reliability'] in ['Verified', 'Partial'] and result['Theory_vs_RAD_Match_Rate'] >= 0.60:
        result['RAD_Trustworthiness'] = 'Medium'
        result['Quality_Score'] = 'B'
    elif result['RAD_NON_Corr'] >= 0.90:
        result['RAD_Trustworthiness'] = 'Medium'
        result['Quality_Score'] = 'B'
    elif result['RAD_NON_Corr'] >= 0.70:
        result['RAD_Trustworthiness'] = 'Low'
        result['Quality_Score'] = 'C'
    else:
        result['RAD_Trustworthiness'] = 'Low'
        result['Quality_Score'] = 'C'
    
    return result


def main():
    print("="*120)
    print("CLC 数据交叉验证 v2 - 理论规则验证")
    print("="*120)
    print("\n逻辑链:")
    print("1. ASC (测试期外) vs 理论规则 → 验证理论规则可靠性")
    print("2. 理论规则 (测试期内) → 推导 Roll Dates")
    print("3. 理论 Roll Dates vs RAD 跳变 → 验证 RAD 可靠性")
    print()
    
    results = []
    for symbol in ALL_CONTRACTS:
        result = validate_symbol(symbol)
        results.append(result)
        
        # 简洁输出
        theory_status = "✅" if result['Theory_Reliability'] == 'Verified' else "⚠️" if result['Theory_Reliability'] == 'Partial' else "❓"
        rad_status = "✅" if result['RAD_Trustworthiness'] == 'High' else "✅" if result['RAD_Trustworthiness'] == 'Medium' else "⚠️" if result['RAD_Trustworthiness'] == 'Low' else "❓"
        
        print(f"{symbol}: Theory={theory_status} ({result['Theory_vs_ASC_Match_Rate']:.0%} if ASC), "
              f"RAD={rad_status} ({result['Theory_vs_RAD_Match_Rate']:.0%}), "
              f"Score={result['Quality_Score']}")
    
    # 保存结果
    RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / 'cross_validation_v2.csv', index=False)
    
    # 汇总统计
    print("\n" + "="*120)
    print("汇总统计")
    print("="*120)
    
    # ASC 覆盖情况
    asc_in_period = sum(1 for r in results if r['ASC_In_Test_Period'])
    print(f"\nASC 覆盖测试期：{asc_in_period}/{len(results)} ({asc_in_period/len(results)*100:.0f}%)")
    
    # 理论规则可靠性
    theory_verified = sum(1 for r in results if r['Theory_Reliability'] == 'Verified')
    theory_partial = sum(1 for r in results if r['Theory_Reliability'] == 'Partial')
    print(f"理论规则验证通过：{theory_verified}/{len(results)} ({theory_verified/len(results)*100:.0f}%)")
    print(f"理论规则部分验证：{theory_partial}/{len(results)} ({theory_partial/len(results)*100:.0f}%)")
    
    # RAD 可信度
    rad_high = sum(1 for r in results if r['RAD_Trustworthiness'] == 'High')
    rad_medium = sum(1 for r in results if r['RAD_Trustworthiness'] == 'Medium')
    rad_low = sum(1 for r in results if r['RAD_Trustworthiness'] == 'Low')
    print(f"\nRAD 可信度高：{rad_high}/{len(results)} ({rad_high/len(results)*100:.0f}%)")
    print(f"RAD 可信度中：{rad_medium}/{len(results)} ({rad_medium/len(results)*100:.0f}%)")
    print(f"RAD 可信度低：{rad_low}/{len(results)} ({rad_low/len(results)*100:.0f}%)")
    
    # 质量评分
    print(f"\n质量评分分布:")
    for score in ['A', 'B', 'C', 'F']:
        count = sum(1 for r in results if r['Quality_Score'] == score)
        print(f"  {score}: {count}/{len(results)} ({count/len(results)*100:.0f}%)")
    
    # 详细表格
    print("\n" + "="*120)
    print("详细结果 (按理论规则可靠性排序)")
    print("="*120)
    
    df_sorted = df.sort_values('Theory_Reliability', ascending=False)
    display_cols = ['Symbol', 'Theory_Reliability', 'Theory_vs_ASC_Match_Rate', 
                    'Theory_vs_RAD_Match_Rate', 'RAD_Trustworthiness', 'Quality_Score']
    
    print(df_sorted[display_cols].to_string(index=False))
    
    print(f"\n输出文件：{RESULTS_DIR / 'cross_validation_v2.csv'}")
    print("="*120)


if __name__ == '__main__':
    main()
