"""
CLC 数据交叉验证脚本 (简化版)

重要发现：ASC 文件不覆盖测试期 (2011-2019)
- 0/22 D-class 合约的 ASC 覆盖测试期
- 15/22 在 2000 年前结束
- 7/22 在 2000s 结束 (最晚到 2008-2010)

因此验证策略调整为：
1. Roll Date: 理论规则 vs RAD 跳变检测
2. Roll Price: 无法验证 (ASC 不在测试期)
3. 数据完整性：RAD vs NON 相关性 + 异常检测
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


def get_theoretical_roll_dates(symbol, start_year=2011, end_year=2019):
    """根据理论规则计算 roll dates"""
    # 找到 symbol 对应的规则
    rule_name = None
    for rname, rdata in ROLL_RULES.items():
        if symbol in rdata.get('symbols', []):
            rule_name = rname
            break
    
    if rule_name is None:
        return []
    
    # 解析规则
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
        return []  # 不支持的规则类型
    
    # 获取合约月份
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


def detect_roll_dates_from_rad(rad_df, non_df, threshold=0.01):
    """从 RAD 数据检测 roll dates (ratio 跳变点)"""
    if rad_df is None or non_df is None:
        return []
    
    # 合并
    merged = pd.merge(rad_df, non_df, on='Date', suffixes=('_rad', '_non'))
    merged['ratio'] = merged['C_rad'] / merged['C_non']
    merged['ratio_change'] = merged['ratio'].diff().abs()
    
    # 检测跳变
    jumps = merged[merged['ratio_change'] > threshold]
    return jumps['Date'].tolist()


def validate_data_integrity(symbol):
    """验证数据完整性"""
    result = {
        'Symbol': symbol,
        'NON_Exists': False,
        'RAD_Exists': False,
        'Has_Roll_Rule': False,
        'Theory_Roll_Count': 0,
        'RAD_Detected_Rolls': 0,
        'RAD_NON_Correlation': np.nan,
        'Max_Single_Day_Gap': np.nan,
        'Large_Gaps_Count': 0,
        'Anomaly_Detected': False,
        'Quality_Score': 'F',
        'Notes': '',
    }
    
    non = load_non(symbol)
    rad = load_rad(symbol)
    
    if non is not None:
        result['NON_Exists'] = True
    
    if rad is not None:
        result['RAD_Exists'] = True
    
    # 检查是否有 roll rule
    for rname, rdata in ROLL_RULES.items():
        if symbol in rdata.get('symbols', []):
            result['Has_Roll_Rule'] = True
            break
    
    if non is None or rad is None:
        return result
    
    # 计算理论 roll dates
    theory_dates = get_theoretical_roll_dates(symbol)
    result['Theory_Roll_Count'] = len(theory_dates)
    
    # 检测 RAD 跳变
    rad_rolls = detect_roll_dates_from_rad(rad, non)
    result['RAD_Detected_Rolls'] = len(rad_rolls)
    
    # 过滤测试期
    test_start = pd.Timestamp('2011-01-01')
    test_end = pd.Timestamp('2019-12-31')
    
    non_test = non[(non['Date'] >= test_start) & (non['Date'] <= test_end)]
    rad_test = rad[(rad['Date'] >= test_start) & (rad['Date'] <= test_end)]
    
    if len(non_test) == 0 or len(rad_test) == 0:
        result['Notes'] = 'No data in test period'
        return result
    
    # 合并测试期数据
    merged = pd.merge(rad_test, non_test, on='Date', suffixes=('_rad', '_non'))
    
    if len(merged) == 0:
        result['Notes'] = 'Cannot merge RAD and NON'
        return result
    
    # 1. 相关性
    corr = merged['C_rad'].corr(merged['C_non'])
    result['RAD_NON_Correlation'] = corr
    
    # 2. 检测单日跳空
    merged['ret_rad'] = merged['C_rad'].pct_change()
    max_gap = merged['ret_rad'].abs().max()
    result['Max_Single_Day_Gap'] = max_gap
    result['Large_Gaps_Count'] = len(merged[merged['ret_rad'].abs() > 0.05])
    
    # 3. 质量评分
    if max_gap > 0.50:  # >50% 单日跳变
        result['Anomaly_Detected'] = True
        result['Quality_Score'] = 'D'
        result['Notes'] = f'Large gap detected: {max_gap:.1%}'
    elif corr < 0.90:
        result['Quality_Score'] = 'C'
        result['Notes'] = f'Low correlation: {corr:.3f}'
    elif corr < 0.95:
        result['Quality_Score'] = 'B'
    else:
        result['Quality_Score'] = 'A'
    
    return result


def main():
    print("="*100)
    print("CLC 数据交叉验证 (简化版)")
    print("="*100)
    print("\n重要发现：ASC 文件不覆盖测试期 (2011-2019)")
    print("验证策略：理论规则 vs RAD 跳变检测 + 数据完整性评分\n")
    
    # 验证所有合约
    results = []
    for symbol in ALL_CONTRACTS:
        result = validate_data_integrity(symbol)
        results.append(result)
        
        status = "✅" if result['Quality_Score'] in ['A', 'B'] else "⚠️" if result['Quality_Score'] == 'C' else "❌"
        corr_str = f"{result['RAD_NON_Correlation']:.3f}" if pd.notna(result['RAD_NON_Correlation']) else 'N/A'
        print(f"{status} {symbol}: Score={result['Quality_Score']}, "
              f"Corr={corr_str}, "
              f"Theory={result['Theory_Roll_Count']}, Detected={result['RAD_Detected_Rolls']}")
    
    # 保存结果
    RESULTS_DIR.mkdir(exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / 'data_quality_full_validation.csv', index=False)
    
    # 汇总统计
    print("\n" + "="*100)
    print("数据质量评分分布:")
    print(df['Quality_Score'].value_counts().to_string())
    
    print("\n按评分分类:")
    for score in ['A', 'B', 'C', 'D', 'F']:
        contracts = df[df['Quality_Score'] == score]['Symbol'].tolist()
        if contracts:
            print(f"  {score}: {len(contracts)} - {', '.join(contracts[:10])}{'...' if len(contracts) > 10 else ''}")
    
    anomalies = df[df['Anomaly_Detected'] == True]['Symbol'].tolist()
    if anomalies:
        print(f"\n⚠️  检测到异常的合约：{anomalies}")
    
    print(f"\n输出文件：{RESULTS_DIR / 'data_quality_full_validation.csv'}")
    print("="*100)


if __name__ == '__main__':
    main()
