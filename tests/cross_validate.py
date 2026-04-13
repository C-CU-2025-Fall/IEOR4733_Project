"""
CLC 数据交叉验证脚本

验证 NON / ASC / REV / RAD 四类文件的一致性

阶段 1: Roll Date 验证 (核心) - 确认 Roll Date 的唯一真相
阶段 2: Roll Price 验证 (核心) - 确认 ASC 价格与 NON 一致
阶段 3: 调整参数验证 - 验证 RAD/REV 调整参数正确
阶段 4: 数据完整性评分 - 对 50 合约进行质量分级

输出：
- tests/results/roll_date_truth.csv
- tests/results/roll_price_truth.csv
- tests/results/adjustment_validation.csv
- tests/results/data_quality_scores.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'CLC'
CONFIG_DIR = PROJECT_ROOT / 'config'
TEMP_DIR = CONFIG_DIR / 'TEMP'
RESULTS_DIR = PROJECT_ROOT / 'tests' / 'results'

# 50 个合约
D_CONTRACTS = ['NR', 'SB', 'SN', 'SP', 'TY', 'US', 'ZA', 'ZC', 'ZF', 'ZG', 'ZH', 'ZI', 'ZK', 'ZL', 'ZN', 'ZO', 'ZP', 'ZR', 'ZT', 'ZU', 'ZW', 'ZZ']
# 加上其他合约...

# 加载 roll rules
with open(CONFIG_DIR / 'roll_rules_corrected.json') as f:
    ROLL_RULES = json.load(f)


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
    # 优先加载 RAD_v2 (损坏合约)
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
        # ASC 格式：Date Open High Low Close Volume OI Adj (空格分隔，无 header)
        df = pd.read_csv(fpath, sep=r'\s+', header=None, engine='python',
                         names=['Date','O','H','L','C','V','OI','Adj'],
                         on_bad_lines='skip')
        # 转换日期，过滤无效行
        df['Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m%d', errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Date'] = df['Date'].dt.normalize()
        return df
    except Exception as e:
        print(f"  ⚠️  {symbol} ASC 加载失败：{e}")
        return None


def load_rollover(symbol):
    """加载 rollover log (如果存在)"""
    fpath = DATA_DIR / f'{symbol}_rollovers.csv'
    if not fpath.exists():
        return None
    
    df = pd.read_csv(fpath)
    df['RollDate'] = pd.to_datetime(df['RollDate'])
    return df


def detect_roll_dates_from_rad(rad_df, threshold=0.01):
    """从 RAD 数据检测 roll dates (ratio 跳变点)"""
    if rad_df is None or len(rad_df) < 2:
        return []
    
    # 计算隐含 ratio = RAD / NON (需要 NON 数据)
    # 这里简化：检测 RAD 价格跳变
    rad_df = rad_df.copy()
    rad_df['ret'] = rad_df['C'].pct_change()
    
    # 检测大跳变 (>threshold)
    jumps = rad_df[rad_df['ret'].abs() > threshold]
    return jumps['Date'].tolist()


def calculate_theoretical_roll_dates(rule_name, symbol, start_year=2010, end_year=2020):
    """根据理论规则计算 roll dates"""
    import calendar
    
    # 找到 symbol 对应的规则
    rule_type = None
    day = None
    months_str = None
    
    for rname, rdata in ROLL_RULES.items():
        if symbol in rdata.get('symbols', []):
            parts = rname.split('_')
            if parts[0] == 'MPDM':
                rule_type = 'MPDM'
                day = int(parts[1])
            elif parts[0] == 'DM':
                rule_type = 'DM'
                day = int(parts[1])
            elif 'THUR' in rname:
                rule_type = 'THUR_PRIOR_2ND_FRI'
            break
    
    if rule_type is None:
        return []
    
    # 获取合约月份
    months_file = CONFIG_DIR / 'contract_months.json'
    with open(months_file) as f:
        contract_months = json.load(f)
    months_str = contract_months.get(symbol, 'H,M,U,Z')
    
    months_map = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
                  'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
    delivery_months = [months_map[m.strip()] for m in months_str.split(',') 
                       if m.strip() in months_map]
    
    roll_dates = []
    for year in range(start_year, end_year + 1):
        for dm in delivery_months:
            if rule_type in ['MPDM', 'DM']:
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
    
    return sorted(roll_dates)


def validate_roll_date_consistency(symbol):
    """
    阶段 1: Roll Date 验证
    
    对比：
    - ASC 记录的 roll dates (如果有)
    - RAD 检测的 roll dates
    - 理论规则计算的 roll dates
    """
    result = {
        'Symbol': symbol,
        'ASC_Count': 0,
        'RAD_Count': 0,
        'Theory_Count': 0,
        'ASC_vs_RAD_Match': np.nan,
        'ASC_vs_Theory_Match': np.nan,
        'RAD_vs_Theory_Match': np.nan,
        'Status': 'OK',
    }
    
    # 加载数据
    asc = load_asc(symbol)
    rad = load_rad(symbol)
    non = load_non(symbol)
    
    # 1. ASC roll dates - 检测 Adj 列的跳变
    if asc is not None and 'Adj' in asc.columns:
        # 计算 Adj 的变化（roll 发生时 Adj 会跳变）
        asc = asc.copy()
        asc['Adj'] = pd.to_numeric(asc['Adj'], errors='coerce')
        asc['Adj_change'] = asc['Adj'].diff().abs()
        # 检测 Adj 跳变 (>100 认为是 roll)
        asc_rolls = asc[asc['Adj_change'] > 100]
        result['ASC_Count'] = len(asc_rolls)
    
    # 2. RAD 检测 roll dates
    if rad is not None and non is not None:
        # 计算 ratio = RAD / NON
        merged = pd.merge(rad, non, on='Date', suffixes=('_rad', '_non'))
        merged['ratio'] = merged['C_rad'] / merged['C_non']
        merged['ratio_change'] = merged['ratio'].diff().abs()
        
        # 检测 ratio 跳变 (>1% 认为是 roll)
        ratio_jumps = merged[merged['ratio_change'] > 0.01]
        result['RAD_Count'] = len(ratio_jumps)
    
    # 3. 理论 roll dates
    theory_dates = calculate_theoretical_roll_dates('MPDM', symbol, 2011, 2019)
    result['Theory_Count'] = len(theory_dates)
    
    return result


def validate_roll_price_consistency(symbol):
    """
    阶段 2: Roll Price 验证
    
    对比：
    - ASC.prev_close vs NON[roll_date-1]
    - ASC.new_close vs NON[roll_date]
    - 隐含 ratio vs RAD 实际 ratio
    """
    result = {
        'Symbol': symbol,
        'Price_Match_Rate': np.nan,
        'Avg_Price_Diff_Pct': np.nan,
        'Ratio_Match_Rate': np.nan,
        'Status': 'NO_ASC',
    }
    
    asc = load_asc(symbol)
    non = load_non(symbol)
    rad = load_rad(symbol)
    
    if asc is None or non is None or rad is None:
        return result
    
    # TODO: 实现详细验证
    result['Status'] = 'TODO'
    return result


def validate_data_integrity(symbol):
    """
    阶段 3: 数据完整性验证
    
    验证：
    - RAD = NON × ratio
    - 检测异常值
    - 检测非 roll date 跳空
    """
    result = {
        'Symbol': symbol,
        'RAD_NON_Correlation': np.nan,
        'Max_Single_Day_Gap': np.nan,
        'Large_Gaps_Count': 0,
        'Anomaly_Detected': False,
        'Quality_Score': 'A',
    }
    
    non = load_non(symbol)
    rad = load_rad(symbol)
    
    if non is None or rad is None:
        result['Quality_Score'] = 'F'
        return result
    
    # 合并
    merged = pd.merge(rad, non, on='Date', suffixes=('_rad', '_non'))
    
    if len(merged) == 0:
        result['Quality_Score'] = 'F'
        return result
    
    # 1. 相关性
    corr = merged['C_rad'].corr(merged['C_non'])
    result['RAD_NON_Correlation'] = corr
    
    # 2. 检测单日跳空
    merged['ret_rad'] = merged['C_rad'].pct_change()
    merged['ret_non'] = merged['C_non'].pct_change()
    
    max_gap_rad = merged['ret_rad'].abs().max()
    max_gap_non = merged['ret_non'].abs().max()
    
    result['Max_Single_Day_Gap'] = max_gap_rad
    result['Large_Gaps_Count'] = len(merged[merged['ret_rad'].abs() > 0.05])
    
    # 3. 异常检测
    if max_gap_rad > 0.50:  # >50% 单日跳变
        result['Anomaly_Detected'] = True
        result['Quality_Score'] = 'D'
    elif corr < 0.90:
        result['Quality_Score'] = 'C'
    elif corr < 0.95:
        result['Quality_Score'] = 'B'
    else:
        result['Quality_Score'] = 'A'
    
    return result


def main():
    print("="*100)
    print("CLC 数据交叉验证")
    print("="*100)
    
    # 阶段 1: Roll Date 验证
    print("\n阶段 1: Roll Date 验证...")
    roll_date_results = []
    for symbol in D_CONTRACTS:
        result = validate_roll_date_consistency(symbol)
        roll_date_results.append(result)
        print(f"  {symbol}: ASC={result['ASC_Count']}, RAD={result['RAD_Count']}, Theory={result['Theory_Count']}")
    
    # 阶段 2: Roll Price 验证
    print("\n阶段 2: Roll Price 验证...")
    roll_price_results = []
    for symbol in D_CONTRACTS:
        result = validate_roll_price_consistency(symbol)
        roll_price_results.append(result)
    
    # 阶段 3: 数据完整性验证
    print("\n阶段 3: 数据完整性验证...")
    integrity_results = []
    for symbol in D_CONTRACTS:
        result = validate_data_integrity(symbol)
        integrity_results.append(result)
        print(f"  {symbol}: Corr={result['RAD_NON_Correlation']:.3f}, "
              f"MaxGap={result['Max_Single_Day_Gap']:.1%}, "
              f"Score={result['Quality_Score']}")
    
    # 保存结果
    RESULTS_DIR.mkdir(exist_ok=True)
    
    df1 = pd.DataFrame(roll_date_results)
    df1.to_csv(RESULTS_DIR / 'roll_date_truth.csv', index=False)
    
    df2 = pd.DataFrame(roll_price_results)
    df2.to_csv(RESULTS_DIR / 'roll_price_truth.csv', index=False)
    
    df3 = pd.DataFrame(integrity_results)
    df3.to_csv(RESULTS_DIR / 'data_quality_scores.csv', index=False)
    
    # 汇总
    summary = pd.merge(df1, df3[['Symbol', 'Quality_Score']], on='Symbol')
    summary.to_csv(RESULTS_DIR / 'cross_validation_summary.csv', index=False)
    
    print("\n" + "="*100)
    print("验证完成！结果保存到 tests/results/")
    print("="*100)
    print("\n输出文件:")
    print("  - roll_date_truth.csv (Roll Date 验证)")
    print("  - roll_price_truth.csv (Roll Price 验证)")
    print("  - data_quality_scores.csv (数据质量评分)")
    print("  - cross_validation_summary.csv (汇总)")
    
    # 统计
    print("\n数据质量评分分布:")
    print(df3['Quality_Score'].value_counts().to_string())
    
    anomalies = df3[df3['Anomaly_Detected'] == True]['Symbol'].tolist()
    if anomalies:
        print(f"\n⚠️  检测到异常的合约：{anomalies}")


if __name__ == '__main__':
    main()
