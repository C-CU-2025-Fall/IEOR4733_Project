"""
RAD_v2 生成与验证测试

测试 CLC RAD 数据生成方法，验证与原始 CLC RAD 的一致性。

Usage:
    python tests/test_rad_v2.py
    
配置:
    config/roll_rules_corrected.json - 合约的 roll rules
    config/contract_months.json - 合约的交割月份映射
"""

import pandas as pd
import calendar
import json
from pathlib import Path

# 配置
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT / 'data' / 'CLC'
CONFIG_FILE = _PROJECT_ROOT / 'config' / 'roll_rules_corrected.json'
CONTRACT_MONTHS_FILE = _PROJECT_ROOT / 'config' / 'contract_months.json'
TEST_START = pd.Timestamp('2011-01-01')
TEST_END = pd.Timestamp('2019-12-31')


def get_nth_weekday(year, month, n, weekday):
    """获取某月第 n 个星期几"""
    first_day = calendar.weekday(year, month, 1)
    if first_day <= weekday:
        first = weekday - first_day + 1
    else:
        first = 7 - first_day + weekday + 1
    return first + (n - 1) * 7


def get_2nd_friday(year, month):
    """获取某月第 2 个星期五"""
    return get_nth_weekday(year, month, 2, 4)  # Friday = 4


def get_roll_dates(rule_type, day, contract_months_str, start_year, end_year):
    """
    生成理论 roll dates
    
    参数:
        rule_type: THUR_PRIOR_2ND_FRI | DM | MPDM
        day: 日期数字 (DM/MPDM 用)
        contract_months_str: 合约月份字符串如 "H,M,U,Z"
        start_year: 起始年份
        end_year: 结束年份
    
    返回:
        排序后的 roll dates 列表
    """
    months_map = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
                  'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
    
    delivery_months = [months_map[m.strip()] for m in contract_months_str.split(',') 
                       if m.strip() in months_map]
    
    roll_dates = []
    for year in range(start_year, end_year + 1):
        for dm in delivery_months:
            if rule_type == 'THUR_PRIOR_2ND_FRI':
                # Thur prior 2nd Fri of DM
                second_friday = get_2nd_friday(year, dm)
                roll_day = second_friday - 1  # 前一天是星期四
                try:
                    rd = pd.Timestamp(year=year, month=dm, day=roll_day)
                    roll_dates.append(rd)
                except:
                    pass
            elif rule_type == 'DM':
                try:
                    rd = pd.Timestamp(year=year, month=dm, day=day)
                    roll_dates.append(rd)
                except:
                    pass
            elif rule_type == 'MPDM':
                pm_year = year - 1 if dm == 1 else year
                pm_month = 12 if dm == 1 else dm - 1
                try:
                    rd = pd.Timestamp(year=pm_year, month=pm_month, day=day)
                    roll_dates.append(rd)
                except:
                    pass
    
    return sorted(roll_dates)


def match_to_trading_days(roll_dates, trading_days, max_diff_days=5):
    """
    将理论 roll dates 匹配到实际交易日
    
    参数:
        roll_dates: 理论 roll dates 列表
        trading_days: 实际交易日 Series
        max_diff_days: 最大允许差异天数
    
    返回:
        匹配到的实际交易日索引列表
    """
    actual_rolls = []
    for rd in roll_dates:
        idx = trading_days[trading_days == rd].index
        if len(idx) > 0:
            actual_rolls.append(idx[0])
        else:
            diffs = (trading_days - rd).abs()
            closest_idx = diffs.idxmin()
            if diffs[closest_idx].days <= max_diff_days:
                actual_rolls.append(closest_idx)
    
    return sorted(set(actual_rolls))


def generate_rad_v2(non_file, output_file, rule_type, day, contract_months):
    """
    生成 RAD_v2
    
    方法:
        - ratio 从数据第一天 = 1.0 开始
        - 在每个 roll date, ratio *= (prev_Close / current_Close)
        - RAD = NON * ratio
        - 自然结果: 第一个 roll 之前 RAD = NON (ratio=1.0)
    
    参数:
        non_file: NON 数据文件路径
        output_file: 输出文件路径
        rule_type: roll rule 类型
        day: 日期数字
        contract_months: 合约月份字符串
    
    返回:
        (roll_count, row_count, status)
    """
    cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInterest']
    df = pd.read_csv(non_file, names=cols)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    start_year = df['Date'].dt.year.min()
    end_year = df['Date'].dt.year.max()
    
    roll_dates = get_roll_dates(rule_type, day, contract_months, start_year, end_year)
    actual_rolls = match_to_trading_days(roll_dates, df['Date'])
    
    if len(actual_rolls) == 0:
        return 0, len(df), "NO_ROLLS"
    
    # 初始化 ratio = 1.0 for all rows
    df['ratio'] = 1.0
    
    # 从第一个 roll 开始，逐个应用 ratio 变化
    # 在 roll date: ratio *= (prev_day_Close / roll_day_Close)
    # 即 roll day 及之后的 ratio 是累积后的值
    current_ratio = 1.0
    for i, roll_idx in enumerate(actual_rolls):
        if roll_idx > 0 and roll_idx < len(df):
            # c = 前一天 close, C = roll day close
            c = df.iloc[roll_idx - 1]['Close']
            C = df.iloc[roll_idx]['Close']
            
            if c > 0 and C > 0:
                ratio_change = c / C
                current_ratio *= ratio_change
                
                # 这个 ratio 从 roll_idx 到下一个 roll 之前
                next_roll = actual_rolls[i + 1] if i + 1 < len(actual_rolls) else len(df)
                df.loc[roll_idx:next_roll - 1, 'ratio'] = current_ratio
    
    # 生成 RAD
    df['RAD_Open'] = df['Open'] * df['ratio']
    df['RAD_High'] = df['High'] * df['ratio']
    df['RAD_Low'] = df['Low'] * df['ratio']
    df['RAD_Close'] = df['Close'] * df['ratio']
    
    output_df = df[['Date', 'RAD_Open', 'RAD_High', 'RAD_Low', 'RAD_Close', 'Volume', 'OpenInterest']].copy()
    output_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInterest']
    output_df['Date'] = output_df['Date'].dt.strftime('%m/%d/%Y')
    output_df.to_csv(output_file, index=False, header=False)
    
    return len(actual_rolls), len(df), "OK"


def validate_rad_v2(symbol, data_dir=DATA_DIR):
    """
    验证 RAD_v2 与 CLC RAD 的一致性
    
    参数:
        symbol: 合约符号
        data_dir: 数据目录
    
    返回:
        验证结果字典，或 None 如果文件不存在
    """
    rad_file = data_dir / f'{symbol}_RAD.CSV'
    v2_file = data_dir / f'{symbol}_RAD_v2.CSV'
    non_file = data_dir / f'{symbol}_NON.CSV'
    
    if not all(f.exists() for f in [rad_file, v2_file, non_file]):
        return None
    
    rad = pd.read_csv(rad_file, names=['Date', 'O', 'H', 'L', 'C', 'V', 'OI'])
    v2 = pd.read_csv(v2_file, names=['Date', 'O', 'H', 'L', 'C', 'V', 'OI'])
    non = pd.read_csv(non_file, names=['Date', 'O', 'H', 'L', 'C', 'V', 'OI'])
    
    rad['Date'] = pd.to_datetime(rad['Date'])
    v2['Date'] = pd.to_datetime(v2['Date'])
    non['Date'] = pd.to_datetime(non['Date'])
    
    # 过滤测试期
    rad_test = rad[(rad['Date'] >= TEST_START) & (rad['Date'] <= TEST_END)]
    v2_test = v2[(v2['Date'] >= TEST_START) & (v2['Date'] <= TEST_END)]
    non_test = non[(non['Date'] >= TEST_START) & (non['Date'] <= TEST_END)]
    
    if len(rad_test) == 0 or len(v2_test) == 0 or len(non_test) == 0:
        return None
    
    min_len = min(len(rad_test), len(v2_test), len(non_test))
    rad_test = rad_test.iloc[:min_len].reset_index(drop=True)
    v2_test = v2_test.iloc[:min_len].reset_index(drop=True)
    non_test = non_test.iloc[:min_len].reset_index(drop=True)
    
    # 计算指标
    corr_v2_rad = v2_test['C'].corr(rad_test['C'])
    corr_v2_non = v2_test['C'].corr(non_test['C'])
    first_diff = abs(v2_test['C'].iloc[0] - non_test['C'].iloc[0]) / non_test['C'].iloc[0] * 100
    last_diff = abs(v2_test['C'].iloc[-1] - rad_test['C'].iloc[-1]) / rad_test['C'].iloc[-1] * 100 if rad_test['C'].iloc[-1] != 0 else None
    
    return {
        'Symbol': symbol,
        'Corr(v2,RAD)': corr_v2_rad,
        'Corr(v2,NON)': corr_v2_non,
        'First_Diff%': first_diff,
        'Last_Diff%': last_diff
    }


def load_roll_rules(config_file=CONFIG_FILE):
    """加载 roll rules 配置"""
    with open(config_file) as f:
        return json.load(f)


def load_contract_months(config_file=CONTRACT_MONTHS_FILE):
    """加载合约交割月份映射"""
    with open(config_file) as f:
        data = json.load(f)
    # 过滤掉 _source, _method 等元数据
    return {k: v for k, v in data.items() if not k.startswith('_')}


def build_symbol_to_rule(roll_rules, contract_months_map):
    """构建 symbol -> rule 映射，使用 per-symbol 的交割月份"""
    symbol_to_rule = {}
    
    for rule_type, rule_data in roll_rules.items():
        symbols = rule_data.get('symbols', [])
        
        if rule_type == 'THUR_PRIOR_2ND_FRI_OF_DM':
            rule_key = 'THUR_PRIOR_2ND_FRI'
            day = None
        elif rule_type.startswith('DM_'):
            rule_key = 'DM'
            day = int(rule_type.split('_')[1])
        elif rule_type.startswith('MPDM_'):
            rule_key = 'MPDM'
            day = int(rule_type.split('_')[1])
        else:
            continue
        
        for symbol in symbols:
            # 使用从 CLC 数据提取的实际交割月份
            cm = contract_months_map.get(symbol)
            if not cm:
                continue
            symbol_to_rule[symbol] = {
                'rule_type': rule_key,
                'day': day,
                'contract_months': cm
            }
    
    return symbol_to_rule


def run_all_tests():
    """运行全部测试"""
    print("=" * 60)
    print("RAD_v2 生成与验证测试")
    print("=" * 60)
    
    # 加载配置
    roll_rules = load_roll_rules()
    contract_months_map = load_contract_months()
    symbol_to_rule = build_symbol_to_rule(roll_rules, contract_months_map)
    print(f"\n加载 {len(symbol_to_rule)} 个合约的 roll rules + contract months")
    
    # 找到所有 NON 文件
    non_files = sorted(DATA_DIR.glob('*_NON.CSV'))
    print(f"找到 {len(non_files)} 个 NON 文件")
    
    # 统计缺失的 contract months
    missing_cm = []
    for non_file in non_files:
        symbol = non_file.stem.replace('_NON', '')
        if symbol not in contract_months_map:
            missing_cm.append(symbol)
    if missing_cm:
        print(f"注意: {len(missing_cm)} 个合约无 contract months: {missing_cm}")
    
    # 生成 RAD_v2
    print("\n生成 RAD_v2...")
    results = []
    for non_file in non_files:
        symbol = non_file.stem.replace('_NON', '')
        output_file = DATA_DIR / f'{symbol}_RAD_v2.CSV'
        
        rule = symbol_to_rule.get(symbol)
        if not rule:
            results.append((symbol, 'NO_RULE', 0, 0))
            continue
        
        try:
            rolls, rows, status = generate_rad_v2(
                non_file, output_file,
                rule['rule_type'], rule['day'], rule['contract_months']
            )
            results.append((symbol, status, rolls, rows))
        except Exception as e:
            results.append((symbol, f'ERROR: {e}', 0, 0))
    
    ok_count = len([r for r in results if r[1] == "OK"])
    print(f"成功生成：{ok_count}/{len(non_files)}")
    
    # 保存生成结果
    results_df = pd.DataFrame(results, columns=['Symbol', 'Status', 'Rolls', 'Rows'])
    results_df.to_csv(DATA_DIR / 'rad_v2_corrected_summary.csv', index=False)
    
    # 验证
    print("\n验证 RAD_v2...")
    validation_results = []
    for symbol, status, _, _ in results:
        if status != "OK":
            continue
        
        val = validate_rad_v2(symbol)
        if val:
            validation_results.append(val)
    
    val_df = pd.DataFrame(validation_results)
    val_df.to_csv(DATA_DIR / 'rad_v2_corrected_validation.csv', index=False)
    
    if len(val_df) == 0:
        print("无验证结果")
        return results_df, val_df
    
    # 统计
    print(f"\n验证合约数：{len(val_df)}")
    print(f"\n相关性分布:")
    print(f"  ≥0.99: {len(val_df[val_df['Corr(v2,RAD)']>=0.99])} ({len(val_df[val_df['Corr(v2,RAD)']>=0.99])/len(val_df)*100:.1f}%)")
    print(f"  0.95-0.99: {len(val_df[(val_df['Corr(v2,RAD)']>=0.95) & (val_df['Corr(v2,RAD)']<0.99)])} ({len(val_df[(val_df['Corr(v2,RAD)']>=0.95) & (val_df['Corr(v2,RAD)']<0.99)])/len(val_df)*100:.1f}%)")
    print(f"  <0.95: {len(val_df[val_df['Corr(v2,RAD)']<0.95])} ({len(val_df[val_df['Corr(v2,RAD)']<0.95])/len(val_df)*100:.1f}%)")
    
    print(f"\n第一天对齐 NON (<0.01%): {len(val_df[val_df['First_Diff%']<0.01])} ({len(val_df[val_df['First_Diff%']<0.01])/len(val_df)*100:.1f}%)")
    
    valid_last_diff = val_df[val_df['Last_Diff%'].notna()]
    if len(valid_last_diff) > 0:
        print(f"最后一天差异中位数：{valid_last_diff['Last_Diff%'].median():.2f}%")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    
    return results_df, val_df


if __name__ == '__main__':
    run_all_tests()