"""
验证 roll rules：对比 ASC 提取的换月日期 vs 我们计算的 roll dates

Usage:
    python tests/verify_roll_rules.py
"""

import pandas as pd
import calendar
from pathlib import Path

# 项目根目录（相对于当前脚本）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'CLC'
CONFIG_DIR = PROJECT_ROOT / 'config'
CONFIG_FILE = CONFIG_DIR / 'roll_rules_corrected.json'

import json
with open(CONFIG_FILE) as f:
    roll_rules = json.load(f)

# 构建 symbol -> rule 映射
symbol_to_rule = {}
for rule_type, rule_data in roll_rules.items():
    symbols = rule_data.get('symbols', [])
    
    if rule_type == 'THUR_PRIOR_2ND_FRI_OF_DM':
        for symbol in symbols:
            symbol_to_rule[symbol] = {
                'rule_type': 'THUR_PRIOR_2ND_FRI',
                'day': None,
                'contract_months': 'H,M,U,Z'
            }
    elif rule_type.startswith('DM_'):
        day = int(rule_type.split('_')[1])
        for symbol in symbols:
            symbol_to_rule[symbol] = {
                'rule_type': 'DM',
                'day': day,
                'contract_months': 'F,H,K,N,Q,U,X,Z'
            }
    elif rule_type.startswith('MPDM_'):
        day = int(rule_type.split('_')[1])
        for symbol in symbols:
            symbol_to_rule[symbol] = {
                'rule_type': 'MPDM',
                'day': day,
                'contract_months': 'H,M,U,Z'
            }


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
    """生成理论 roll dates"""
    months_map = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
                  'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
    
    delivery_months = [months_map[m.strip()] for m in contract_months_str.split(',') 
                       if m.strip() in months_map]
    
    roll_dates = []
    for year in range(start_year, end_year + 1):
        for dm in delivery_months:
            if rule_type == 'THUR_PRIOR_2ND_FRI':
                second_friday = get_2nd_friday(year, dm)
                roll_day = second_friday - 1
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


def verify_symbol(symbol):
    """验证单个合约的 roll dates"""
    asc_rollover_file = DATA_DIR / f'{symbol}_rollovers.csv'
    
    if not asc_rollover_file.exists():
        return None
    
    # 读取 ASC 提取的换月日期
    asc_df = pd.read_csv(asc_rollover_file)
    asc_df['RollDate'] = pd.to_datetime(asc_df['RollDate'])
    asc_dates = sorted(asc_df['RollDate'].tolist())
    
    # 获取 rule
    rule = symbol_to_rule.get(symbol)
    if not rule:
        return {
            'Symbol': symbol,
            'Status': 'NO_RULE',
            'ASC_Count': len(asc_dates),
            'Calc_Count': 0,
            'Match_Count': 0,
            'Match_Rate': 0
        }
    
    # 计算理论 roll dates
    start_year = asc_dates[0].year - 1
    end_year = asc_dates[-1].year + 1
    calc_dates = get_roll_dates(
        rule['rule_type'], rule['day'], rule['contract_months'],
        start_year, end_year
    )
    
    # 匹配
    asc_set = set(asc_dates)
    calc_set = set(calc_dates)
    match_count = len(asc_set & calc_set)
    
    # 计算匹配率（以 ASC 为基准）
    match_rate = match_count / len(asc_dates) * 100 if len(asc_dates) > 0 else 0
    
    return {
        'Symbol': symbol,
        'Status': 'OK',
        'ASC_Count': len(asc_dates),
        'Calc_Count': len(calc_dates),
        'Match_Count': match_count,
        'Match_Rate': match_rate
    }


def main():
    print("=" * 60)
    print("验证 Roll Rules: ASC 提取 vs 理论计算")
    print("=" * 60)
    
    results = []
    for symbol in symbol_to_rule.keys():
        result = verify_symbol(symbol)
        if result:
            results.append(result)
            if result['Status'] == 'OK':
                print(f"  {symbol}: {result['Match_Count']}/{result['ASC_Count']} ({result['Match_Rate']:.1f}%)")
            else:
                print(f"  {symbol}: {result['Status']}")
    
    result_df = pd.DataFrame(results)
    
    # 保存
    output_file = DATA_DIR / 'roll_rules_verification.csv'
    result_df.to_csv(output_file, index=False)
    print(f"\n已保存到：{output_file}")
    
    # 统计
    ok_df = result_df[result_df['Status'] == 'OK']
    print(f"\n=== 统计 ===")
    print(f"验证合约数：{len(ok_df)}")
    print(f"匹配率 ≥90%: {len(ok_df[ok_df['Match_Rate']>=90])} ({len(ok_df[ok_df['Match_Rate']>=90])/len(ok_df)*100:.1f}%)")
    print(f"匹配率 50-90%: {len(ok_df[(ok_df['Match_Rate']>=50) & (ok_df['Match_Rate']<90)])} ({len(ok_df[(ok_df['Match_Rate']>=50) & (ok_df['Match_Rate']<90)])/len(ok_df)*100:.1f}%)")
    print(f"匹配率 <50%: {len(ok_df[ok_df['Match_Rate']<50])} ({len(ok_df[ok_df['Match_Rate']<50])/len(ok_df)*100:.1f}%)")
    
    # 显示匹配率低的
    print(f"\n=== 匹配率 <50% 的合约 ===")
    low_match = ok_df[ok_df['Match_Rate'] < 50].sort_values('Match_Rate')
    print(low_match.to_string(index=False))
    
    # 显示 ES 详情
    print(f"\n=== ES 合约详情 ===")
    es_result = result_df[result_df['Symbol'] == 'ES'].iloc[0]
    print(f"ASC 换月数：{es_result['ASC_Count']}")
    print(f"计算换月数：{es_result['Calc_Count']}")
    print(f"匹配数：{es_result['Match_Count']}")
    print(f"匹配率：{es_result['Match_Rate']:.1f}%")


if __name__ == '__main__':
    main()
