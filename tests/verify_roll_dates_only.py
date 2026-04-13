"""
验证：只用理论规则能否准确预测 roll date？

方法：
1. 用 roll_rules_corrected.json 计算理论 roll dates
2. 与 ASC 提取的实际 roll dates 对比
3. 验证匹配率

如果匹配率高 → 测试期可以只用 NON + 理论规则生成 RAD_v2
"""

import pandas as pd
import calendar
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'CLC'
CONFIG_DIR = PROJECT_ROOT / 'config'
CONFIG_FILE = CONFIG_DIR / 'roll_rules_corrected.json'

with open(CONFIG_FILE) as f:
    roll_rules = json.load(f)

# D 类合约
D_CONTRACTS = ['NR', 'SB', 'SN', 'SP', 'TY', 'US', 'ZA', 'ZC', 'ZF', 'ZG', 'ZH', 'ZI', 'ZK', 'ZL', 'ZN', 'ZO', 'ZP', 'ZR', 'ZT', 'ZU', 'ZW', 'ZZ']

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
    return get_nth_weekday(year, month, 2, 4)


def get_roll_dates_theoretical(rule_type, day, contract_months_str, start_year, end_year):
    """理论计算的 roll dates"""
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
    """验证单个合约"""
    rollover_file = DATA_DIR / f'{symbol}_rollovers.csv'
    
    if not rollover_file.exists():
        return {'Symbol': symbol, 'Status': 'NO_ROLLOVER_FILE'}
    
    # 读取 ASC 提取的实际换月日期
    df = pd.read_csv(rollover_file)
    df['RollDate'] = pd.to_datetime(df['RollDate'])
    actual_dates = sorted(df['RollDate'].tolist())
    
    # 获取 rule
    rule = symbol_to_rule.get(symbol)
    if not rule:
        return {'Symbol': symbol, 'Status': 'NO_RULE', 'Rule_Type': None}
    
    # 计算理论 roll dates
    start_year = actual_dates[0].year - 1
    end_year = actual_dates[-1].year + 1
    theoretical_dates = get_roll_dates_theoretical(
        rule['rule_type'], rule['day'], rule['contract_months'],
        start_year, end_year
    )
    
    # 匹配
    actual_set = set(actual_dates)
    theoretical_set = set(theoretical_dates)
    
    match_count = len(actual_set & theoretical_set)
    actual_only = actual_set - theoretical_set
    theoretical_only = theoretical_set - actual_set
    
    match_rate = match_count / len(actual_dates) * 100 if len(actual_dates) > 0 else 0
    
    return {
        'Symbol': symbol,
        'Status': 'OK',
        'Rule_Type': rule['rule_type'],
        'Actual_Count': len(actual_dates),
        'Theoretical_Count': len(theoretical_dates),
        'Match_Count': match_count,
        'Match_Rate': match_rate,
        'Missing_Count': len(actual_only),
        'Extra_Count': len(theoretical_only),
    }


def main():
    print("="*100)
    print("验证：理论规则能否准确预测 roll dates？")
    print("="*100)
    
    results = []
    
    for symbol in D_CONTRACTS:
        result = verify_symbol(symbol)
        results.append(result)
        
        if result['Status'] == 'OK':
            status_str = f"✅ {result['Match_Count']}/{result['Actual_Count']} ({result['Match_Rate']:.1f}%)"
            if result['Match_Rate'] < 90:
                status_str = f"⚠️ {result['Match_Count']}/{result['Actual_Count']} ({result['Match_Rate']:.1f}%)"
            if result['Match_Rate'] < 50:
                status_str = f"❌ {result['Match_Count']}/{result['Actual_Count']} ({result['Match_Rate']:.1f}%)"
            print(f"{symbol}: {status_str} (rule={result['Rule_Type']})")
        else:
            print(f"{symbol}: {result['Status']}")
    
    # 保存
    df = pd.DataFrame(results)
    output_file = PROJECT_ROOT / 'tests' / 'results' / 'roll_dates_verification.csv'
    df.to_csv(output_file, index=False)
    print(f"\n详细结果：{output_file}")
    
    # 统计
    print("\n" + "="*100)
    print("统计摘要")
    print("="*100)
    
    ok_df = df[df['Status'] == 'OK']
    
    if len(ok_df) > 0:
        print(f"验证合约数：{len(ok_df)}")
        print(f"匹配率 ≥90%: {len(ok_df[ok_df['Match_Rate']>=90])} ({len(ok_df[ok_df['Match_Rate']>=90])/len(ok_df)*100:.1f}%)")
        print(f"匹配率 50-90%: {len(ok_df[(ok_df['Match_Rate']>=50) & (ok_df['Match_Rate']<90)])}")
        print(f"匹配率 <50%: {len(ok_df[ok_df['Match_Rate']<50])}")
        
        # 显示匹配率低的
        low_match = ok_df[ok_df['Match_Rate'] < 90].sort_values('Match_Rate')
        if len(low_match) > 0:
            print(f"\n=== 匹配率 <90% 的合约 ===")
            print(low_match[['Symbol', 'Rule_Type', 'Match_Count', 'Actual_Count', 'Match_Rate', 'Missing_Count']].to_string(index=False))
    
    # 显示 SP 的详情（corr=0.9994）
    print(f"\n=== SP 合约详情（高相关性示例）===")
    sp_result = df[df['Symbol'] == 'SP'].iloc[0]
    print(f"规则：{sp_result['Rule_Type']}")
    print(f"实际换月：{sp_result['Actual_Count']}")
    print(f"理论换月：{sp_result['Theoretical_Count']}")
    print(f"匹配：{sp_result['Match_Count']} ({sp_result['Match_Rate']:.1f}%)")


if __name__ == '__main__':
    main()
