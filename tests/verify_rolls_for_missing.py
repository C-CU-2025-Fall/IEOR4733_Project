"""
验证 ZH, ZU, US 的理论 roll dates 是否在 NON 数据中有对应的价格调整

步骤：
1. 从 roll_rules_corrected.json 获取规则
2. 计算测试期 (2011-2019) 的理论 roll dates
3. 检查 NON 在这些日期前后的价格变化
4. 如果有调整，计算 ratio；如果没有，说明 NON 已经是连续合约
"""

import pandas as pd
import calendar
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'CLC'
CONFIG_FILE = PROJECT_ROOT / 'config' / 'roll_rules_corrected.json'
TEST_START = pd.Timestamp('2011-01-01')
TEST_END = pd.Timestamp('2019-12-31')

# 合约映射：symbol -> reference_symbol -> rule
CONTRACTS = {
    'ZH': 'HO',  # Heating Oil
    'ZU': 'CL',  # Crude Oil
    'US': 'US',  # T-Bonds
}

# 从 config 加载规则
with open(CONFIG_FILE) as f:
    rules = json.load(f)

def get_rule_for_symbol(ref_symbol):
    """根据 reference symbol 找到规则"""
    for rule_name, rule_data in rules.items():
        if ref_symbol in rule_data.get('symbols', []):
            rule_type, day = rule_name.split('_')
            return rule_type, int(day)
    return None, None

def get_contract_months(ref_symbol):
    """获取合约月份"""
    months_file = PROJECT_ROOT / 'config' / 'contract_months.json'
    if months_file.exists():
        with open(months_file) as f:
            months = json.load(f)
        return months.get(ref_symbol, months.get('HO', 'H,M,U,Z'))
    return 'H,M,U,Z'

def get_nth_weekday(year, month, n, weekday):
    first_day = calendar.weekday(year, month, 1)
    if first_day <= weekday:
        first = weekday - first_day + 1
    else:
        first = 7 - first_day + weekday + 1
    return first + (n - 1) * 7

def get_roll_dates(rule_type, day, months_str, start_year, end_year):
    months_map = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
                  'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
    
    delivery_months = [months_map[m.strip()] for m in months_str.split(',') 
                       if m.strip() in months_map]
    
    roll_dates = []
    for year in range(start_year, end_year + 1):
        for dm in delivery_months:
            if rule_type == 'MPDM':
                # 前一个月第 N 天
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

def check_non_adjustment(symbol, ref_symbol, roll_dates):
    """检查 NON 数据在 roll dates 前后的价格调整"""
    non_file = DATA_DIR / f'{symbol}_NON.CSV'
    
    if not non_file.exists():
        return None
    
    non = pd.read_csv(non_file, names=['Date','O','H','L','C','V','OI'])
    non['Date'] = pd.to_datetime(non['Date'])
    non['C'] = pd.to_numeric(non['C'])
    non = non.sort_values('Date').reset_index(drop=True)
    
    # 过滤测试期
    test = non[(non['Date'] >= TEST_START) & (non['Date'] <= TEST_END)].copy()
    test = test.reset_index(drop=True)
    
    if len(test) == 0:
        return None
    
    # 匹配 roll dates 到交易日
    adjustments = []
    for rd in roll_dates:
        if not (TEST_START <= rd <= TEST_END):
            continue
        
        # 找 rd 当天或前后 5 天的交易日
        window = test[(test['Date'] >= rd - pd.Timedelta(days=5)) & 
                      (test['Date'] <= rd + pd.Timedelta(days=5))]
        
        if len(window) == 0:
            continue
        
        # 找最接近 rd 的日期
        closest = window.loc[(window['Date'] - rd).abs().idxmin()]
        idx = closest.name
        
        # 检查前后价格变化
        if idx > 0 and idx < len(test) - 1:
            prev_c = test.iloc[idx - 1]['C']
            curr_c = test.iloc[idx]['C']
            next_c = test.iloc[idx + 1]['C']
            
            # 计算价格变化
            change_at = (curr_c - prev_c) / prev_c if prev_c > 0 else 0
            change_after = (next_c - curr_c) / curr_c if curr_c > 0 else 0
            
            # 检查是否有显著调整 (>1%)
            is_adjustment = abs(change_at) > 0.01
            
            adjustments.append({
                'RollDate': rd,
                'TradeDate': closest['Date'],
                'Idx': idx,
                'Prev_Close': prev_c,
                'Curr_Close': curr_c,
                'Change_At': change_at,
                'Change_After': change_after,
                'Is_Adjustment': is_adjustment,
                'Ratio': prev_c / curr_c if curr_c > 0 else 1.0,
            })
    
    return adjustments, test

def main():
    print("="*100)
    print("验证 ZH, ZU, US 的理论 roll dates 与 NON 价格调整")
    print("="*100)
    
    for symbol, ref_symbol in CONTRACTS.items():
        rule_type, day = get_rule_for_symbol(ref_symbol)
        months = get_contract_months(ref_symbol)
        
        print(f"\n{'='*100}")
        print(f"{symbol} (参考: {ref_symbol}, 规则: {rule_type}_{day}, 月份: {months})")
        print(f"{'='*100}")
        
        if not rule_type:
            print(f"  ❌ 未找到 {ref_symbol} 的规则")
            continue
        
        # 计算理论 roll dates
        roll_dates = get_roll_dates(rule_type, day, months, 2011, 2019)
        print(f"  理论 roll dates (2011-2019): {len(roll_dates)} 个")
        
        # 检查 NON 调整
        result = check_non_adjustment(symbol, ref_symbol, roll_dates)
        
        if result is None:
            print(f"  ❌ 无 NON 数据")
            continue
        
        adjustments, test = result
        
        # 统计
        has_adjustment = [a for a in adjustments if a['Is_Adjustment']]
        
        print(f"  测试期天数：{len(test)}")
        print(f"  检查的 roll dates: {len(adjustments)} 个")
        print(f"  有显著调整 (>1%): {len(has_adjustment)} 个 ({len(has_adjustment)/len(adjustments)*100:.1f}%)")
        
        # 显示前 5 个 roll dates 的详情
        print(f"\n  前 5 个 roll dates 详情:")
        for i, adj in enumerate(adjustments[:5]):
            status = "✅" if adj['Is_Adjustment'] else "❌"
            print(f"    {i+1}. {adj['RollDate'].date()} → {adj['TradeDate'].date()}: "
                  f"{adj['Prev_Close']:.2f} → {adj['Curr_Close']:.2f} "
                  f"({adj['Change_At']:>7.2%}) {status}")
        
        # 如果没有调整，说明 NON 已经是连续合约
        if len(has_adjustment) == 0:
            print(f"\n  ⚠️  NON 数据在 roll dates 没有显著调整 → NON 可能已经是连续合约")
            print(f"  建议：直接用 NON 作为 RAD_v2（不做调整）")
        else:
            print(f"\n  ✅ NON 数据在 roll dates 有调整，可以生成 RAD_v2")


if __name__ == '__main__':
    main()
