"""
用交易日日历为 ZH, ZU, US 生成 RAD_v2

方法论（倒推法）：
1. 理论 roll date 可能是周末/假期
2. 实际调整日 = 理论日期后的第一个交易日
3. 检查那天的价格变化，计算 ratio

关键：不是"窗口内找最大跳空"，而是"理论日期 + 下一个交易日"
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

CONTRACTS = {
    'ZH': 'HO',
    'ZU': 'CL',
    'US': 'US',
}

with open(CONFIG_FILE) as f:
    rules = json.load(f)

def get_rule_for_symbol(ref_symbol):
    for rule_name, rule_data in rules.items():
        if ref_symbol in rule_data.get('symbols', []):
            rule_type, day = rule_name.split('_')
            return rule_type, int(day)
    return None, None

def get_contract_months(ref_symbol):
    months_file = PROJECT_ROOT / 'config' / 'contract_months.json'
    if months_file.exists():
        with open(months_file) as f:
            months = json.load(f)
        return months.get(ref_symbol, months.get('HO', 'H,M,U,Z'))
    return 'H,M,U,Z'

def get_roll_dates(rule_type, day, months_str, start_year, end_year):
    months_map = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
                  'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
    
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
    
    return sorted(roll_dates)

def find_next_trading_day(non_df, theoretical_date, max_days=10):
    """
    找理论日期后的第一个交易日
    
    方法：从理论日期开始，找 NON 数据中 >= 理论日期的第一个日期
    """
    future = non_df[non_df['Date'] >= theoretical_date]
    
    if len(future) == 0:
        return None
    
    # 第一个 >= 理论日期的交易日
    first_trading = future.iloc[0]
    
    # 检查是否在合理范围内（最多往后找 max_days）
    if (first_trading['Date'] - theoretical_date).days > max_days:
        return None
    
    return first_trading

def generate_rad_v2(symbol, ref_symbol):
    """生成 RAD_v2"""
    non_file = DATA_DIR / f'{symbol}_NON.CSV'
    
    if not non_file.exists():
        return None
    
    rule_type, day = get_rule_for_symbol(ref_symbol)
    months = get_contract_months(ref_symbol)
    
    if not rule_type:
        return {'Symbol': symbol, 'Status': 'NO_RULE'}
    
    # 读取 NON
    non = pd.read_csv(non_file, names=['Date','O','H','L','C','V','OI'])
    non['Date'] = pd.to_datetime(non['Date'])
    non['C'] = pd.to_numeric(non['C'])
    non = non.sort_values('Date').reset_index(drop=True)
    
    # 计算理论 roll dates
    roll_dates = get_roll_dates(rule_type, day, months, 2010, 2020)
    
    # 找实际调整日（理论日期后的第一个交易日）
    actual_adjustments = []
    for rd in roll_dates:
        first_trading = find_next_trading_day(non, rd)
        
        if first_trading is None:
            continue
        
        actual_date = first_trading['Date']
        actual_idx = int(first_trading.name)
        
        # 获取调整前后的价格
        if actual_idx > 0:
            prev_close = non.iloc[actual_idx - 1]['C']
            curr_close = non.iloc[actual_idx]['C']
            change = (curr_close - prev_close) / prev_close if prev_close > 0 else 0
        else:
            prev_close = None
            curr_close = curr_close
            change = None
        
        actual_adjustments.append({
            'theoretical': rd,
            'actual': actual_date,
            'idx': actual_idx,
            'prev_close': prev_close,
            'curr_close': curr_close,
            'change': change,
        })
    
    # 统计理论 vs 实际差异
    diffs = [(a['actual'] - a['theoretical']).days for a in actual_adjustments]
    
    # 找到测试期开始索引
    test_start_mask = non['Date'] >= TEST_START
    if not test_start_mask.any():
        return None
    test_start_idx = int(test_start_mask.idxmax())
    
    # 过滤测试期内的调整
    test_adjustments = [a for a in actual_adjustments if a['idx'] >= test_start_idx]
    
    if len(test_adjustments) == 0:
        return {'Symbol': symbol, 'Status': 'NO_ADJUSTMENTS_IN_TEST'}
    
    # 生成 ratio（按实际调整日排序）
    test_adjustments = sorted(test_adjustments, key=lambda x: x['idx'])
    
    non['ratio'] = 1.0
    current_ratio = 1.0
    
    for i, adj in enumerate(test_adjustments):
        idx = adj['idx']
        c = adj['prev_close']
        C = adj['curr_close']
        
        if c is not None and c > 0 and C > 0:
            ratio_change = c / C
            current_ratio *= ratio_change
            
            next_idx = test_adjustments[i + 1]['idx'] if i + 1 < len(test_adjustments) else len(non)
            non.loc[idx:next_idx - 1, 'ratio'] = current_ratio
    
    # 生成 RAD_v2
    non['rad_v2'] = non['C'] * non['ratio']
    
    # 保存测试期数据
    test = non[(non['Date'] >= TEST_START) & (non['Date'] <= TEST_END)].copy()
    
    output_file = DATA_DIR / f'{symbol}_RAD_v2.CSV'
    output_df = test[['Date', 'O', 'H', 'L', 'rad_v2', 'V', 'OI']].copy()
    output_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInterest']
    output_df['Date'] = output_df['Date'].dt.strftime('%m/%d/%Y')
    output_df.to_csv(output_file, index=False, header=False)
    
    # 统计匹配情况
    exact_match = sum(1 for d in diffs if d == 0)
    plus_1 = sum(1 for d in diffs if d == 1)
    plus_2 = sum(1 for d in diffs if d == 2)
    plus_3_or_more = sum(1 for d in diffs if d >= 3)
    
    return {
        'Symbol': symbol,
        'Status': 'OK',
        'Rule': f'{rule_type}_{day}',
        'Theoretical_Rolls': len(roll_dates),
        'Detected_Adjustments': len(actual_adjustments),
        'Test_Adjustments': len(test_adjustments),
        'Test_Days': len(test),
        'NON_Price_Range': f"{test['C'].min():.2f} - {test['C'].max():.2f}",
        'RAD_v2_Price_Range': f"{test['rad_v2'].min():.2f} - {test['rad_v2'].max():.2f}",
        'Ratio_Range': f"{test['ratio'].min():.4f} - {test['ratio'].max():.4f}",
        'Exact_Match': exact_match,
        'Plus_1': plus_1,
        'Plus_2': plus_2,
        'Plus_3_or_more': plus_3_or_more,
        'Avg_Days_Diff': f"{sum(diffs)/len(diffs):.1f}",
        'Days_Diff_Range': f"{min(diffs)} to {max(diffs)}",
        'Output': str(output_file),
    }

def main():
    print("="*100)
    print("用交易日日历法为 ZH, ZU, US 生成 RAD_v2")
    print("="*100)
    
    results = []
    
    for symbol, ref_symbol in CONTRACTS.items():
        print(f"\n{symbol} (参考：{ref_symbol}):")
        
        result = generate_rad_v2(symbol, ref_symbol)
        if result:
            results.append(result)
            
            if result['Status'] == 'OK':
                print(f"  规则：{result['Rule']}")
                print(f"  理论 roll dates: {result['Theoretical_Rolls']}")
                print(f"  检测到调整：{result['Detected_Adjustments']}")
                print(f"  测试期调整：{result['Test_Adjustments']}")
                print(f"  测试期天数：{result['Test_Days']}")
                print(f"  NON 价格范围：{result['NON_Price_Range']}")
                print(f"  RAD_v2 价格范围：{result['RAD_v2_Price_Range']}")
                print(f"  Ratio 范围：{result['Ratio_Range']}")
                print()
                print(f"  匹配分布:")
                print(f"    精确匹配 (0 天): {result['Exact_Match']}")
                print(f"    +1 天：{result['Plus_1']}")
                print(f"    +2 天：{result['Plus_2']}")
                print(f"    +3 天或更多：{result['Plus_3_or_more']}")
                print(f"  平均差异：{result['Avg_Days_Diff']} 天 ({result['Days_Diff_Range']})")
                print(f"  输出：{result['Output']}")
            else:
                print(f"  状态：{result['Status']}")
    
    # 保存汇总
    if results:
        df = pd.DataFrame(results)
        output_file = PROJECT_ROOT / 'tests' / 'results' / 'rad_v2_trading_day_method.csv'
        df.to_csv(output_file, index=False)
        print(f"\n汇总：{output_file}")
    
    print(f"\n{'='*100}")
    print("生成完成！")
    print(f"{'='*100}")

if __name__ == '__main__':
    main()
