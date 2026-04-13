"""
用 Roll date + N 方法为 ZH, ZU, US 生成 RAD_v2

方法论：
1. 计算理论 roll dates（从 roll_rules_corrected.json）
2. 在理论 roll date 前后窗口（-2 到 +5 天）内找价格跳空最大的交易日
3. 用实际调整日计算 ratio
4. 生成 RAD_v2

关键：理论规则给出大致时间，实际调整日由数据决定
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

def find_actual_adjustment_day(non_df, theoretical_roll, window_before=2, window_after=5):
    """
    在理论 roll date 前后窗口内找实际调整日
    
    方法：找价格变化绝对值最大的那天
    """
    # 定义窗口
    start = theoretical_roll - pd.Timedelta(days=window_before)
    end = theoretical_roll + pd.Timedelta(days=window_after)
    
    # 找窗口内的交易日
    window = non_df[(non_df['Date'] >= start) & (non_df['Date'] <= end)].copy()
    
    if len(window) < 2:
        return None, None, None
    
    # 计算价格变化
    window = window.sort_values('Date').reset_index(drop=True)
    window['ret'] = window['C'].pct_change()
    
    # 找变化最大的那天（从第二天开始，因为第一天没有 pct_change）
    if len(window) < 2:
        return None, None, None
    
    max_idx = window['ret'].abs().idxmax()
    max_ret = window.loc[max_idx, 'ret']
    
    # 如果最大变化太小（<0.5%），说明没有显著调整
    if abs(max_ret) < 0.005:
        return None, None, None
    
    actual_date = window.loc[max_idx, 'Date']
    actual_change = window.loc[max_idx, 'ret']
    
    # 获取调整前后的价格
    actual_idx = window.loc[max_idx].name
    if actual_idx > 0:
        prev_close = window.iloc[actual_idx - 1]['C']
        curr_close = window.iloc[actual_idx]['C']
    else:
        prev_close = None
        curr_close = window.iloc[actual_idx]['C']
    
    return actual_date, actual_change, (prev_close, curr_close)

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
    
    # 找实际调整日
    actual_adjustments = []
    for rd in roll_dates:
        actual_date, change, prices = find_actual_adjustment_day(non, rd)
        
        if actual_date is not None and prices[0] is not None:
            actual_idx = non[non['Date'] == actual_date].index
            if len(actual_idx) > 0:
                actual_adjustments.append({
                    'theoretical': rd,
                    'actual': actual_date,
                    'change': change,
                    'idx': int(actual_idx[0]),
                    'prev_close': prices[0],
                    'curr_close': prices[1],
                })
    
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
        
        if c > 0 and C > 0:
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
    
    # 统计理论 vs 实际差异
    diffs = [(a['actual'] - a['theoretical']).days for a in test_adjustments]
    
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
        'Avg_Days_Diff': f"{sum(diffs)/len(diffs):.1f}",
        'Days_Diff_Range': f"{min(diffs)} to {max(diffs)}",
        'Output': str(output_file),
    }

def main():
    print("="*100)
    print("用 Roll date + N 方法为 ZH, ZU, US 生成 RAD_v2")
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
                print(f"  理论 vs 实际差异：平均 {result['Avg_Days_Diff']} 天 ({result['Days_Diff_Range']})")
                print(f"  输出：{result['Output']}")
            else:
                print(f"  状态：{result['Status']}")
    
    # 保存汇总
    if results:
        df = pd.DataFrame(results)
        output_file = PROJECT_ROOT / 'tests' / 'results' / 'rad_v2_roll_date_plus_n.csv'
        df.to_csv(output_file, index=False)
        print(f"\n汇总：{output_file}")
    
    print(f"\n{'='*100}")
    print("生成完成！")
    print(f"{'='*100}")

if __name__ == '__main__':
    main()
