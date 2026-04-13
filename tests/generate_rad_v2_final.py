"""
为 ZH, ZU, US 生成 RAD_v2

方法：
1. 从 roll_rules_corrected.json 获取理论规则
2. 计算测试期 (2011-2019) 的理论 roll dates
3. 用 NON 数据在 roll dates 的价格变化计算 ratio
4. 生成 RAD_v2 = NON × cumulative_ratio
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
    
    # 匹配到交易日
    matched_rolls = []
    for rd in roll_dates:
        matches = non[non['Date'] == rd]
        if len(matches) > 0:
            matched_rolls.append(int(matches.index[0]))
        else:
            # 找前后 5 天内最近的
            window = non[(non['Date'] >= rd - pd.Timedelta(days=5)) & 
                         (non['Date'] <= rd + pd.Timedelta(days=5))]
            if len(window) > 0:
                closest_idx = int((window['Date'] - rd).abs().idxmin())
                matched_rolls.append(closest_idx)
    
    # 生成 ratio
    non['ratio'] = 1.0
    current_ratio = 1.0
    
    # 找到测试期开始的索引
    test_start_mask = non['Date'] >= TEST_START
    if not test_start_mask.any():
        return None
    test_start_idx = int(test_start_mask.idxmax())
    
    # 只应用测试期内的换月
    test_rolls = [int(idx) for idx in matched_rolls if int(idx) >= test_start_idx]
    
    for i, roll_idx in enumerate(test_rolls):
        if roll_idx > 0 and roll_idx < len(non):
            c = non.iloc[roll_idx - 1]['C']  # 前一天 close
            C = non.iloc[roll_idx]['C']      # roll day close
            
            if c > 0 and C > 0:
                ratio_change = c / C
                current_ratio *= ratio_change
                
                next_roll = test_rolls[i + 1] if i + 1 < len(test_rolls) else len(non)
                non.loc[roll_idx:next_roll - 1, 'ratio'] = current_ratio
    
    # 生成 RAD_v2
    non['rad_v2'] = non['C'] * non['ratio']
    
    # 保存测试期数据
    test = non[(non['Date'] >= TEST_START) & (non['Date'] <= TEST_END)].copy()
    
    output_file = DATA_DIR / f'{symbol}_RAD_v2.CSV'
    output_df = test[['Date', 'O', 'H', 'L', 'rad_v2', 'V', 'OI']].copy()
    output_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInterest']
    output_df['Date'] = output_df['Date'].dt.strftime('%m/%d/%Y')
    output_df.to_csv(output_file, index=False, header=False)
    
    return {
        'Symbol': symbol,
        'Status': 'OK',
        'Rule': f'{rule_type}_{day}',
        'Total_Rolls': len(matched_rolls),
        'Test_Rolls': len(test_rolls),
        'Test_Days': len(test),
        'NON_Price_Range': f"{test['C'].min():.2f} - {test['C'].max():.2f}",
        'RAD_v2_Price_Range': f"{test['rad_v2'].min():.2f} - {test['rad_v2'].max():.2f}",
        'Ratio_Range': f"{test['ratio'].min():.4f} - {test['ratio'].max():.4f}",
        'Output': str(output_file),
    }

def main():
    print("="*100)
    print("为 ZH, ZU, US 生成 RAD_v2")
    print("="*100)
    
    results = []
    
    for symbol, ref_symbol in CONTRACTS.items():
        print(f"\n{symbol} (参考: {ref_symbol}):")
        
        result = generate_rad_v2(symbol, ref_symbol)
        if result:
            results.append(result)
            
            if result['Status'] == 'OK':
                print(f"  规则：{result['Rule']}")
                print(f"  总换月数：{result['Total_Rolls']}")
                print(f"  测试期换月数：{result['Test_Rolls']}")
                print(f"  测试期天数：{result['Test_Days']}")
                print(f"  NON 价格范围：{result['NON_Price_Range']}")
                print(f"  RAD_v2 价格范围：{result['RAD_v2_Price_Range']}")
                print(f"  Ratio 范围：{result['Ratio_Range']}")
                print(f"  输出：{result['Output']}")
            else:
                print(f"  状态：{result['Status']}")
    
    # 保存汇总
    if results:
        df = pd.DataFrame(results)
        output_file = PROJECT_ROOT / 'tests' / 'results' / 'rad_v2_generation_summary.csv'
        df.to_csv(output_file, index=False)
        print(f"\n汇总：{output_file}")
    
    print(f"\n{'='*100}")
    print("生成完成！可以用这些 RAD_v2 文件进行回测")
    print(f"{'='*100}")

if __name__ == '__main__':
    main()
