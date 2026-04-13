"""
用 REV 方法为 ZH, ZU, US 生成 RAD_v2

方法：
1. 从 NON 数据检测价格跳空（换月日）
2. 用跳空价格计算调整值（REV 是数值调整：adj = prev_Close - current_Close）
3. 转换成比值调整（ratio = prev_Close / current_Close）
4. 生成 RAD_v2 = NON × ratio

REV vs RAD:
- REV: REV_t = NON_t + cumulative_adj (加法)
- RAD: RAD_t = NON_t × cumulative_ratio (乘法)
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'CLC'
TEST_START = pd.Timestamp('2011-01-01')
TEST_END = pd.Timestamp('2019-12-31')

# 合约配置：symbol -> (reference_symbol, rule_type, day, contract_months)
CONTRACTS = {
    'ZH': {'ref': 'HO', 'rule': 'MPDM', 'day': 11, 'months': 'H,M,U,Z'},
    'ZU': {'ref': 'CL', 'rule': 'MPDM', 'day': 11, 'months': 'F,G,J,K,M,N,Q,U,V,X,Z'},
    'US': {'ref': 'US', 'rule': 'MPDM', 'day': 25, 'months': 'H,M,U,Z'},
}


def get_nth_weekday(year, month, n, weekday):
    """获取某月第 n 个星期几"""
    import calendar
    first_day = calendar.weekday(year, month, 1)
    if first_day <= weekday:
        first = weekday - first_day + 1
    else:
        first = 7 - first_day + weekday + 1
    return first + (n - 1) * 7


def get_mpdm_dates(year, month, day):
    """获取交割月前一个月第 N 天"""
    import calendar
    if month == 1:
        pm_year = year - 1
        pm_month = 12
    else:
        pm_year = year
        pm_month = month - 1
    
    max_day = calendar.monthrange(pm_year, pm_month)[1]
    actual_day = min(day, max_day)
    
    try:
        return pd.Timestamp(year=pm_year, month=pm_month, day=actual_day)
    except:
        return None


def get_roll_dates(rule, day, months_str, start_year, end_year):
    """计算理论换月日期"""
    months_map = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
                  'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
    
    delivery_months = [months_map[m.strip()] for m in months_str.split(',') 
                       if m.strip() in months_map]
    
    roll_dates = []
    for year in range(start_year, end_year + 1):
        for dm in delivery_months:
            if rule == 'MPDM':
                rd = get_mpdm_dates(year, dm, day)
                if rd:
                    roll_dates.append(rd)
    
    return sorted(roll_dates)


def detect_rolls_from_price_jumps(non_df, threshold=0.03, min_gap=15):
    """
    从价格跳空检测 roll dates
    
    假设：换月时价格会有显著跳空（>3%）
    min_gap: 两次换月之间至少间隔的天数
    """
    non = non_df.sort_values('Date').reset_index(drop=True)
    non['ret'] = non['C'].pct_change()
    
    # 检测大幅跳空
    jumps = non[non['ret'].abs() > threshold].copy()
    
    if len(jumps) == 0:
        return []
    
    # 简单过滤：间隔至少 min_gap 天
    rolls = []
    last_idx = -min_gap
    
    for idx, row in jumps.iterrows():
        if idx - last_idx >= min_gap:
            rolls.append(idx)
            last_idx = idx
    
    return rolls


def generate_rad_v2_with_rev_method(symbol, config):
    """
    用 REV 方法生成 RAD_v2
    
    步骤：
    1. 用理论规则计算 roll dates
    2. 用 NON 数据的价格跳空验证/调整 roll dates
    3. 计算比值调整因子
    4. 生成 RAD_v2
    """
    non_file = DATA_DIR / f'{symbol}_NON.CSV'
    
    if not non_file.exists():
        return None
    
    # 读取 NON
    non = pd.read_csv(non_file, names=['Date','O','H','L','C','V','OI'])
    non['Date'] = pd.to_datetime(non['Date'])
    non['C'] = pd.to_numeric(non['C'])
    non = non.sort_values('Date').reset_index(drop=True)
    
    # 计算理论 roll dates
    roll_dates = get_roll_dates(
        config['rule'], config['day'], config['months'],
        2010, 2020
    )
    
    # 匹配到交易日
    matched_rolls = []
    for rd in roll_dates:
        idx = non[non['Date'] == rd].index
        if len(idx) > 0:
            matched_rolls.append(idx[0])
        else:
            # 找最近的 (±5 天)
            diffs = (non['Date'] - rd).abs()
            min_diff = diffs.min()
            if min_diff.days <= 5:
                closest_idx = diffs.idxmin()
                matched_rolls.append(closest_idx)
    
    # 过滤测试期内的换月
    test_start_idx = non[non['Date'] >= TEST_START].index
    if len(test_start_idx) == 0:
        return None
    test_start_idx = test_start_idx[0]
    
    test_rolls = [idx for idx in matched_rolls if idx >= test_start_idx]
    
    if len(test_rolls) == 0:
        return {
            'Symbol': symbol,
            'Status': 'NO_ROLLS_IN_TEST',
            'Method': 'theoretical',
        }
    
    # 用价格跳空验证 roll dates
    detected_rolls = detect_rolls_from_price_jumps(non, threshold=0.03, min_gap=15)
    detected_test_rolls = [idx for idx in detected_rolls if idx >= test_start_idx]
    
    # 比较理论和检测的 roll dates
    match_count = len(set(test_rolls) & set(detected_test_rolls))
    
    # 生成 RAD_v2（用理论 roll dates）
    non['ratio'] = 1.0
    current_ratio = 1.0
    
    for i, roll_idx in enumerate(test_rolls):
        if roll_idx > 0 and roll_idx < len(non):
            # c = 前一天 close, C = roll day close
            c = non.iloc[roll_idx - 1]['C']
            C = non.iloc[roll_idx]['C']
            
            if c > 0 and C > 0:
                ratio_change = c / C
                current_ratio *= ratio_change
                
                next_roll = test_rolls[i + 1] if i + 1 < len(test_rolls) else len(non)
                non.loc[roll_idx:next_roll - 1, 'ratio'] = current_ratio
    
    non['rad_v2'] = non['C'] * non['ratio']
    
    # 过滤测试期
    test = non[(non['Date'] >= TEST_START) & (non['Date'] <= TEST_END)]
    
    return {
        'Symbol': symbol,
        'Status': 'OK',
        'Theoretical_Rolls': len(test_rolls),
        'Detected_Rolls': len(detected_test_rolls),
        'Matched_Rolls': match_count,
        'Test_Days': len(test),
        'Price_Range': f"{test['C'].min():.2f} - {test['C'].max():.2f}",
        'RAD_v2_Range': f"{test['rad_v2'].min():.2f} - {test['rad_v2'].max():.2f}",
        'Method': 'REV-inspired (theoretical rolls + ratio adjustment)',
    }


def main():
    print("="*80)
    print("用 REV 方法为 ZH, ZU, US 生成 RAD_v2")
    print("="*80)
    
    results = []
    
    for symbol, config in CONTRACTS.items():
        print(f"\n{symbol} (参考 {config['ref']}, 规则={config['rule']}_{config['day']}):")
        
        result = generate_rad_v2_with_rev_method(symbol, config)
        if result:
            results.append(result)
            
            if result['Status'] == 'OK':
                print(f"  理论换月：{result['Theoretical_Rolls']}")
                print(f"  检测换月：{result['Detected_Rolls']}")
                print(f"  匹配换月：{result['Matched_Rolls']}")
                print(f"  测试期天数：{result['Test_Days']}")
                print(f"  NON 价格范围：{result['Price_Range']}")
                print(f"  RAD_v2 范围：{result['RAD_v2_Range']}")
                print(f"  方法：{result['Method']}")
            else:
                print(f"  状态：{result['Status']}")
    
    # 保存
    if results:
        df = pd.DataFrame(results)
        output_file = PROJECT_ROOT / 'tests' / 'results' / 'rad_v2_rev_method.csv'
        df.to_csv(output_file, index=False)
        print(f"\n详细结果：{output_file}")
    
    # 详细分析 US（有完整理论规则）
    print(f"\n{'='*80}")
    print("US 合约详细分析")
    print(f"{'='*80}")
    
    us_result = generate_rad_v2_with_rev_method('US', CONTRACTS['US'])
    if us_result and us_result['Status'] == 'OK':
        non_file = DATA_DIR / 'US_NON.CSV'
        non = pd.read_csv(non_file, names=['Date','O','H','L','C','V','OI'])
        non['Date'] = pd.to_datetime(non['Date'])
        non['C'] = pd.to_numeric(non['C'])
        non = non.sort_values('Date').reset_index(drop=True)
        
        # 显示前 5 次换月的价格变化
        roll_dates = get_roll_dates('MPDM', 25, 'H,M,U,Z', 2010, 2012)
        print(f"\n前 5 次理论换月日期:")
        for rd in roll_dates[:5]:
            idx = non[non['Date'] == rd].index
            if len(idx) > 0:
                i = idx[0]
                if i > 0:
                    prev_c = non.iloc[i-1]['C']
                    curr_c = non.iloc[i]['C']
                    ratio = prev_c / curr_c if curr_c > 0 else 0
                    print(f"  {rd.date()}: prev={prev_c:.2f}, curr={curr_c:.2f}, ratio={ratio:.4f}")


if __name__ == '__main__':
    main()
