"""
为 ZH, ZU, US 生成 RAD_v2

方法：
- ZH → 参考 HO (Heating Oil) 的换月规则：MPDM_11 (交割月前一个月第 11 天)
- ZU → 参考 CL (Crude Oil) 的换月规则：MPDM_11
- US → 使用已有规则：MPDM_25 (交割月前一个月第 25 天)

用 NON 数据 + 理论换月日期生成 back-adjusted RAD_v2
"""

import pandas as pd
import calendar
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'CLC'
TEST_START = pd.Timestamp('2011-01-01')
TEST_END = pd.Timestamp('2019-12-31')

# 合约配置：symbol -> (reference_symbol, rule_type, day, contract_months)
CONTRACTS = {
    'ZH': {'ref': 'HO', 'rule': 'MPDM', 'day': 11, 'months': 'H,M,U,Z'},  # Heating Oil
    'ZU': {'ref': 'CL', 'rule': 'MPDM', 'day': 11, 'months': 'F,G,J,K,M,N,Q,U,V,X,Z'},  # Crude Oil
    'US': {'ref': 'US', 'rule': 'MPDM', 'day': 25, 'months': 'H,M,U,Z'},  # T-Bonds
}


def get_nth_weekday(year, month, n, weekday):
    """获取某月第 n 个星期几"""
    first_day = calendar.weekday(year, month, 1)
    if first_day <= weekday:
        first = weekday - first_day + 1
    else:
        first = 7 - first_day + weekday + 1
    return first + (n - 1) * 7


def get_mpdm_dates(year, month, day):
    """获取交割月前一个月第 N 天"""
    # 前一个月
    if month == 1:
        pm_year = year - 1
        pm_month = 12
    else:
        pm_year = year
        pm_month = month - 1
    
    # 如果 day > 前一个月的天数，用最后一天
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


def generate_rad_v2(symbol, config):
    """用 NON + 理论换月日期生成 RAD_v2"""
    non_file = DATA_DIR / f'{symbol}_NON.CSV'
    
    if not non_file.exists():
        return None
    
    # 读取 NON
    non = pd.read_csv(non_file, names=['Date','O','H','L','C','V','OI'])
    non['Date'] = pd.to_datetime(non['Date'])
    non['C'] = pd.to_numeric(non['C'])
    non = non.sort_values('Date').reset_index(drop=True)
    
    # 计算理论换月日期
    roll_dates = get_roll_dates(
        config['rule'], config['day'], config['months'],
        2010, 2020
    )
    
    # 找到测试期内的换月
    test_rolls = [(i, rd) for i, rd in enumerate(roll_dates) 
                  if TEST_START <= rd <= TEST_END]
    
    if len(test_rolls) == 0:
        print(f"  {symbol}: 测试期无换月日期")
        return None
    
    # 生成 back-adjusted ratio
    # 从最后一个换月开始向前调整
    non['ratio'] = 1.0
    
    # 简化：假设每个换月日 ratio 变化为 1（即不调整）
    # 因为没有 ASC 数据验证，这是最保守的做法
    non['rad_v2'] = non['C']
    
    # 过滤测试期
    test = non[(non['Date'] >= TEST_START) & (non['Date'] <= TEST_END)]
    
    return {
        'Symbol': symbol,
        'Roll_Dates': len(test_rolls),
        'Test_Days': len(test),
        'Price_Range': f"{test['C'].min():.2f} - {test['C'].max():.2f}",
        'Method': 'NON (no adjustment, unverified)',
    }


def main():
    print("="*80)
    print("为 ZH, ZU, US 生成 RAD_v2")
    print("="*80)
    
    results = []
    
    for symbol, config in CONTRACTS.items():
        print(f"\n{symbol} (参考 {config['ref']}, 规则={config['rule']}_{config['day']}, 月份={config['months']}):")
        
        result = generate_rad_v2(symbol, config)
        if result:
            results.append(result)
            print(f"  换月数：{result['Roll_Dates']}")
            print(f"  测试期天数：{result['Test_Days']}")
            print(f"  价格范围：{result['Price_Range']}")
            print(f"  方法：{result['Method']}")
    
    # 保存
    if results:
        df = pd.DataFrame(results)
        output_file = PROJECT_ROOT / 'tests' / 'results' / 'rad_v2_zh_zu_us.csv'
        df.to_csv(output_file, index=False)
        print(f"\n详细结果：{output_file}")
    
    print(f"\n{'='*80}")
    print("建议")
    print(f"{'='*80}")
    print("""
由于这 3 个合约没有 ASC 数据验证换月规则，有两种选择：

1. **保守方案**：排除 ZH, ZU, US，使用 47/50 合约 (94% 覆盖率)
   - 优点：数据质量有保证
   - 缺点：覆盖率略降

2. **激进方案**：用 NON 直接作为 RAD_v2（不做换月调整）
   - 优点：100% 覆盖率
   - 缺点：换月调整未验证，可能引入偏差

推荐：方案 1（保守方案）
    """)


if __name__ == '__main__':
    main()
