"""
50 个合约数据丰富度完整分析

分析维度：
1. 测试期 (2011-2019) 各数据源行数
2. 2011 年前各数据源行数（用于验证方法）
3. 数据补齐可行性分类

分类标准：
- A: 完整数据 (RAD+ASC+rollover 测试期全) → 可直接用
- B: RAD+rollover 测试期全 → 可用 NON 验证
- C: RAD+ASC 测试期全 (无 rollover) → 需提取换月
- D: 仅 RAD 测试期 → 无法验证调整来源
- E: ASC+rollover 测试期全 (RAD 缺失) → 可生成 RAD_v2
- F: ASC+rollover 只到 2011 年前 → ❌ 无法生成测试期数据
- G: 仅 NON 测试期 → ❌ 无法生成 RAD_v2
- H: 数据不全 → ❌ 需手动检查
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'CLC'
TEMP_DIR = PROJECT_ROOT / 'config' / 'TEMP'

TEST_START = pd.Timestamp('2011-01-01')
TEST_END = pd.Timestamp('2019-12-31')

# 50 个论文合约
PAPER_CONTRACTS = [
    'CC', 'DA', 'GI', 'JO', 'KC', 'KW', 'LB', 'NR', 'SB',
    'ZA', 'ZC', 'ZF', 'ZG', 'ZH', 'ZI', 'ZK', 'ZL',
    'ZO', 'ZP', 'ZR', 'ZT', 'ZU', 'ZW', 'ZZ',
    'CA', 'EN', 'ER', 'ES', 'LX', 'MD', 'SC', 'SP', 'XU', 'XX', 'YM',
    'DT', 'FB', 'TY', 'UB', 'US', 'ZN',
    'AN', 'BN', 'CN', 'DX', 'FN', 'JN', 'MP', 'NK', 'SN',
]


def check_file(file_path, is_rollover=False):
    """检查文件并返回详细统计"""
    if not file_path.exists():
        return None
    
    try:
        if file_path.suffix.lower() == '.asc':
            rows = []
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 8 and len(parts[0]) == 8 and parts[0].isdigit():
                        year = int(parts[0][:4])
                        if 1900 <= year <= 2099:
                            rows.append(parts[:8])
            
            if not rows:
                return None
            
            df = pd.DataFrame(rows, columns=['Date','O','H','L','C','V','OI','_'])
            df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
            
        elif file_path.suffix.lower() == '.csv':
            if is_rollover:
                df = pd.read_csv(file_path)
                df['Date'] = pd.to_datetime(df['RollDate'])
            else:
                df = pd.read_csv(file_path, names=['Date','O','H','L','C','V','OI'])
                df['Date'] = pd.to_datetime(df['Date'])
        else:
            return None
        
        rows_total = len(df)
        rows_pre2011 = len(df[df['Date'] < TEST_START])
        rows_in_test = len(df[(df['Date'] >= TEST_START) & (df['Date'] <= TEST_END)])
        rows_post2019 = len(df[df['Date'] > TEST_END])
        
        return {
            'exists': True,
            'rows_total': rows_total,
            'rows_pre2011': rows_pre2011,
            'rows_in_test': rows_in_test,
            'rows_post2019': rows_post2019,
            'date_min': df['Date'].min(),
            'date_max': df['Date'].max(),
        }
    
    except Exception as e:
        return None


def analyze_contract(symbol):
    """分析单个合约"""
    result = {'Symbol': symbol}
    
    # NON
    non_info = check_file(DATA_DIR / f'{symbol}_NON.CSV')
    result['NON_file'] = non_info is not None
    result['NON_pre2011'] = non_info['rows_pre2011'] if non_info else 0
    result['NON_test'] = non_info['rows_in_test'] if non_info else 0
    result['NON_post2019'] = non_info['rows_post2019'] if non_info else 0
    
    # RAD
    rad_info = check_file(DATA_DIR / f'{symbol}_RAD.CSV')
    result['RAD_file'] = rad_info is not None
    result['RAD_pre2011'] = rad_info['rows_pre2011'] if rad_info else 0
    result['RAD_test'] = rad_info['rows_in_test'] if rad_info else 0
    result['RAD_post2019'] = rad_info['rows_post2019'] if rad_info else 0
    
    # ASC
    asc_info = check_file(TEMP_DIR / f'{symbol}_CLC.ASC')
    result['ASC_file'] = asc_info is not None
    result['ASC_pre2011'] = asc_info['rows_pre2011'] if asc_info else 0
    result['ASC_test'] = asc_info['rows_in_test'] if asc_info else 0
    result['ASC_post2019'] = asc_info['rows_post2019'] if asc_info else 0
    
    # Rollover
    rollover_info = check_file(DATA_DIR / f'{symbol}_rollovers.csv', is_rollover=True)
    result['Roll_file'] = rollover_info is not None
    result['Roll_pre2011'] = rollover_info['rows_pre2011'] if rollover_info else 0
    result['Roll_test'] = rollover_info['rows_in_test'] if rollover_info else 0
    result['Roll_post2019'] = rollover_info['rows_post2019'] if rollover_info else 0
    
    # 分类
    has_test_rad = result['RAD_test'] > 0
    has_test_asc = result['ASC_test'] > 0
    has_test_roll = result['Roll_test'] > 0
    has_pre2011_asc = result['ASC_pre2011'] > 0
    has_pre2011_roll = result['Roll_pre2011'] > 0
    
    if has_test_rad and has_test_asc and has_test_roll:
        result['Category'] = 'A'
        result['Notes'] = '✅ 完整数据'
    elif has_test_rad and has_test_roll:
        result['Category'] = 'B'
        result['Notes'] = '✅ RAD+rollover 全'
    elif has_test_rad and has_test_asc and not has_test_roll:
        result['Category'] = 'C'
        result['Notes'] = '⚠️ 需提取换月'
    elif has_test_rad and not has_test_asc and not has_test_roll:
        result['Category'] = 'D'
        result['Notes'] = '⚠️ 仅 RAD'
    elif not has_test_rad and has_test_asc and has_test_roll:
        result['Category'] = 'E'
        result['Notes'] = '🔧 可生成 RAD_v2'
    elif not has_test_rad and has_pre2011_asc and has_pre2011_roll and not has_test_roll:
        result['Category'] = 'F'
        result['Notes'] = '❌ 换月 2011 年前结束'
    elif not has_test_rad and not has_test_asc and result['NON_test'] > 0:
        result['Category'] = 'G'
        result['Notes'] = '❌ 仅 NON'
    else:
        result['Category'] = 'H'
        result['Notes'] = '❌ 需检查'
    
    return result


def main():
    print("="*140)
    print("50 个合约数据丰富度完整分析")
    print(f"测试期：{TEST_START.strftime('%Y-%m-%d')} 至 {TEST_END.strftime('%Y-%m-%d')}")
    print("="*140)
    
    results = []
    for symbol in PAPER_CONTRACTS:
        results.append(analyze_contract(symbol))
    
    # 按 Category 排序
    cat_order = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
    results.sort(key=lambda x: (cat_order.get(x['Category'], 9), x['Symbol']))
    
    # 输出表格
    header = (
        f"{'合约':<4} {'类':<2} {'状态':<12} | "
        f"{'NON(前/测/后)':<12} {'RAD(前/测/后)':<12} {'ASC(前/测/后)':<12} {'Roll(前/测/后)':<14} | "
        f"备注"
    )
    print(header)
    print("-"*140)
    
    for r in results:
        non_str = f"{r['NON_pre2011']:>4}/{r['NON_test']:>4}/{r['NON_post2019']:>3}"
        rad_str = f"{r['RAD_pre2011']:>4}/{r['RAD_test']:>4}/{r['RAD_post2019']:>3}"
        asc_str = f"{r['ASC_pre2011']:>4}/{r['ASC_test']:>4}/{r['ASC_post2019']:>3}"
        roll_str = f"{r['Roll_pre2011']:>3}/{r['Roll_test']:>3}/{r['Roll_post2019']:>3}"
        
        print(f"{r['Symbol']:<4} {r['Category']:<2} {r['Notes']:<12} | {non_str:<12} {rad_str:<12} {asc_str:<12} {roll_str:<14} | {r['Notes']}")
    
    # 统计
    print("\n" + "="*140)
    print("统计摘要")
    print("="*140)
    
    cat_counts = {}
    for r in results:
        cat = r['Category']
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    
    cat_labels = {
        'A': '完整数据 (RAD+ASC+rollover)',
        'B': 'RAD+rollover 全',
        'C': 'RAD+ASC (无 rollover)',
        'D': '仅 RAD',
        'E': 'ASC+rollover (可生成)',
        'F': '换月 2011 年前结束',
        'G': '仅 NON',
        'H': '需检查',
    }
    
    for cat in sorted(cat_counts.keys()):
        count = cat_counts[cat]
        print(f"  {cat}: {count}/50 ({count/50*100:.1f}%) - {cat_labels.get(cat, '')}")
    
    # 保存 CSV
    df = pd.DataFrame(results)
    output_file = PROJECT_ROOT / 'tests' / 'results' / 'data_completeness_full.csv'
    df.to_csv(output_file, index=False)
    print(f"\n详细结果：{output_file}")
    
    return results


if __name__ == '__main__':
    main()
