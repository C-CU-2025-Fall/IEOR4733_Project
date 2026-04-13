"""
全面分析 50 个合约的数据丰富度

数据源优先级：
1. NON + RAD + ASC + rollovers (最完整)
2. RAD + ASC (可用 ASC 调整)
3. RAD only (可用 NON 验证调整频率)
4. 数据不全，无法理论补齐

测试期：2011-01-01 至 2019-12-31
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

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


def parse_date(date_str):
    """解析多种日期格式"""
    if pd.isna(date_str):
        return None
    
    date_str = str(date_str).strip()
    
    # 尝试不同格式
    for fmt in ['%Y%m%d', '%m/%d/%Y', '%Y-%m-%d']:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue
    
    return None


def check_file(file_path, is_rollover=False):
    """检查文件是否存在并返回日期范围"""
    if not file_path.exists():
        return None
    
    try:
        if file_path.suffix == '.ASC':
            # ASC 文件：空格分隔
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
                # Rollover 文件
                df = pd.read_csv(file_path)
                df['Date'] = pd.to_datetime(df['RollDate'])
            else:
                # NON/RAD 文件
                df = pd.read_csv(file_path, names=['Date','O','H','L','C','V','OI'])
                df['Date'] = pd.to_datetime(df['Date'])
        
        else:
            return None
        
        return {
            'exists': True,
            'rows': len(df),
            'date_min': df['Date'].min(),
            'date_max': df['Date'].max(),
            'rows_in_test': len(df[(df['Date'] >= TEST_START) & (df['Date'] <= TEST_END)]),
        }
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def analyze_contract(symbol):
    """分析单个合约的数据丰富度"""
    result = {
        'Symbol': symbol,
        'NON_exists': False,
        'NON_test_rows': 0,
        'NON_date_range': None,
        'RAD_exists': False,
        'RAD_test_rows': 0,
        'RAD_date_range': None,
        'ASC_exists': False,
        'ASC_test_rows': 0,
        'ASC_date_range': None,
        'Rollover_exists': False,
        'Rollover_test_count': 0,
        'Rollover_date_range': None,
        'Category': 'Unknown',
        'Notes': '',
    }
    
    # 检查 NON 文件
    non_file = DATA_DIR / f'{symbol}_NON.CSV'
    non_info = check_file(non_file)
    if non_info:
        result['NON_exists'] = True
        result['NON_test_rows'] = non_info['rows_in_test']
        result['NON_date_range'] = f"{non_info['date_min'].strftime('%Y-%m')} - {non_info['date_max'].strftime('%Y-%m')}"
    
    # 检查 RAD 文件
    rad_file = DATA_DIR / f'{symbol}_RAD.CSV'
    rad_info = check_file(rad_file)
    if rad_info:
        result['RAD_exists'] = True
        result['RAD_test_rows'] = rad_info['rows_in_test']
        result['RAD_date_range'] = f"{rad_info['date_min'].strftime('%Y-%m')} - {rad_info['date_max'].strftime('%Y-%m')}"
    
    # 检查 ASC 文件
    asc_file = TEMP_DIR / f'{symbol}_CLC.ASC'
    asc_info = check_file(asc_file)
    if asc_info:
        result['ASC_exists'] = True
        result['ASC_test_rows'] = asc_info['rows_in_test']
        result['ASC_date_range'] = f"{asc_info['date_min'].strftime('%Y-%m')} - {asc_info['date_max'].strftime('%Y-%m')}"
    
    # 检查 rollover 文件
    rollover_file = DATA_DIR / f'{symbol}_rollovers.csv'
    rollover_info = check_file(rollover_file, is_rollover=True)
    if rollover_info:
        result['Rollover_exists'] = True
        result['Rollover_test_count'] = rollover_info['rows_in_test']
        result['Rollover_date_range'] = f"{rollover_info['date_min'].strftime('%Y-%m')} - {rollover_info['date_max'].strftime('%Y-%m')}"
    
    # 分类 - 基于数据补齐可行性
    has_test_non = result['NON_test_rows'] > 0
    has_test_rad = result['RAD_test_rows'] > 0
    has_test_asc = result['ASC_test_rows'] > 0
    has_test_rollover = result['Rollover_test_count'] > 0
    
    # 检查 2011 年前是否有数据
    has_pre2011_asc = result['ASC_exists'] and result['ASC_date_range'] and result['ASC_date_range'].split(' - ')[0] < '2011-01'
    has_pre2011_rollover = result['Rollover_exists'] and result['Rollover_date_range'] and result['Rollover_date_range'].split(' - ')[0] < '2011-01'
    
    if has_test_rad and has_test_asc and has_test_rollover:
        result['Category'] = 'A: 完整数据 (RAD+ASC+rollover 测试期全)'
        result['Notes'] = '✅ 可直接用，无需补齐'
        
    elif has_test_rad and has_test_rollover:
        result['Category'] = 'B: RAD+rollover 测试期全 (ASC 部分)'
        result['Notes'] = '✅ 可用 NON 验证 RAD 调整'
        
    elif has_test_rad and has_test_asc and not has_test_rollover:
        result['Category'] = 'C: RAD+ASC 测试期全 (无 rollover)'
        result['Notes'] = '⚠️ 需从 ASC 提取换月'
        
    elif has_test_rad and not has_test_asc and not has_test_rollover:
        result['Category'] = 'D: 仅 RAD 测试期数据'
        result['Notes'] = '⚠️ 无法验证调整来源，RAD 本身可用'
        
    elif not has_test_rad and has_test_asc and has_test_rollover:
        result['Category'] = 'E: ASC+rollover 测试期全 (RAD 缺失)'
        result['Notes'] = '🔧 可用 ASC+rollover 生成 RAD_v2'
        
    elif not has_test_rad and has_pre2011_asc and has_pre2011_rollover and not has_test_rollover:
        result['Category'] = 'F: ASC+rollover 只到 2011 年前'
        result['Notes'] = '❌ 换月在测试期前结束，无法生成 RAD_v2'
        
    elif not has_test_rad and not has_test_asc and has_test_non:
        result['Category'] = 'G: 仅 NON 测试期数据'
        result['Notes'] = '❌ 无 RAD/ASC 参考，无法生成 RAD_v2'
        
    else:
        result['Category'] = 'H: 数据不全'
        result['Notes'] = '❌ 需手动检查'
    
    # 备注
    notes = []
    if not has_test_non and result['NON_exists']:
        notes.append("NON 测试期无数据")
    if not has_test_rad and result['RAD_exists']:
        notes.append("RAD 测试期无数据")
    if not has_test_asc and result['ASC_exists']:
        notes.append("ASC 测试期无数据")
    if not has_test_rollover:
        notes.append("无换月记录")
    
    result['Notes'] = '; '.join(notes) if notes else '数据完整'
    
    return result


def main():
    print("="*120)
    print("50 个合约数据丰富度分析")
    print(f"测试期：{TEST_START.strftime('%Y-%m-%d')} 至 {TEST_END.strftime('%Y-%m-%d')}")
    print("="*120)
    
    results = []
    for symbol in PAPER_CONTRACTS:
        result = analyze_contract(symbol)
        results.append(result)
    
    # 按 Category 排序
    tier_order = {
        'Tier 1': 1, 'Tier 2': 2, 'Tier 3': 3, 'Tier 4': 4,
        'Tier 5': 5, 'Tier 6': 6, 'Tier 7': 7
    }
    results.sort(key=lambda x: (tier_order.get(x['Category'][:6], 9), x['Symbol']))
    
    # 输出详细表格
    print(f"\n{'合约':<4} {'类别':<35} {'NON 测试期':>10} {'RAD 测试期':>10} {'ASC 测试期':>10} {'换月次数':>10} {'备注':<30}")
    print("-"*120)
    
    for r in results:
        print(f"{r['Symbol']:<4} {r['Category']:<35} {r['NON_test_rows']:>10} {r['RAD_test_rows']:>10} {r['ASC_test_rows']:>10} {r['Rollover_test_count']:>10} {r['Notes']:<30}")
    
    # 统计
    print("\n" + "="*120)
    print("统计摘要")
    print("="*120)
    
    tier_counts = {}
    for r in results:
        tier = r['Category'].split(':')[0]
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    
    for tier, count in sorted(tier_counts.items()):
        print(f"{tier}: {count}/50 ({count/50*100:.1f}%)")
    
    # 保存 CSV
    df = pd.DataFrame(results)
    output_file = PROJECT_ROOT / 'tests' / 'results' / 'data_completeness_analysis.csv'
    df.to_csv(output_file, index=False)
    print(f"\n详细结果已保存：{output_file}")
    
    return results


if __name__ == '__main__':
    main()
