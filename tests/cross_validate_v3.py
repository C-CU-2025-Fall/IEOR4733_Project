"""
CLC 数据交叉验证 v3 - 最终版

核心逻辑:
1. ASC 和 RAD 都是调整后的连续合约 (不同基准)
2. ASC 覆盖期：验证 ASC vs RAD 相关性 → 验证 RAD 方法正确
3. 测试期：ASC 不覆盖，但 RAD 已验证 → 信任 RAD

关键发现:
- ASC 文件不覆盖测试期 (2011-2019)
- 但 ASC 覆盖期 RAD 验证通过 → RAD 方法可靠
- 测试期可信任 RAD 数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'CLC'
CONFIG_DIR = PROJECT_ROOT / 'config'
TEMP_DIR = CONFIG_DIR / 'TEMP'
RESULTS_DIR = PROJECT_ROOT / 'tests' / 'results'

# 50 个合约
ALL_CONTRACTS = ['NR', 'SB', 'SN', 'SP', 'TY', 'US', 'ZA', 'ZC', 'ZF', 'ZG', 'ZH', 'ZI', 'ZK', 'ZL', 'ZN', 'ZO', 'ZP', 'ZR', 'ZT', 'ZU', 'ZW', 'ZZ',
                 'CC', 'DA', 'GI', 'JO', 'KC', 'KW', 'LB', 'CA', 'EN', 'ER', 'ES', 'LX', 'MD', 'SC', 'XU', 'XX', 'YM',
                 'DT', 'FB', 'UB', 'AN', 'BN', 'CN', 'DX', 'FN', 'JN', 'MP', 'NK']

# 加载 roll rules
with open(CONFIG_DIR / 'roll_rules_corrected.json') as f:
    ROLL_RULES = json.load(f)


def load_asc(symbol):
    """加载 ASC 数据"""
    fpath = TEMP_DIR / f'{symbol}_CLC.ASC'
    if not fpath.exists():
        return None
    
    try:
        df = pd.read_csv(fpath, sep=r'\s+', header=None, engine='python',
                         names=['Date','O','H','L','C','V','OI','Adj'],
                         on_bad_lines='skip')
        df['Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m%d', errors='coerce')
        df = df.dropna(subset=['Date'])
        df['C'] = pd.to_numeric(df['C'])
        return df.sort_values('Date').reset_index(drop=True)
    except:
        return None


def load_rad(symbol):
    """加载 RAD 数据"""
    if symbol in ['ZH', 'ZN', 'ZU', 'US']:
        fpath = DATA_DIR / f'{symbol}_RAD_v2.CSV'
    else:
        fpath = DATA_DIR / f'{symbol}_RAD.CSV'
    
    if not fpath.exists():
        return None
    
    df = pd.read_csv(fpath, names=['Date','O','H','L','C','V','OI'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['C'] = pd.to_numeric(df['C'])
    return df.sort_values('Date').reset_index(drop=True)


def load_non(symbol):
    """加载 NON 数据"""
    fpath = DATA_DIR / f'{symbol}_NON.CSV'
    if not fpath.exists():
        return None
    
    df = pd.read_csv(fpath, names=['Date','O','H','L','C','V','OI'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['C'] = pd.to_numeric(df['C'])
    return df.sort_values('Date').reset_index(drop=True)


def validate_symbol(symbol):
    """验证单个合约"""
    result = {
        'Symbol': symbol,
        
        # 数据存在性
        'Has_ASC': False,
        'Has_RAD': False,
        'Has_NON': False,
        'Has_Roll_Rule': False,
        
        # 日期范围
        'ASC_Start': None,
        'ASC_End': None,
        'ASC_Covers_Test_Period': False,
        
        # ASC vs RAD 验证 (重叠期)
        'Overlap_Start': None,
        'Overlap_End': None,
        'Overlap_Days': 0,
        'ASC_RAD_Correlation': np.nan,
        'ASC_RAD_Return_Corr': np.nan,
        'ASC_RAD_Mean_Diff_Pct': np.nan,
        
        # RAD vs NON 验证 (测试期)
        'Test_Period_Corr': np.nan,
        'Test_Period_Max_Gap': np.nan,
        
        # 综合评估
        'RAD_Validated': False,
        'RAD_Trustworthiness': 'Unknown',
        'Quality_Score': 'F',
        'Notes': '',
    }
    
    # 加载数据
    asc = load_asc(symbol)
    rad = load_rad(symbol)
    non = load_non(symbol)
    
    # 检查数据存在性
    if asc is not None:
        result['Has_ASC'] = True
        result['ASC_Start'] = asc['Date'].min()
        result['ASC_End'] = asc['Date'].max()
        result['ASC_Covers_Test_Period'] = result['ASC_End'] >= pd.Timestamp('2011-01-01')
    
    if rad is not None:
        result['Has_RAD'] = True
    
    if non is not None:
        result['Has_NON'] = True
    
    # 检查 Roll Rule
    for rname, rdata in ROLL_RULES.items():
        if symbol in rdata.get('symbols', []):
            result['Has_Roll_Rule'] = True
            break
    
    # ASC vs RAD 验证 (重叠期)
    if asc is not None and rad is not None:
        merged = pd.merge(asc, rad, on='Date', suffixes=('_asc', '_rad'))
        
        if len(merged) > 0:
            result['Overlap_Start'] = merged['Date'].min()
            result['Overlap_End'] = merged['Date'].max()
            result['Overlap_Days'] = len(merged)
            
            # 价格相关性
            result['ASC_RAD_Correlation'] = merged['C_asc'].corr(merged['C_rad'])
            
            # 收益率相关性
            merged['ret_asc'] = merged['C_asc'].pct_change()
            merged['ret_rad'] = merged['C_rad'].pct_change()
            result['ASC_RAD_Return_Corr'] = merged['ret_asc'].corr(merged['ret_rad'])
            
            # 平均差异%
            merged['diff_pct'] = (merged['C_asc'] - merged['C_rad']) / merged['C_rad']
            result['ASC_RAD_Mean_Diff_Pct'] = merged['diff_pct'].mean()
            
            # RAD 验证通过标准：收益率相关性 > 0.99
            if result['ASC_RAD_Return_Corr'] >= 0.99:
                result['RAD_Validated'] = True
    
    # RAD vs NON 验证 (测试期)
    if rad is not None and non is not None:
        test_start = pd.Timestamp('2011-01-01')
        test_end = pd.Timestamp('2019-12-31')
        
        rad_test = rad[(rad['Date'] >= test_start) & (rad['Date'] <= test_end)]
        non_test = non[(non['Date'] >= test_start) & (non['Date'] <= test_end)]
        
        if len(rad_test) > 0 and len(non_test) > 0:
            merged_test = pd.merge(rad_test, non_test, on='Date', suffixes=('_rad', '_non'))
            
            if len(merged_test) > 0:
                result['Test_Period_Corr'] = merged_test['C_rad'].corr(merged_test['C_non'])
                result['Test_Period_Max_Gap'] = merged_test['C_rad'].pct_change().abs().max()
    
    # 综合评估
    if not result['Has_RAD']:
        result['Quality_Score'] = 'F'
        result['Notes'] = 'No RAD data'
    elif result['RAD_Validated']:
        # ASC 验证通过
        if result['Test_Period_Corr'] >= 0.95:
            result['RAD_Trustworthiness'] = 'High'
            result['Quality_Score'] = 'A'
            result['Notes'] = 'ASC validated + high corr'
        elif result['Test_Period_Corr'] >= 0.90:
            result['RAD_Trustworthiness'] = 'High'
            result['Quality_Score'] = 'A'
            result['Notes'] = 'ASC validated'
        else:
            result['RAD_Trustworthiness'] = 'Medium'
            result['Quality_Score'] = 'B'
            result['Notes'] = 'ASC validated, lower test corr'
    elif result['Test_Period_Corr'] >= 0.95:
        # 无 ASC 验证，但测试期相关性高
        result['RAD_Trustworthiness'] = 'Medium'
        result['Quality_Score'] = 'B'
        result['Notes'] = 'No ASC, high test corr'
    elif result['Test_Period_Corr'] >= 0.90:
        result['RAD_Trustworthiness'] = 'Medium'
        result['Quality_Score'] = 'B'
        result['Notes'] = 'No ASC, ok test corr'
    elif result['Test_Period_Corr'] >= 0.70:
        result['RAD_Trustworthiness'] = 'Low'
        result['Quality_Score'] = 'C'
        result['Notes'] = 'Low correlation'
    else:
        result['RAD_Trustworthiness'] = 'Low'
        result['Quality_Score'] = 'C'
        result['Notes'] = 'Very low correlation'
    
    # 检查异常跳变
    if result['Test_Period_Max_Gap'] is not None and result['Test_Period_Max_Gap'] > 0.50:
        result['Notes'] += f' | Large gap: {result["Test_Period_Max_Gap"]:.1%}'
        if result['Quality_Score'] in ['A', 'B']:
            result['Quality_Score'] = 'C'
    
    return result


def main():
    print("="*120)
    print("CLC 数据交叉验证 v3 - ASC/RAD 一致性验证")
    print("="*120)
    print("\n逻辑链:")
    print("1. ASC 和 RAD 都是调整后的连续合约 (不同基准)")
    print("2. ASC 覆盖期：验证 ASC vs RAD 收益率相关性 → 验证 RAD 方法正确")
    print("3. 测试期：ASC 不覆盖，但 RAD 已验证 → 信任 RAD")
    print()
    
    results = []
    for symbol in ALL_CONTRACTS:
        result = validate_symbol(symbol)
        results.append(result)
        
        # 简洁输出
        validated = "✅" if result['RAD_Validated'] else "❓" if not result['Has_ASC'] else "❌"
        trust = result['RAD_Trustworthiness']
        trust_icon = "✅" if trust == 'High' else "✅" if trust == 'Medium' else "⚠️" if trust == 'Low' else "❓"
        
        asc_corr_str = f"{result['ASC_RAD_Return_Corr']:.3f}" if pd.notna(result['ASC_RAD_Return_Corr']) else 'N/A'
        test_corr_str = f"{result['Test_Period_Corr']:.3f}" if pd.notna(result['Test_Period_Corr']) else 'N/A'
        print(f"{symbol}: {validated} {trust_icon} Score={result['Quality_Score']} "
              f"(ASC_RAD_ret_corr={asc_corr_str}, Test_corr={test_corr_str})")
    
    # 保存结果
    RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / 'cross_validation_v3_final.csv', index=False)
    
    # 汇总统计
    print("\n" + "="*120)
    print("汇总统计")
    print("="*120)
    
    # ASC 覆盖情况
    asc_has = sum(1 for r in results if r['Has_ASC'])
    asc_covers = sum(1 for r in results if r['ASC_Covers_Test_Period'])
    print(f"\n有 ASC 文件：{asc_has}/{len(results)}")
    print(f"ASC 覆盖测试期：{asc_covers}/{len(results)} ({asc_covers/len(results)*100:.0f}%)")
    
    # RAD 验证情况
    rad_validated = sum(1 for r in results if r['RAD_Validated'])
    print(f"\nRAD 经 ASC 验证：{rad_validated}/{len(results)} ({rad_validated/len(results)*100:.0f}%)")
    
    # RAD 可信度
    rad_high = sum(1 for r in results if r['RAD_Trustworthiness'] == 'High')
    rad_medium = sum(1 for r in results if r['RAD_Trustworthiness'] == 'Medium')
    rad_low = sum(1 for r in results if r['RAD_Trustworthiness'] == 'Low')
    print(f"\nRAD 可信度高：{rad_high}/{len(results)} ({rad_high/len(results)*100:.0f}%)")
    print(f"RAD 可信度中：{rad_medium}/{len(results)} ({rad_medium/len(results)*100:.0f}%)")
    print(f"RAD 可信度低：{rad_low}/{len(results)} ({rad_low/len(results)*100:.0f}%)")
    
    # 质量评分
    print(f"\n质量评分分布:")
    for score in ['A', 'B', 'C', 'F']:
        count = sum(1 for r in results if r['Quality_Score'] == score)
        print(f"  {score}: {count}/{len(results)} ({count/len(results)*100:.0f}%)")
    
    # 详细表格
    print("\n" + "="*120)
    print("详细结果")
    print("="*120)
    
    df_sorted = df.sort_values(['Quality_Score', 'ASC_RAD_Return_Corr'], ascending=[False, False])
    display_cols = ['Symbol', 'RAD_Validated', 'ASC_RAD_Return_Corr', 
                    'Test_Period_Corr', 'RAD_Trustworthiness', 'Quality_Score', 'Notes']
    
    print(df_sorted[display_cols].to_string(index=False))
    
    print(f"\n输出文件：{RESULTS_DIR / 'cross_validation_v3_final.csv'}")
    print("="*120)
    
    # 结论
    print("\n" + "="*120)
    print("结论")
    print("="*120)
    
    a_count = sum(1 for r in results if r['Quality_Score'] == 'A')
    b_count = sum(1 for r in results if r['Quality_Score'] == 'B')
    ab_pct = (a_count + b_count) / len(results) * 100
    
    print(f"\n✅ {a_count + b_count}/{len(results)} ({ab_pct:.0f}%) 合约 A/B 级，可用于回测")
    
    if rad_validated > 0:
        print(f"✅ {rad_validated} 合约经 ASC 验证，RAD 方法可靠")
        print(f"→ 测试期可信任 RAD 数据 (即使 ASC 不覆盖)")
    
    print("="*120)


if __name__ == '__main__':
    main()
