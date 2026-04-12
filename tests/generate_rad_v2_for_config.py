"""
为 config.py 中定义的 27 个合约生成 RAD_v2_fixed

专注于数据对齐：
1. 只处理 config.ASSET_CLASSES 中的合约
2. 确保测试期 (2011-2019) 内有换月数据
3. 正确应用换月 ratio（换月日 +1 开始）
4. 输出与 CLC RAD 格式对齐

Usage:
    python tests/generate_rad_v2_for_config.py
"""

import pandas as pd
from pathlib import Path
import sys

# 添加项目根目录到 Python 路径，以便导入 config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

# =============================================================================
# 路径配置
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / 'config'
DATA_DIR = PROJECT_ROOT / 'data' / 'CLC'
TEMP_DIR = CONFIG_DIR / 'TEMP'
OUTPUT_DIR = DATA_DIR

# =============================================================================
# 测试期（用于验证对比）
# =============================================================================
TEST_START = pd.Timestamp('2011-01-01')
TEST_END = pd.Timestamp('2019-12-31')


def generate_rad_v2_fixed(symbol: str) -> dict:
    """
    生成单个合约的 RAD_v2_fixed 文件
    
    Args:
        symbol: 合约代码 (如 'ES', 'DJ')
    
    Returns:
        生成结果字典，包含 status, corr, scale 等信息
    """
    asc_file = TEMP_DIR / f'{symbol}_CLC.ASC'
    non_file = DATA_DIR / f'{symbol}_NON.CSV'
    rollover_file = DATA_DIR / f'{symbol}_rollovers.csv'
    output_file = OUTPUT_DIR / f'{symbol}_RAD_v2_fixed.CSV'
    
    # 检查必需文件
    if not all(f.exists() for f in [asc_file, non_file, rollover_file]):
        missing = [f for f in [asc_file, non_file, rollover_file] if not f.exists()]
        return {'status': f'MISSING_FILES: {", ".join(f.name for f in missing)}'}
    
    # 读取换月数据
    rollover_df = pd.read_csv(rollover_file)
    rollover_df['RollDate'] = pd.to_datetime(rollover_df['RollDate'])
    
    # 读取 NON 数据
    non = pd.read_csv(non_file, names=['Date', 'O', 'H', 'L', 'C', 'V', 'OI'])
    non['Date'] = pd.to_datetime(non['Date'])
    non = non.sort_values('Date').reset_index(drop=True)
    
    if len(non) == 0:
        return {'status': 'EMPTY_NON'}
    
    # 匹配换月日期到 NON 数据
    rolls = []
    for _, row in rollover_df.iterrows():
        roll_date = row['RollDate']
        idx = non[non['Date'] == roll_date].index
        if len(idx) > 0:
            rolls.append({'idx': idx[0], 'c': row['PrevClose_c'], 'C': row['NewClose_C']})
        else:
            # 允许 ±5 天误差
            diffs = (non['Date'] - roll_date).abs()
            if diffs.min().days <= 5:
                rolls.append({'idx': diffs.idxmin(), 'c': row['PrevClose_c'], 'C': row['NewClose_C']})
    
    if len(rolls) == 0:
        return {'status': 'NO_ROLLS_MATCHED'}
    
    # 过滤测试期内的换月
    # 找到测试期开始索引
    test_start_mask = non['Date'] >= TEST_START
    if not test_start_mask.any():
        return {'status': 'NO_TEST_DATA'}
    test_start_idx = test_start_mask.idxmax()
    
    rolls_in_test = [r for r in rolls if r['idx'] >= test_start_idx]
    if len(rolls_in_test) == 0:
        return {'status': 'NO_ROLLS_IN_TEST'}
    
    # 生成 RAD - 关键修复：换月日当天仍用旧 ratio，次日开始用新 ratio
    non['ratio'] = 1.0
    current_ratio = 1.0
    
    for i, roll in enumerate(rolls_in_test):
        idx = roll['idx']
        c, C = roll['c'], roll['C']
        
        if C > 0 and c > 0:
            current_ratio *= (c / C)
            # 下一个换月前的所有天数都用这个 ratio
            next_idx = rolls_in_test[i + 1]['idx'] if i + 1 < len(rolls_in_test) else len(non)
            # 从 roll_date + 1 开始应用新 ratio
            non.loc[idx + 1:next_idx, 'ratio'] = current_ratio
    
    # 计算 RAD 价格
    non['RAD_Open'] = non['O'] * non['ratio']
    non['RAD_High'] = non['H'] * non['ratio']
    non['RAD_Low'] = non['L'] * non['ratio']
    non['RAD_Close'] = non['C'] * non['ratio']
    
    # 输出测试期数据
    test_data = non[(non['Date'] >= TEST_START) & (non['Date'] <= TEST_END)].copy()
    output_df = test_data[['Date', 'RAD_Open', 'RAD_High', 'RAD_Low', 'RAD_Close', 'V', 'OI']].copy()
    output_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInterest']
    output_df['Date'] = output_df['Date'].dt.strftime('%m/%d/%Y')
    output_df.to_csv(output_file, index=False, header=False)
    
    # 验证与 CLC RAD 对比
    rad_file = DATA_DIR / f'{symbol}_RAD.CSV'
    if rad_file.exists():
        rad = pd.read_csv(rad_file, names=['Date', 'O', 'H', 'L', 'C', 'V', 'OI'])
        rad['Date'] = pd.to_datetime(rad['Date'])
        rad_test = rad[(rad['Date'] >= TEST_START) & (rad['Date'] <= TEST_END)].reset_index(drop=True)
        
        if len(rad_test) > 0 and len(test_data) > 0:
            min_len = min(len(rad_test), len(test_data))
            # 注意：test_data 的列名是 'RAD_Close'（还未重命名，重命名的是 output_df）
            # 关键：reset_index 确保 index 对齐，否则 corr() 返回 nan
            corr = test_data['RAD_Close'].iloc[:min_len].reset_index(drop=True).corr(
                   rad_test['C'].iloc[:min_len].reset_index(drop=True))
            
            # 对齐因子（第一天价格）
            scale = rad_test.iloc[0]['C'] / test_data['RAD_Close'].iloc[0] if test_data['RAD_Close'].iloc[0] > 0 else None
            
            # 确保 corr 是 Python float 类型（避免 numpy 类型序列化问题）
            corr_float = float(corr) if corr is not None and not pd.isna(corr) else None
            scale_float = float(scale) if scale is not None and not pd.isna(scale) else None
            
            return {
                'status': 'OK',
                'rolls': len(rolls_in_test),
                'rows': len(test_data),
                'corr': corr_float,
                'scale': scale_float,
                'first_date': str(test_data['Date'].iloc[0]),
                'first_rad': float(rad_test.iloc[0]['C']),
                'first_v2': float(test_data['RAD_Close'].iloc[0])
            }
    
    return {'status': 'NO_RAD_FILE', 'rolls': len(rolls_in_test)}


def main():
    print("=" * 80)
    print("为 config.py 中的合约生成 RAD_v2_fixed")
    print("测试期：2011-01-01 至 2019-12-31")
    print("=" * 80)
    
    results = []
    
    for asset_class, contracts in config.ASSET_CLASSES.items():
        print(f"\n{asset_class} ({len(contracts)} 个合约):")
        
        for symbol in contracts:
            result = generate_rad_v2_fixed(symbol)
            if result:
                results.append({'Symbol': symbol, 'AssetClass': asset_class, **result})
                
                if result['status'] == 'OK':
                    corr_str = f"{result['corr']:.6f}" if result.get('corr') else 'N/A'
                    print(f"  {symbol}: {result['rolls']} rolls, corr={corr_str} ✓")
                else:
                    print(f"  {symbol}: {result['status']} ✗")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv('tests/results/rad_v2_config_generation_summary.csv', index=False)
    print(f"\n详细结果：tests/results/rad_v2_config_generation_summary.csv")
    
    # 统计
    ok_df = results_df[results_df['status'] == 'OK']
    print(f"\n{'=' * 80}")
    print(f"成功生成：{len(ok_df)}/{len(results_df)} 合约")
    
    if len(ok_df) > 0:
        valid_corr = ok_df[ok_df['corr'].notna()]
        if len(valid_corr) > 0:
            print(f"\n相关性统计:")
            print(f"  ≥0.9999: {len(valid_corr[valid_corr['corr']>=0.9999])} ({len(valid_corr[valid_corr['corr']>=0.9999])/len(valid_corr)*100:.1f}%)")
            print(f"  ≥0.999:  {len(valid_corr[valid_corr['corr']>=0.999])} ({len(valid_corr[valid_corr['corr']>=0.999])/len(valid_corr)*100:.1f}%)")
            print(f"  ≥0.99:   {len(valid_corr[valid_corr['corr']>=0.99])} ({len(valid_corr[valid_corr['corr']>=0.99])/len(valid_corr)*100:.1f}%)")
            print(f"  中位数：{valid_corr['corr'].median():.6f}")


if __name__ == '__main__':
    main()
