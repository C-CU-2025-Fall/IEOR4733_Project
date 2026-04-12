"""
从 CLC ASC 原始文件提取换月数据

ASC 文件格式:
- 正常行：Date Open High Low Close Volume OI ???
- 换月行：00000000 c 0.00 0.00 C 0 0 0
  - c = 旧合约 Close
  - C = 新合约 Close

Usage:
    python tests/extract_rollover_from_asc.py
"""

import pandas as pd
from pathlib import Path

TEMP_DIR = Path('/home/congge2026/.openclaw/workspace/IEOR4733_Project/config/TEMP')
OUTPUT_DIR = Path('/home/congge2026/.openclaw/workspace/IEOR4733_Project/data/CLC')


def parse_asc_file(asc_file):
    """解析 ASC 文件，提取换月记录"""
    symbol = asc_file.stem.replace('_CLC', '')
    
    rollovers = []
    prev_date = None
    prev_close = None
    
    with open(asc_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            date_str = parts[0]
            
            if date_str == '00000000':
                # 换月行：00000000 c 0.00 0.00 C 0 0 0
                c = float(parts[1])  # 旧合约 Close
                C = float(parts[4])  # 新合约 Close
                ratio = c / C if C > 0 else None
                
                rollovers.append({
                    'Symbol': symbol,
                    'RollDate': prev_date,
                    'PrevClose_c': c,
                    'NewClose_C': C,
                    'Ratio_c_over_C': ratio
                })
            else:
                # 正常行
                date = pd.to_datetime(date_str, format='%Y%m%d')
                close = float(parts[4])
                prev_date = date
                prev_close = close
    
    return pd.DataFrame(rollovers)


def main():
    print("=" * 60)
    print("从 ASC 文件提取换月数据")
    print("=" * 60)
    
    asc_files = list(TEMP_DIR.glob('*_CLC.ASC'))
    print(f"\n找到 {len(asc_files)} 个 ASC 文件")
    
    all_rollovers = []
    for asc_file in asc_files:
        symbol = asc_file.stem.replace('_CLC', '')
        try:
            df = parse_asc_file(asc_file)
            all_rollovers.append(df)
            print(f"  {symbol}: {len(df)} 次换月")
        except Exception as e:
            print(f"  {symbol}: ERROR - {e}")
    
    if len(all_rollovers) == 0:
        print("没有成功解析任何文件")
        return
    
    # 合并所有换月记录
    rollover_df = pd.concat(all_rollovers, ignore_index=True)
    
    # 保存
    output_file = OUTPUT_DIR / 'clc_rollovers_from_asc.csv'
    rollover_df.to_csv(output_file, index=False)
    print(f"\n已保存到：{output_file}")
    
    # 统计
    print(f"\n=== 统计 ===")
    print(f"总换月次数：{len(rollover_df)}")
    print(f"合约数量：{rollover_df['Symbol'].nunique()}")
    
    # 每个合约的换月次数
    print(f"\n=== 每个合约换月次数 ===")
    roll_counts = rollover_df.groupby('Symbol').size().sort_values(ascending=False)
    print(roll_counts.to_string())
    
    # 检查 ratio 分布
    print(f"\n=== Ratio (c/C) 分布 ===")
    valid_ratios = rollover_df[rollover_df['Ratio_c_over_C'].notna()]['Ratio_c_over_C']
    print(f"中位数：{valid_ratios.median():.6f}")
    print(f"平均值：{valid_ratios.mean():.6f}")
    print(f"最小值：{valid_ratios.min():.6f}")
    print(f"最大值：{valid_ratios.max():.6f}")
    
    # 保存每个合约的换月日期
    for symbol in rollover_df['Symbol'].unique():
        symbol_df = rollover_df[rollover_df['Symbol'] == symbol]
        symbol_output = OUTPUT_DIR / f'{symbol}_rollovers.csv'
        symbol_df.to_csv(symbol_output, index=False)


if __name__ == '__main__':
    main()
