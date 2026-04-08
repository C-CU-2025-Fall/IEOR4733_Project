#!/usr/bin/env python3
"""
data_quality_check.py — 论文 50 个期货合约的完整数据质量检查

参考：Zhang, Zohren, Roberts (2020) "Deep Reinforcement Learning for Trading"
- 50 most liquid futures contracts
- Test period: 2011-2019
- Data source: Pinnacle/CLC ratio-adjusted continuous contracts

检查维度:
1. 数据完整性（行数、日期覆盖）
2. 价格合理性（范围、跳空）
3. 收益分布（极端值、波动率）
4. 展期跳跃检测（ratio-adjusted 数据的关键问题）
5. 与论文对齐（测试期内有效数据）

输出:
- data_quality_report_50contracts.md - Markdown 报告
- data_quality_summary.csv - CSV 汇总
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# =============================================================================
# 论文中的 50 个合约（Appendix A）
# =============================================================================
PAPER_50_CONTRACTS = {
    'Commodity': [
        'CC', 'DA', 'GI', 'JO', 'KC', 'KW', 'LB', 'NR', 'SB',  # 9
        'ZA', 'ZC', 'ZF', 'ZG', 'ZI', 'ZK', 'ZL', 'ZN', 'ZO',  # 8
        'ZP', 'ZR', 'ZT', 'ZW', 'ZZ',  # 5
        # 论文还有 ZH, ZU (25 个商品期货)
    ],
    'Equity Index': [
        'CA', 'EN', 'ER', 'ES', 'LX', 'MD', 'SC', 'SP', 'XU', 'XX', 'YM',  # 11
    ],
    'Fixed Income': [
        'DT', 'FB', 'TY', 'UB', 'US',  # 5
    ],
    'Forex': [
        'AN', 'BN', 'CN', 'DX', 'FN', 'JN', 'MP', 'NK', 'SN',  # 9
    ],
}

# 论文中的完整 50 合约列表（根据 config.py 和论文 Appendix）
ALL_50_TICKERS = [
    # Commodity (25)
    'CC', 'DA', 'GI', 'JO', 'KC', 'KW', 'LB', 'NR', 'SB',
    'ZA', 'ZC', 'ZF', 'ZG', 'ZH', 'ZI', 'ZK', 'ZL', 'ZN', 'ZO',
    'ZP', 'ZR', 'ZT', 'ZU', 'ZW', 'ZZ',
    # Equity Index (11)
    'CA', 'EN', 'ER', 'ES', 'LX', 'MD', 'SC', 'SP', 'XU', 'XX', 'YM',
    # Fixed Income (5)
    'DT', 'FB', 'TY', 'UB', 'US',
    # Forex (9)
    'AN', 'BN', 'CN', 'DX', 'FN', 'JN', 'MP', 'NK', 'SN',
]

# 测试期
TEST_START = '2011-01-01'
TEST_END = '2019-12-31'

# CLC 数据路径
CLC_DATA_DIR = 'data/CLC'


# =============================================================================
# 数据加载
# =============================================================================
def load_clc_data(ticker: str) -> Optional[pd.DataFrame]:
    """
    加载 CLC ratio-adjusted 数据
    
    CLC 格式：MM/DD/YYYY, Open, High, Low, Close, Volume, OI (无表头)
    """
    filepath = os.path.join(CLC_DATA_DIR, f'{ticker}_RAD.CSV')
    
    if not os.path.exists(filepath):
        return None
    
    try:
        df = pd.read_csv(
            filepath,
            header=None,
            names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'],
            dtype={
                'Open': float, 'High': float, 'Low': float,
                'Close': float, 'Volume': float, 'OI': float
            }
        )
        
        # 解析日期
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
        
        # 清理无效数据
        df = df[df['Date'].notna()]
        df = df[df['Close'].notna() & (df['Close'] > 0)]
        
        # 按日期排序
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df
    
    except Exception as e:
        print(f"Error loading {ticker}: {e}")
        return None


# =============================================================================
# 质量检查指标
# =============================================================================
def check_data_completeness(df: pd.DataFrame, ticker: str) -> Dict:
    """检查数据完整性"""
    total_rows = len(df)
    
    # 测试期数据
    test_mask = (df['Date'] >= TEST_START) & (df['Date'] <= TEST_END)
    test_rows = test_mask.sum()
    
    # 日期覆盖
    if len(df) > 0:
        date_start = df['Date'].min()
        date_end = df['Date'].max()
        date_span_days = (date_end - date_start).days
    else:
        date_start = None
        date_end = None
        date_span_days = 0
    
    # 缺失日期检测（交易日应该约 252 天/年）
    expected_test_days = len(pd.date_range(TEST_START, TEST_END, freq='B'))
    actual_test_days = test_rows
    coverage_pct = actual_test_days / expected_test_days * 100 if expected_test_days > 0 else 0
    
    return {
        'ticker': ticker,
        'total_rows': total_rows,
        'test_rows': int(test_rows),
        'date_start': str(date_start) if date_start is not None else 'N/A',
        'date_end': str(date_end) if date_end is not None else 'N/A',
        'date_span_days': date_span_days,
        'expected_test_days': expected_test_days,
        'coverage_pct': round(coverage_pct, 1),
    }


def check_price_reasonableness(df: pd.DataFrame, ticker: str) -> Dict:
    """检查价格合理性"""
    if len(df) == 0:
        return {'ticker': ticker, 'price_min': None, 'price_max': None, 'price_ratio': None}
    
    prices = df['Close'].values
    price_min = prices.min()
    price_max = prices.max()
    price_ratio = price_max / price_min if price_min > 0 else np.inf
    
    # 检测极端价格（>1000 倍增长通常有问题）
    extreme_price = price_ratio > 1000
    
    return {
        'ticker': ticker,
        'price_min': round(price_min, 4),
        'price_max': round(price_max, 4),
        'price_ratio': round(price_ratio, 2) if price_ratio != np.inf else np.inf,
        'extreme_price': extreme_price,
    }


def check_return_anomalies(df: pd.DataFrame, ticker: str) -> Dict:
    """检查收益异常（跳空检测）"""
    if len(df) < 2:
        return {'ticker': ticker, 'return_max_pct': None, 'return_min_pct': None, 'gaps_count': {}}
    
    prices = df['Close'].values
    # 日收益 = (p_t - p_{t-1}) / p_{t-1}
    returns = np.zeros(len(prices))
    returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1] * 100  # 百分比
    
    return_max = returns.max()
    return_min = returns.min()
    
    # 统计不同阈值的跳空数量
    gaps = {
        '>5%': int((np.abs(returns) > 5).sum()),
        '>10%': int((np.abs(returns) > 10).sum()),
        '>20%': int((np.abs(returns) > 20).sum()),
        '>50%': int((np.abs(returns) > 50).sum()),
        '>100%': int((np.abs(returns) > 100).sum()),
    }
    
    # 找到最大跳空的日期
    max_gap_idx = np.argmax(np.abs(returns))
    max_gap_date = str(df['Date'].iloc[max_gap_idx]) if max_gap_idx > 0 else 'N/A'
    max_gap_value = returns[max_gap_idx] if max_gap_idx > 0 else 0
    
    return {
        'ticker': ticker,
        'return_max_pct': round(return_max, 2),
        'return_min_pct': round(return_min, 2),
        'gaps_count': gaps,
        'max_gap_date': max_gap_date,
        'max_gap_value_pct': round(max_gap_value, 2),
    }


def check_volatility_stability(df: pd.DataFrame, ticker: str) -> Dict:
    """检查波动率稳定性（EWMA vol）"""
    if len(df) < 252:  # 至少需要 1 年数据
        return {'ticker': ticker, 'vol_mean': None, 'vol_std': None, 'vol_cv': None}
    
    prices = df['Close'].values
    returns = np.zeros(len(prices))
    returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
    
    # 60 日 EWMA 波动率
    vol = pd.Series(returns).ewm(span=60, adjust=False).std().values * np.sqrt(252)
    vol = vol[60:]  # 去掉 warmup
    
    vol_mean = vol.mean()
    vol_std = vol.std()
    vol_cv = vol_std / vol_mean if vol_mean > 0 else 0  # 变异系数
    
    return {
        'ticker': ticker,
        'vol_mean': round(vol_mean, 4),
        'vol_std': round(vol_std, 4),
        'vol_cv': round(vol_cv, 4),
        'vol_min': round(vol.min(), 4),
        'vol_max': round(vol.max(), 4),
    }


def check_roll_artifacts(df: pd.DataFrame, ticker: str) -> Dict:
    """
    检测展期异常（ratio-adjusted 数据的关键问题）
    
    Ratio-adjusted 数据应该在展期日没有价格跳跃。
    如果检测到大的跳跃，说明调整因子可能有问题。
    """
    if len(df) < 2:
        return {'ticker': ticker, 'roll_issues': None}
    
    prices = df['Close'].values
    returns = np.zeros(len(prices))
    returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
    
    # 检测异常大的单日收益（可能是展期未正确调整）
    # 期货正常日波动通常 <5%，>10% 很可能是展期问题
    roll_issue_threshold = 0.10  # 10%
    roll_issues = np.abs(returns) > roll_issue_threshold
    
    # 统计展期问题
    issue_dates = df['Date'].iloc[np.where(roll_issues)[0]]
    
    return {
        'ticker': ticker,
        'roll_issue_count': int(roll_issues.sum()),
        'roll_issue_pct': round(roll_issues.sum() / len(returns) * 100, 2),
        'recent_issue_dates': [str(d)[:10] for d in issue_dates[-5:]] if len(issue_dates) > 0 else [],
    }


# =============================================================================
# 综合评分
# =============================================================================
def compute_quality_score(completeness: Dict, price: Dict, returns: Dict, vol: Dict, roll: Dict) -> Tuple[float, str]:
    """
    计算数据质量综合评分（0-100）
    
    评分标准:
    - 完整性 (25 分): 测试期覆盖率
    - 价格合理性 (25 分): 价格比率不过大
    - 收益分布 (25 分): 跳空数量
    - 波动率稳定 (15 分): CV 不过大
    - 展期质量 (10 分): 展期问题数量
    """
    score = 0
    
    # 完整性 (25 分)
    coverage_score = min(completeness['coverage_pct'], 100) * 0.25
    score += coverage_score
    
    # 价格合理性 (25 分)
    if price['price_ratio'] is not None and price['price_ratio'] != np.inf:
        if price['price_ratio'] < 10:
            price_score = 25
        elif price['price_ratio'] < 100:
            price_score = 20
        elif price['price_ratio'] < 1000:
            price_score = 10
        else:
            price_score = 0
    else:
        price_score = 0
    score += price_score
    
    # 收益分布 (25 分)
    gaps_10pct = returns['gaps_count'].get('>10%', 0)
    total_rows = completeness['test_rows']
    gap_ratio = gaps_10pct / total_rows if total_rows > 0 else 0
    
    if gap_ratio < 0.001:  # <0.1% 跳空
        gap_score = 25
    elif gap_ratio < 0.01:  # <1%
        gap_score = 20
    elif gap_ratio < 0.05:  # <5%
        gap_score = 10
    else:
        gap_score = 0
    score += gap_score
    
    # 波动率稳定 (15 分)
    if vol['vol_cv'] is not None:
        if vol['vol_cv'] < 0.5:
            vol_score = 15
        elif vol['vol_cv'] < 1.0:
            vol_score = 10
        else:
            vol_score = 5
    else:
        vol_score = 0
    score += vol_score
    
    # 展期质量 (10 分)
    roll_issue_pct = roll.get('roll_issue_pct', 100)
    if roll_issue_pct < 0.1:
        roll_score = 10
    elif roll_issue_pct < 1:
        roll_score = 7
    elif roll_issue_pct < 5:
        roll_score = 3
    else:
        roll_score = 0
    score += roll_score
    
    # 评级
    if score >= 90:
        grade = 'A'
    elif score >= 75:
        grade = 'B'
    elif score >= 60:
        grade = 'C'
    elif score >= 40:
        grade = 'D'
    else:
        grade = 'F'
    
    return round(score, 1), grade


# =============================================================================
# 报告生成
# =============================================================================
def generate_report(results: List[Dict]) -> str:
    """生成 Markdown 报告"""
    report = []
    report.append("# 论文 50 合约数据质量检查报告")
    report.append("")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**数据源**: CLC Ratio-Adjusted Futures Data")
    report.append(f"**测试期**: {TEST_START} 至 {TEST_END}")
    report.append("")
    
    # 汇总统计
    total = len(results)
    grades = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
    for r in results:
        grades[r['grade']] += 1
    
    report.append("## 📊 汇总统计")
    report.append("")
    report.append(f"| 评级 | 数量 | 比例 |")
    report.append(f"|------|------|------|")
    for grade, count in grades.items():
        pct = count / total * 100 if total > 0 else 0
        report.append(f"| {grade} | {count} | {pct:.1f}% |")
    report.append("")
    
    # 按资产类别分组
    report.append("## 📈 按资产类别")
    report.append("")
    
    for asset_class, tickers in PAPER_50_CONTRACTS.items():
        report.append(f"### {asset_class}")
        report.append("")
        report.append(f"| Ticker | 评分 | 等级 | 测试期行数 | 覆盖率 | >10% 跳空 | 最大跳空 | 状态 |")
        report.append(f"|--------|------|------|------------|--------|----------|----------|------|")
        
        for ticker in tickers:
            r = next((x for x in results if x['ticker'] == ticker), None)
            if r is None:
                report.append(f"| {ticker} | - | F | 0 | 0% | 0 | - | ❌ 无数据 |")
            else:
                status = "✅" if r['grade'] in ['A', 'B'] else "⚠️" if r['grade'] == 'C' else "❌"
                report.append(
                    f"| {ticker} | {r['score']} | {r['grade']} | "
                    f"{r['completeness']['test_rows']} | "
                    f"{r['completeness']['coverage_pct']}% | "
                    f"{r['returns']['gaps_count'].get('>10%', 0)} | "
                    f"{r['returns']['max_gap_value_pct']:.1f}% | "
                    f"{status} |"
                )
        report.append("")
    
    # 问题合约详情
    report.append("## 🔴 问题合约详情")
    report.append("")
    
    problem_contracts = [r for r in results if r['grade'] in ['D', 'F']]
    if problem_contracts:
        for r in problem_contracts:
            report.append(f"### {r['ticker']} (评分：{r['score']}, 等级：{r['grade']})")
            report.append("")
            report.append(f"- **测试期行数**: {r['completeness']['test_rows']} / {r['completeness']['expected_test_days']}")
            report.append(f"- **覆盖率**: {r['completeness']['coverage_pct']}%")
            report.append(f"- **价格范围**: {r['price']['price_min']} → {r['price']['price_max']} (比率：{r['price']['price_ratio']})")
            report.append(f"- **最大跳空**: {r['returns']['max_gap_value_pct']:.1f}% ({r['returns']['max_gap_date']})")
            report.append(f"- **>10% 跳空数量**: {r['returns']['gaps_count'].get('>10%', 0)}")
            report.append(f"- **展期问题**: {r['roll'].get('roll_issue_count', 0)} ({r['roll'].get('roll_issue_pct', 0)}%)")
            report.append("")
    else:
        report.append("所有合约质量良好！")
        report.append("")
    
    # 建议
    report.append("## 💡 建议")
    report.append("")
    
    a_grade = [r for r in results if r['grade'] == 'A']
    f_grade = [r for r in results if r['grade'] == 'F']
    
    if a_grade:
        report.append("### ✅ 可使用的合约")
        report.append(", ".join([r['ticker'] for r in a_grade]))
        report.append("")
    
    if f_grade:
        report.append("### ❌ 需要修复的合约")
        report.append(", ".join([r['ticker'] for r in f_grade]))
        report.append("")
        report.append("建议:")
        report.append("1. 检查 CLC 数据源是否正确下载")
        report.append("2. 验证 ratio-adjustment 处理逻辑")
        report.append("3. 考虑使用原始合约数据 + 自定义展期")
        report.append("")
    
    return "\n".join(report)


def generate_csv(results: List[Dict]) -> pd.DataFrame:
    """生成 CSV 汇总"""
    rows = []
    for r in results:
        row = {
            'Ticker': r['ticker'],
            'Score': r['score'],
            'Grade': r['grade'],
            'Test_Rows': r['completeness']['test_rows'],
            'Coverage_Pct': r['completeness']['coverage_pct'],
            'Price_Min': r['price']['price_min'],
            'Price_Max': r['price']['price_max'],
            'Price_Ratio': r['price']['price_ratio'],
            'Return_Max_Pct': r['returns']['return_max_pct'],
            'Return_Min_Pct': r['returns']['return_min_pct'],
            'Gaps_>10%': r['returns']['gaps_count'].get('>10%', 0),
            'Gaps_>50%': r['returns']['gaps_count'].get('>50%', 0),
            'Max_Gap_Date': r['returns']['max_gap_date'],
            'Max_Gap_Pct': r['returns']['max_gap_value_pct'],
            'Roll_Issues': r['roll'].get('roll_issue_count', 0),
            'Roll_Issue_Pct': r['roll'].get('roll_issue_pct', 0),
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


# =============================================================================
# 主函数
# =============================================================================
def main():
    print("=" * 80)
    print("论文 50 合约数据质量检查")
    print("=" * 80)
    print()
    
    results = []
    
    for ticker in ALL_50_TICKERS:
        print(f"检查 {ticker}...", end=" ")
        
        # 加载数据
        df = load_clc_data(ticker)
        
        if df is None:
            # 无数据
            results.append({
                'ticker': ticker,
                'score': 0,
                'grade': 'F',
                'completeness': {'test_rows': 0, 'coverage_pct': 0, 'expected_test_days': 2268},
                'price': {'price_min': None, 'price_max': None, 'price_ratio': None},
                'returns': {'gaps_count': {}, 'max_gap_date': 'N/A', 'max_gap_value_pct': 0, 'return_max_pct': None, 'return_min_pct': None},
                'vol': {'vol_cv': None},
                'roll': {'roll_issue_count': 0, 'roll_issue_pct': 0},
            })
            print("❌ 无数据")
            continue
        
        # 各项检查
        completeness = check_data_completeness(df, ticker)
        price = check_price_reasonableness(df, ticker)
        returns = check_return_anomalies(df, ticker)
        vol = check_volatility_stability(df, ticker)
        roll = check_roll_artifacts(df, ticker)
        
        # 综合评分
        score, grade = compute_quality_score(completeness, price, returns, vol, roll)
        
        results.append({
            'ticker': ticker,
            'score': score,
            'grade': grade,
            'completeness': completeness,
            'price': price,
            'returns': returns,
            'vol': vol,
            'roll': roll,
        })
        
        print(f"评分={score}, 等级={grade}")
    
    print()
    print("=" * 80)
    print("生成报告...")
    
    # Markdown 报告
    report_md = generate_report(results)
    with open('data_quality_report_50contracts.md', 'w', encoding='utf-8') as f:
        f.write(report_md)
    print("✅ Markdown 报告：data_quality_report_50contracts.md")
    
    # CSV 汇总
    df_csv = generate_csv(results)
    df_csv.to_csv('data_quality_summary.csv', index=False)
    print("✅ CSV 汇总：data_quality_summary.csv")
    
    # 打印汇总
    print()
    print("=" * 80)
    print("汇总")
    print("=" * 80)
    
    grades = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
    for r in results:
        grades[r['grade']] += 1
    
    total = len(results)
    for grade, count in grades.items():
        pct = count / total * 100 if total > 0 else 0
        print(f"  等级 {grade}: {count} ({pct:.1f}%)")
    
    print()
    a_grade = [r['ticker'] for r in results if r['grade'] == 'A']
    if a_grade:
        print(f"✅ A 级合约：{', '.join(a_grade)}")
    
    f_grade = [r['ticker'] for r in results if r['grade'] == 'F']
    if f_grade:
        print(f"❌ F 级合约：{', '.join(f_grade)}")


if __name__ == '__main__':
    main()
