#!/usr/bin/env python3
"""
table2_table3_unified.py — Baseline reproduction (additive profits framework)

Paper: Zhang, Zohren, Roberts (2019) "Deep Reinforcement Learning for Trading"

Key decisions (documented in config.py):
  - r_t = p_t - p_{t-1}  (additive profits, paper Section 3.2)
  - σ_t = EWMA(60) std of r_t  (same units: price-diff/day)
  - σ_tgt = same units, constant across all contracts
  - σ_tgt/σ_t is dimensionless → normalises contracts to same daily vol
  - Cost = bp × p_{t-1} × |c_{t-1} − c_{t-2}|  (paper Formula 4, exact)
  - Table 3 = per-contract vol scaling only (Eq.4)
  - Table 2 = Table 3 + portfolio-level scaling (Eq.13)

Structure:
    config.py       — parameters, contract lists, paper targets
    data_loader.py  — CLC data loading
    strategies.py   — signal functions (Long, Sign(R), MACD)
    vol_scaling.py  — volatility scaling (per-contract + portfolio)
    metrics.py      — 9 individual metric functions
    table2_table3_unified.py — this file: orchestration + comparison
"""
import argparse
import math
import numpy as np
import pandas as pd

from config import (
    BP, SIGMA_TGT_DAILY, PORT_TGT_STD, MAX_LEVERAGE,
    ASSET_CLASSES, PAPER_TABLE3, PAPER_TABLE2, METRIC_NAMES,
    TEST_START, TEST_END,
)
from data_loader import load_clc_full, extract_test_period, get_price_diffs, get_pct_returns
from strategies import strategy_long_only, strategy_sign_r, strategy_macd
from vol_scaling import scale_per_contract, scale_portfolio
from metrics import compute_all_metrics


# =============================================================================
# Trade return computation — paper Formula 4 (additive profits)
# =============================================================================
def compute_trade_returns(price_diffs, prices, positions, bp=BP,
                          sigma_tgt_daily=SIGMA_TGT_DAILY):
    """
    Paper Formula 4:

      R_t = A_{t-1} × (σ_tgt / σ_{t-1}) × r_t
            − bp × p_{t-1} × |σ_tgt/σ_{t-1} × A_{t-1} − σ_tgt/σ_{t-2} × A_{t-2}|

    where:
      r_t = p_t − p_{t-1}           (additive profits)
      σ_t = EWMA(60) std of r_t     (price-diff/day units)
      σ_tgt = same units, constant  (normalises all contracts to same daily vol)
      σ_tgt / σ_t = dimensionless
      bp = 0.0020                    (20 bps)
    """
    n = len(prices)

    # Convert to percentage returns for consistent vol scaling across contracts
    pct_returns = np.zeros(n)
    pct_returns[1:] = price_diffs[1:] / prices[:-1]

    # Per-contract vol scaling: c_t = A_t × (σ_tgt / σ_t)
    # σ_tgt = 0.10/√252 (daily percentage vol target)
    scaling = scale_per_contract(pct_returns, sigma_tgt_daily, max_leverage=MAX_LEVERAGE)
    scaled = positions * scaling

    # R_t = c_{t-1} × r_t − bp × |c_{t-1} − c_{t-2}| (percentage framework)
    trade_rets = np.zeros(n)
    for t in range(2, n):
        trade_rets[t] = (scaled[t - 1] * pct_returns[t]
                         - bp * abs(scaled[t - 1] - scaled[t - 2]))

    return trade_rets


# =============================================================================
# Portfolio construction with date alignment
# =============================================================================
def build_portfolio(contract_data):
    """
    Equal-weight average of per-contract returns, aligned by date.
    """
    if not contract_data:
        return None, None

    dfs = []
    for dates, rets in contract_data:
        s = pd.Series(rets, index=dates, name='ret')
        dfs.append(s)

    merged = pd.concat(dfs, axis=1, join='inner').dropna()
    return merged.index, merged.mean(axis=1).values


# =============================================================================
# Display helpers
# =============================================================================
def status_icon(ours, paper, metric):
    if paper == 0:
        return '  ' if abs(ours) < 0.01 else '❌'
    pct = abs(ours - paper) / abs(paper) * 100
    if metric == 'std(R)':
        return '✅' if pct < 5 else '⚠️' if pct < 15 else '❌'
    elif metric in ('% +ve', 'Ave P/L'):
        return '✅' if pct < 10 else '⚠️' if pct < 25 else '❌'
    elif metric == 'MDD':
        return '✅' if pct < 30 else '⚠️' if pct < 60 else '❌'
    else:
        return '✅' if pct < 30 else '⚠️' if pct < 60 else '❌'


def print_comparison(ours_dict, paper_dict):
    print(f"    {'Metric':<10} {'Ours':>8}  {'Paper':>8}  {'Diff':>8}  {'%':>7}  Status")
    print(f"    {'-' * 60}")
    for mn in METRIC_NAMES:
        ov = ours_dict[mn]
        pv = paper_dict.get(mn)
        if pv is not None:
            diff = ov - pv
            pct = abs(diff / abs(pv)) * 100 if pv != 0 else 0
            s = status_icon(ov, pv, mn)
            print(f"    {mn:<10} {ov:>+8.3f}  {pv:>+8.3f}  {diff:>+8.3f}  {pct:>6.1f}%  {s}")
        else:
            print(f"    {mn:<10} {ov:>+8.3f}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Table 2/3 Baseline Reproduction')
    parser.add_argument('--portfolio', action='store_true',
                        help='Add portfolio-level vol scaling (Table 2)')
    parser.add_argument('--sigma-tgt', type=float, default=SIGMA_TGT_DAILY,
                        help=f'Daily σ_tgt in price-diff units (default: {SIGMA_TGT_DAILY})')
    parser.add_argument('--no-scaling', action='store_true',
                        help='Disable ALL vol scaling (debugging)')
    args = parser.parse_args()

    paper_targets = PAPER_TABLE2 if args.portfolio else PAPER_TABLE3
    table_name = "Table 2" if args.portfolio else "Table 3"

    print("=" * 100)
    if args.no_scaling:
        print(f"  {table_name} (DEBUG: NO vol scaling)")
    elif args.portfolio:
        print(f"  {table_name}: per-contract σ_tgt={args.sigma_tgt} daily "
              f"+ portfolio → std={PORT_TGT_STD}")
    else:
        print(f"  {table_name}: per-contract σ_tgt={args.sigma_tgt} daily")
    print(f"  r_t = p_t − p_{{t-1}} (additive) | cost = bp × p_{{t-1}} × |Δc|")
    print(f"  Test: {TEST_START}–{TEST_END} | BP={BP} | EWMA=60")
    print("=" * 100)

    for ac, tickers in ASSET_CLASSES.items():
        print(f"\n{'=' * 100}")
        print(f"  {ac} ({len(tickers)} contracts)")
        print(f"{'=' * 100}")

        strat_data = {'Long': [], 'Sign(R)': [], 'MACD': []}
        loaded = []

        for tk in tickers:
            df = load_clc_full(tk)
            if df is None:
                continue

            prices = df['Close'].values
            dates = df['Date']

            price_diffs = get_price_diffs(prices)
            pct_returns = get_pct_returns(prices)

            t0, t1, _ = extract_test_period(df)
            if t0 is None:
                continue

            pos_long = strategy_long_only(len(prices))
            pos_sign = strategy_sign_r(pct_returns)
            pos_macd = strategy_macd(prices)

            for pos, key in [(pos_long, 'Long'), (pos_sign, 'Sign(R)'),
                             (pos_macd, 'MACD')]:
                if args.no_scaling:
                    n = len(prices)
                    raw_rets = np.zeros(n)
                    for t in range(2, n):
                        raw_rets[t] = (pos[t-1] * price_diffs[t]
                                       - BP * abs(pos[t-1] - pos[t-2]))
                    start = max(t0, 252)
                    strat_data[key].append((dates.iloc[start:t1+1], raw_rets[start:t1+1]))
                else:
                    all_tr = compute_trade_returns(
                        price_diffs, prices, pos,
                        sigma_tgt_daily=args.sigma_tgt,
                    )
                    start = max(t0, 252)
                    strat_data[key].append((dates.iloc[start:t1+1], all_tr[start:t1+1]))

            loaded.append(tk)

        print(f"  Loaded: {len(loaded)}/{len(tickers)}")
        if not loaded:
            continue

        pp = paper_targets.get(ac, {})

        for strat in ['Long', 'Sign(R)', 'MACD']:
            if not strat_data[strat]:
                continue

            port_dates, port_raw = build_portfolio(strat_data[strat])
            if port_raw is None:
                continue

            if args.portfolio and not args.no_scaling:
                port_scaled = scale_portfolio(port_raw)
            else:
                port_scaled = port_raw

            metrics = compute_all_metrics(port_scaled)
            paper = pp.get(strat, {})
            print(f"\n  {strat}:")
            print_comparison(metrics, paper)


if __name__ == '__main__':
    main()
