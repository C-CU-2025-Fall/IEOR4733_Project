#!/usr/bin/env python3
"""
baseline_run.py — Table 2 & Table 3 baseline reproduction

Usage:
    python baseline_run.py                    # Table 3 (per-contract vol scaling only)
    python baseline_run.py --portfolio        # Table 2 (+ portfolio-level vol scaling)
    python baseline_run.py --sigma-tgt 0.12   # Custom σ_tgt (annualized)
    python baseline_run.py --no-scaling       # Debug: no vol scaling at all

Paper: Zhang, Zohren, Roberts (2019) "Deep Reinforcement Learning for Trading"
References:
  [4] Baz et al. (2015) — MACD signal definition
  [27] Lim, Zohren, Roberts (2019) — Volatility scaling framework
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Import modular components
from config import (
    BP, TRADING_DAYS, EWMA_SPAN, SIGN_LOOKBACK,
    MACD_PAIRS, MACD_VOL_WINDOW, MACD_STD_WINDOW,
    TEST_START, TEST_END, WARMUP_DAYS,
    ASSET_CLASSES, PAPER_TABLE3, PAPER_TABLE2, METRIC_NAMES,
)
from data_loader import load_clc_full
from strategies import strategy_long_only, strategy_sign_r, strategy_macd
from vol_scaling import compute_vol_scaling, apply_portfolio_scaling
from metrics import (
    compute_expected_return, compute_annualized_std, compute_downside_deviation,
    compute_sharpe, compute_sortino, compute_max_drawdown, compute_calmar,
    compute_pct_positive, compute_avg_pl_ratio, compute_all_metrics,
)


def compute_trade_returns(prices, positions, sigma_tgt_annual, bp=BP):
    """
    Compute per-contract trade returns using paper's Formula 4.

    R_t = c_{t-1} × r_t − bp × p_{t-1} × |c_{t-1} − c_{t-2}|

    where:
        r_t = p_t - p_{t-1}              (additive profits)
        c_t = A_t × (σ_tgt / σ_t)         (scaled position)
        σ_t = EWMA(60) std of r_t         (same units as σ_tgt)

    Args:
        prices:            array of close prices
        positions:         array of signal positions (A_t)
        sigma_tgt_annual:  annualized volatility target (e.g., 0.10 = 10%)
        bp:                transaction cost rate (default 0.0020 = 20 bps)

    Returns:
        trade_rets: array of daily trade returns (in price-diff units)
    """
    n = len(prices)

    # Additive profits: r_t = p_t - p_{t-1}
    r_add = np.zeros(n)
    r_add[1:] = prices[1:] - prices[:-1]

    # Per-contract vol scaling: c_t = A_t × (σ_tgt / σ_t)
    scaling = compute_vol_scaling(r_add, sigma_tgt_annual, TRADING_DAYS, EWMA_SPAN)
    scaled_positions = positions * scaling

    # Trade returns with transaction cost
    trade_rets = np.zeros(n)
    for t in range(2, n):
        c_prev = scaled_positions[t - 1]
        c_prev2 = scaled_positions[t - 2]
        trade_rets[t] = c_prev * r_add[t] - bp * prices[t - 1] * abs(c_prev - c_prev2)

    return trade_rets


def build_aligned_portfolio(contract_data):
    """
    Build equal-weight portfolio from per-contract returns, aligned by date.

    Args:
        contract_data: list of (dates_series, returns_array) tuples

    Returns:
        port_dates: aligned dates index
        port_returns: equal-weight average returns
    """
    if not contract_data:
        return None, None

    series_list = []
    for dates, rets in contract_data:
        s = pd.Series(rets, index=dates, name='ret')
        series_list.append(s)

    # Inner join: only dates where all contracts have data
    merged = pd.concat(series_list, axis=1, join='inner').dropna()
    port_returns = merged.mean(axis=1).values

    return merged.index, port_returns


def print_comparison(ours_dict, paper_dict, strat_name):
    """Print metric comparison table against paper targets."""
    print(f"\n  {strat_name}:")
    print(f"    {'Metric':<10} {'Ours':>8}  {'Paper':>8}  {'Diff':>8}  {'%':>7}  Status")
    print(f"    {'-' * 62}")

    for mn in METRIC_NAMES:
        ov = ours_dict[mn]
        pv = paper_dict.get(mn)
        if pv is not None:
            diff = ov - pv
            pct = abs(diff / abs(pv)) * 100 if pv != 0 else 0

            # Status icon based on tolerance
            if mn == 'std(R)':
                status = '✅' if pct < 5 else '⚠️' if pct < 15 else '❌'
            elif mn in ('% +ve', 'Ave P/L'):
                status = '✅' if pct < 10 else '⚠️' if pct < 25 else '❌'
            elif mn == 'MDD':
                status = '✅' if pct < 30 else '⚠️' if pct < 60 else '❌'
            else:
                status = '✅' if pct < 30 else '⚠️' if pct < 60 else '❌'

            print(f"    {mn:<10} {ov:>+8.3f}  {pv:>+8.3f}  {diff:>+8.3f}  {pct:>6.1f}%  {status}")
        else:
            print(f"    {mn:<10} {ov:>+8.3f}")


def run_asset_class(ac_name, tickers, table_name, paper_targets,
                    do_per_contract_scaling, do_portfolio_scaling,
                    sigma_tgt_annual):
    """Run backtest for one asset class."""
    print(f"\n{'=' * 100}")
    print(f"  {ac_name} ({len(tickers)} contracts)")
    print(f"{'=' * 100}")

    strat_data = {'Long': [], 'Sign(R)': [], 'MACD': []}
    loaded = []

    for tk in tickers:
        df = load_clc_full(tk)
        if df is None:
            continue

        prices = df['Close'].values
        dates = df['Date']

        # Compute test period boundaries
        test_start_idx = dates[dates >= TEST_START].index[0] if len(dates[dates >= TEST_START]) > 0 else None
        test_end_idx = dates[dates <= TEST_END].index[-1] if len(dates[dates <= TEST_END]) > 0 else None

        if test_start_idx is None or test_end_idx is None:
            continue

        # Compute positions on FULL history (for warmup)
        pos_long = strategy_long_only(len(prices))
        pct_returns = prices[1:] / prices[:-1] - 1 if len(prices) > 1 else np.zeros(len(prices) - 1)
        pct_returns = np.insert(pct_returns, 0, 0)  # Align with prices
        pos_sign = strategy_sign_r(pct_returns, SIGN_LOOKBACK)
        pos_macd = strategy_macd(prices, MACD_PAIRS, MACD_VOL_WINDOW, MACD_STD_WINDOW)

        for pos, key in [(pos_long, 'Long'), (pos_sign, 'Sign(R)'), (pos_macd, 'MACD')]:
            if do_per_contract_scaling:
                trade_rets = compute_trade_returns(prices, pos, sigma_tgt_annual)
            else:
                # No scaling: raw signal returns (debug mode)
                n = len(prices)
                trade_rets = np.zeros(n)
                r_add = np.zeros(n)
                r_add[1:] = prices[1:] - prices[:-1]
                for t in range(2, n):
                    trade_rets[t] = pos[t - 1] * r_add[t] - BP * prices[t - 1] * abs(pos[t - 1] - pos[t - 2])

            # Extract test period only (skip first WARMUP_DAYS for indicator warmup)
            start_idx = max(test_start_idx, WARMUP_DAYS)
            strat_data[key].append((dates.iloc[start_idx:test_end_idx + 1],
                                    trade_rets[start_idx:test_end_idx + 1]))

        loaded.append(tk)

    print(f"  Loaded: {len(loaded)}/{len(tickers)} — {loaded}")
    if not loaded:
        return

    pp = paper_targets.get(ac_name, {})

    for strat in ['Long', 'Sign(R)', 'MACD']:
        if not strat_data[strat]:
            continue

        port_dates, port_raw = build_aligned_portfolio(strat_data[strat])
        if port_raw is None or len(port_raw) == 0:
            continue

        # Apply portfolio-level vol scaling if requested (Table 2)
        if do_portfolio_scaling:
            port_scaled = apply_portfolio_scaling(port_raw, TRADING_DAYS)
        else:
            port_scaled = port_raw.copy()

        # Compute all 9 metrics
        metrics = compute_all_metrics(port_scaled, port_raw, TRADING_DAYS)
        paper = pp.get(strat, {})
        print_comparison(metrics, paper, strat)


def main():
    parser = argparse.ArgumentParser(
        description='Table 2/3 Baseline Reproduction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python baseline_run.py                      # Table 3 (per-contract scaling only)
  python baseline_run.py --portfolio          # Table 2 (+ portfolio scaling)
  python baseline_run.py --sigma-tgt 0.15     # Use σ_tgt = 15% annualized
  python baseline_run.py --no-scaling         # Debug: raw signal, no scaling
        """
    )
    parser.add_argument('--portfolio', action='store_true',
                        help='Add portfolio-level vol scaling (Table 2)')
    parser.add_argument('--sigma-tgt', type=float, default=0.10,
                        help='Annualized σ_tgt (default: 0.10 = 10%)')
    parser.add_argument('--no-scaling', action='store_true',
                        help='Disable ALL vol scaling (debug mode)')
    args = parser.parse_args()

    # Determine which table we're reproducing
    do_per_contract = not args.no_scaling
    do_portfolio = args.portfolio and not args.no_scaling

    if args.no_scaling:
        table_name = "DEBUG (No Scaling)"
        paper_targets = PAPER_TABLE3  # Still compare against Table 3
    elif args.portfolio:
        table_name = "Table 2"
        paper_targets = PAPER_TABLE2
    else:
        table_name = "Table 3"
        paper_targets = PAPER_TABLE3

    # Print configuration header
    print("=" * 100)
    if args.no_scaling:
        print(f"  {table_name}")
        print(f"  r_t = p_t − p_{{t-1}} (additive) | NO vol scaling")
    elif args.portfolio:
        print(f"  {table_name}: per-contract σ_tgt={args.sigma_tgt:.3f} annualized")
        print(f"           + portfolio-level scaling → std≈{SIGMA_TGT_ANNUAL:.3f}")
    else:
        print(f"  {table_name}: per-contract σ_tgt={args.sigma_tgt:.3f} annualized")

    print(f"  cost = bp × p_{{t-1}} × |Δc|  (bp={BP})")
    print(f"  Test: {TEST_START} to {TEST_END} (warmup: {WARMUP_DAYS} days from 2010)")
    print(f"  References: [4] Baz et al. 2015 (MACD) | [27] Lim et al. 2019 (vol scaling)")
    print("=" * 100)

    # Run for each asset class
    for ac_name, tickers in ASSET_CLASSES.items():
        run_asset_class(
            ac_name, tickers, table_name, paper_targets,
            do_per_contract, do_portfolio, args.sigma_tgt
        )


if __name__ == '__main__':
    main()
