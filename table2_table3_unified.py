#!/usr/bin/env python3
"""
table2_table3_unified.py — Modular baseline reproduction

Usage:
    python table2_table3_unified.py                         # Table 3 (no vol scaling)
    python table2_table3_unified.py --per-contract           # Table 3 + per-contract scaling
    python table2_table3_unified.py --per-contract --portfolio  # Table 2 (both layers)
    python table2_table3_unified.py --sigma-tgt 0.15         # custom σ_tgt (default from [27])

Structure:
    config.py       — parameters, contract lists, paper target values
    data_loader.py  — CLC data loading
    strategies.py   — signal functions (Long, Sign(R), MACD)
    vol_scaling.py  — volatility scaling (per-contract + portfolio)
    metrics.py      — 9 individual metric functions
    table2_table3_unified.py — this file: orchestration + comparison
"""
import argparse
import math
import numpy as np

from config import (
    BP, TRADING_DAYS, SIGMA_TGT_ANNUAL, PORT_TGT_STD,
    ASSET_CLASSES, PAPER_TABLE3, PAPER_TABLE2, METRIC_NAMES,
    TEST_START, TEST_END,
)
from data_loader import load_clc_full, extract_test_period, get_returns
from strategies import strategy_long_only, strategy_sign_r, strategy_macd
from vol_scaling import scale_per_contract, scale_portfolio
from metrics import compute_all_metrics


# =============================================================================
# Trade return computation — paper Formula 4
# =============================================================================
def compute_trade_returns(returns, prices, positions, bp=BP,
                          per_contract_sigma=None):
    """
    Compute per-contract trade returns.

    Paper Formula 4:
        R_t = c_{t-1} × r_t − bp × |c_{t-1} − c_{t-2}|

    where c_t = A_t × scaling_factor (if per-contract vol scaling applied)
          c_t = A_t              (if no per-contract vol scaling)

    Note on transaction cost:
        For Long Only, c_t changes slowly (only via σ_t drift), so cost ≈ 0.
        The paper does not special-case Long Only; the same formula applies.
        bp is applied to |position change|, not multiplied by price level,
        because returns are already in percentage terms.

    Args:
        returns:            daily percentage returns (full history)
        prices:             close prices (full history)
        positions:          signal positions A_t (full history)
        bp:                 transaction cost rate
        per_contract_sigma: annualised σ_tgt for per-contract scaling (None = no scaling)
    """
    n = len(returns)

    if per_contract_sigma is not None:
        # Per-contract vol scaling: c_t = A_t × (σ_tgt / σ_t_annualised)
        scaling = scale_per_contract(returns, per_contract_sigma)
        scaled = positions * scaling
    else:
        scaled = positions.copy()

    # R_t = c_{t-1} × r_t − bp × |c_{t-1} − c_{t-2}|
    trade_rets = np.zeros(n)
    for t in range(2, n):
        trade_rets[t] = scaled[t - 1] * returns[t] - bp * abs(scaled[t - 1] - scaled[t - 2])

    return trade_rets


# =============================================================================
# Portfolio construction
# =============================================================================
def build_portfolio(contract_returns_list):
    """Equal-weight average of per-contract returns."""
    min_len = min(len(r) for r in contract_returns_list)
    return np.mean([r[:min_len] for r in contract_returns_list], axis=0)


# =============================================================================
# Display helpers
# =============================================================================
def status_icon(ours, paper, metric):
    """Generate ✅/⚠️/❌ based on % difference."""
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
    """Print side-by-side comparison of 9 metrics."""
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
    parser.add_argument('--per-contract', action='store_true',
                        help='Apply per-contract volatility scaling')
    parser.add_argument('--portfolio', action='store_true',
                        help='Apply portfolio-level volatility scaling (Table 2)')
    parser.add_argument('--sigma-tgt', type=float, default=SIGMA_TGT_ANNUAL,
                        help=f'Annualised σ_tgt (default: {SIGMA_TGT_ANNUAL} from [27])')
    args = parser.parse_args()

    paper_targets = PAPER_TABLE2 if args.portfolio else PAPER_TABLE3
    table_name = "Table 2" if args.portfolio else "Table 3"

    # Config description
    parts = [table_name]
    if args.per_contract:
        parts.append(f"per-contract σ_tgt={args.sigma_tgt}")
    else:
        parts.append("no per-contract scaling")
    if args.portfolio:
        parts.append(f"portfolio → std={PORT_TGT_STD}")

    print("=" * 100)
    print(f"  {' | '.join(parts)}")
    print(f"  Data: CLC ratio-adjusted | Test: {TEST_START}–{TEST_END} | "
          f"BP={BP} | EWMA=60")
    print("=" * 100)

    for ac, tickers in ASSET_CLASSES.items():
        print(f"\n{'=' * 100}")
        print(f"  {ac} ({len(tickers)} contracts)")
        print(f"{'=' * 100}")

        strat_rets = {'Long': [], 'Sign(R)': [], 'MACD': []}
        loaded = []

        for tk in tickers:
            df = load_clc_full(tk)
            if df is None:
                continue

            prices = df['Close'].values
            returns = get_returns(prices)

            t0, t1 = extract_test_period(df)
            if t0 is None:
                continue

            # Compute positions on FULL history (warmup from 1988+)
            pos_long = strategy_long_only(len(prices))
            pos_sign = strategy_sign_r(returns)
            pos_macd = strategy_macd(prices)

            sigma = args.sigma_tgt if args.per_contract else None

            for pos, key in [(pos_long, 'Long'), (pos_sign, 'Sign(R)'),
                             (pos_macd, 'MACD')]:
                all_tr = compute_trade_returns(
                    returns, prices, pos,
                    per_contract_sigma=sigma,
                )
                # Extract test period only (skip first 252 days for warmup)
                start = max(t0, 252)
                strat_rets[key].append(all_tr[start:t1 + 1])

            loaded.append(tk)

        print(f"  Loaded: {len(loaded)}/{len(tickers)}")
        if not loaded:
            continue

        pp = paper_targets.get(ac, {})

        for strat in ['Long', 'Sign(R)', 'MACD']:
            if not strat_rets[strat]:
                continue

            port_raw = build_portfolio(strat_rets[strat])

            # Portfolio-level scaling (Table 2 only)
            if args.portfolio:
                port_scaled = scale_portfolio(port_raw)
            else:
                port_scaled = port_raw

            # Compute metrics on scaled returns
            metrics = compute_all_metrics(port_scaled)

            paper = pp.get(strat, {})
            print(f"\n  {strat}:")
            print_comparison(metrics, paper)


if __name__ == '__main__':
    main()
