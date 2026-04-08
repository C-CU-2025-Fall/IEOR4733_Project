"""
baseline_backtest.py — Baseline strategy backtest.

Equation 4 exact implementation:
  p_t → p_t / p_0  (p0-normalized, dimensionless)
  r_t = p_t - p_{t-1}  (additive on normalized prices)
  σ_t = EWMA(60) std of r_t
  R_t = A_{t-1} × (σ_tgt/σ_{t-1}) × r_t − bp × p_{t-1} × |Δ(scaled position)|

Portfolio: NAV-based (each contract NAV=100, sum NAVs, pct_change)
Metrics: ALL from metrics.py (single source of truth)
"""
import argparse
import numpy as np
import pandas as pd
from data_loader import load_clc_full, extract_test_period
from strategies import strategy_long_only, strategy_sign_r, strategy_macd
from vol_scaling import compute_ewma_vol
from metrics import compute_all_metrics
from config import (ASSET_CLASSES, BP, TRADING_DAYS,
                    SIGMA_TGT_DAILY, SIGMA_TGT_ANNUAL)

# Paper reference values
PAPER = {
    'Table 3': {
        'Equity Index': {
            'Long':    [0.504, 0.928, 0.606, 0.543, 0.831, 0.127, 0.466, 0.541, 0.928],
            'Sign(R)': [0.168, 0.799, 0.526, 0.211, 0.319, 0.299, 0.075, 0.528, 0.928],
            'MACD':    [-0.068, 0.586, 0.385, -0.117, -0.178, 0.351, -0.041, 0.519, 0.904],
        },
    },
    'Table 2': {
        'Equity Index': {
            'Long':    [0.668, 0.970, 0.606, 0.688, 1.102, 0.132, 0.509, 0.542, 0.948],
            'Sign(R)': [0.228, 0.966, 0.610, 0.236, 0.374, 0.344, 0.077, 0.528, 0.930],
            'MACD':    [0.016, 0.962, 0.618, 0.017, 0.027, 0.311, 0.006, 0.519, 0.927],
        },
    },
}

METRIC_ORDER = ['E(R)', 'std(R)', 'DD', 'Sharpe', 'Sortino', 'MDD', 'Calmar', '% +ve', 'Ave P/L']


# =============================================================================
# Portfolio construction
# =============================================================================

def build_portfolio(strategy_data):
    """NAV-based portfolio: each contract NAV=100, sum, pct_change.
    
    Returns:
        port_rets: 1D array of portfolio daily returns
        total_nav: pd.Series of total NAV (for MDD computation)
    """
    series_list = [pd.Series(r, index=d) for d, r in strategy_data]
    port_df = pd.DataFrame(series_list).T.dropna()
    navs = port_df.apply(lambda col: 100 * np.cumprod(1 + col))
    total_nav = navs.sum(axis=1)
    port_rets = total_nav.pct_change().dropna().values
    return port_rets, total_nav


# =============================================================================
# Per-contract return computation
# =============================================================================

def compute_returns_with_vol_scaling(ticker, strategy_pos):
    """Table 3: per-contract vol scaling (Equation 4).
    
    R_t = A_{t-1} × (σ_tgt/σ_{t-1}) × r_t − bp × p_{t-1} × |Δ(scaled pos)|
    """
    df = load_clc_full(ticker)
    if df is None:
        return None, None

    prices = df['Close'].values.astype(float)
    p0 = prices[0]
    norm_p = prices / p0                          # p0-normalized
    rt = np.diff(norm_p)                          # additive on normalized
    sigma = compute_ewma_vol(rt, span=60)         # EWMA(60) std

    t0, t1, _ = extract_test_period(df)
    if t0 is None:
        return None, None
    start = max(t0, 252)

    n = len(rt)
    Rt = np.zeros(n)
    for t in range(1, n):
        if sigma[t-1] > 0 and (t < 2 or sigma[t-2] > 0):
            sp = strategy_pos[t-1] * SIGMA_TGT_DAILY / sigma[t-1]
            spp = strategy_pos[t-2] * SIGMA_TGT_DAILY / sigma[t-2] if t >= 2 else 0
            Rt[t] = sp * rt[t] - BP * norm_p[t-1] * abs(sp - spp)

    dates = df['Date'].iloc[start:t1].values[:len(Rt[start:t1])]
    return dates, Rt[start:t1]


def compute_returns_raw(ticker, strategy_pos):
    """Table 2: raw returns without per-contract vol scaling.
    
    R_t = A_{t-1} × r_t − bp × p_{t-1} × |A_{t-1} − A_{t-2}|
    """
    df = load_clc_full(ticker)
    if df is None:
        return None, None

    prices = df['Close'].values.astype(float)
    p0 = prices[0]
    norm_p = prices / p0
    rt = np.diff(norm_p)

    t0, t1, _ = extract_test_period(df)
    if t0 is None:
        return None, None
    start = max(t0, 252)

    n = len(rt)
    rets = np.zeros(n)
    for t in range(1, n):
        rets[t] = strategy_pos[t-1] * rt[t] - BP * norm_p[t-1] * abs(strategy_pos[t-1] - strategy_pos[t-2])

    dates = df['Date'].iloc[start:t1].values[:len(rets[start:t1])]
    return dates, rets[start:t1]


# =============================================================================
# Printing
# =============================================================================

def print_table(title, strat_data_dict, paper_dict, n_contracts):
    """Print one table (Table 2 or Table 3) with ours vs paper comparison."""
    print(f"\n{'=' * 95}")
    print(f"  {title}")
    print(f"  Contracts: {n_contracts} | σ_tgt_annual={SIGMA_TGT_ANNUAL}")
    print(f"  p0-normalized | additive rt | NAV portfolio | MDD=rolling 252d on NAV")
    print(f"{'=' * 95}")

    for strat in ['Long', 'Sign(R)', 'MACD']:
        port_rets, total_nav = build_portfolio(strat_data_dict[strat])
        ours = compute_all_metrics(port_rets, total_nav)

        print(f"\n  {strat}:")
        print(f"  {'Metric':8s} | {'Ours':>8s} | {'Paper':>8s} | {'%Err':>6s} | OK?")
        print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*4}")

        if strat in paper_dict:
            pp = paper_dict[strat]
            for i, m in enumerate(METRIC_ORDER):
                o = ours[m]
                p = pp[i]
                e = abs((o - p) / abs(p)) * 100 if p != 0 else 0
                ok = '✅' if e < 15 else ('⚠️' if e < 30 else '❌')
                print(f"  {m:8s} | {o:>+8.3f} | {p:>+8.3f} | {e:>5.1f}% | {ok}")


# =============================================================================
# Main
# =============================================================================

def run_backtest(asset_class='Equity Index'):
    tickers = ASSET_CLASSES[asset_class]

    sd_t3 = {'Long': [], 'Sign(R)': [], 'MACD': []}
    sd_t2 = {'Long': [], 'Sign(R)': [], 'MACD': []}
    loaded = []

    for tk in tickers:
        df = load_clc_full(tk)
        if df is None:
            continue
        prices = df['Close'].values.astype(float)
        pct = np.diff(prices) / prices[:-1]

        for pos, key in [(strategy_long_only(len(pct)), 'Long'),
                         (strategy_sign_r(pct), 'Sign(R)'),
                         (strategy_macd(prices), 'MACD')]:
            # Table 3: with per-contract vol scaling
            d, r = compute_returns_with_vol_scaling(tk, pos)
            if d is not None:
                sd_t3[key].append((d, r))

            # Table 2: raw (portfolio-level vol applied later)
            d2, r2 = compute_returns_raw(tk, pos)
            if d2 is not None:
                sd_t2[key].append((d2, r2))
        loaded.append(tk)

    # ── Table 3 ──
    paper_t3 = PAPER['Table 3'].get(asset_class, {})
    print_table(f"TABLE 3 — {asset_class}", sd_t3, paper_t3, len(loaded))

    # ── Table 2: raw → NAV → portfolio-level vol scaling ──
    sd_t2_scaled = {}
    for strat in ['Long', 'Sign(R)', 'MACD']:
        port_rets_raw, total_nav_raw = build_portfolio(sd_t2[strat])
        ps = compute_ewma_vol(port_rets_raw, span=60)
        scaled_rets = np.zeros_like(port_rets_raw)
        for t in range(1, len(port_rets_raw)):
            if ps[t] > 0:
                scaled_rets[t] = SIGMA_TGT_DAILY / ps[t] * port_rets_raw[t]
        # Reconstruct scaled NAV for MDD
        dates = total_nav_raw.index[1:]  # skip first NaN from pct_change
        scaled_nav = pd.Series(
            len(loaded) * 100 * np.cumprod(1 + scaled_rets),
            index=dates
        )
        sd_t2_scaled[strat] = [(dates.values, scaled_rets)]

    paper_t2 = PAPER['Table 2'].get(asset_class, {})
    print_table(f"TABLE 2 — {asset_class}", sd_t2_scaled, paper_t2, len(loaded))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-class', default='Equity Index')
    args = parser.parse_args()
    run_backtest(args.asset_class)
