"""
baseline_backtest.py — Baseline strategy backtest.

Two modes:
  A) Pct returns + per-contract vol scaling (σ_tgt/σ_t on pct returns)
  B) Pure pct returns, no vol scaling

All 9 metrics come from metrics.py (single source of truth).
Portfolio construction: NAV-based (each contract starts at 100, sum NAVs, pct_change).
"""
import argparse
import numpy as np
import pandas as pd
from data_loader import load_clc_full, get_pct_returns, extract_test_period
from strategies import strategy_long_only, strategy_sign_r, strategy_macd
from vol_scaling import compute_ewma_vol
from metrics import compute_all_metrics
from config import ASSET_CLASSES, BP, TRADING_DAYS

SIGMA_TGT_DAILY = 0.064

PAPER_T3 = {
    'Equity Index': {
        'Long':    [0.504, 0.928, 0.606, 0.543, 0.831, 0.127, 0.466, 0.541, 0.928],
        'Sign(R)': [0.168, 0.799, 0.526, 0.211, 0.319, 0.299, 0.075, 0.528, 0.928],
        'MACD':    [-0.068, 0.586, 0.385, -0.117, -0.178, 0.351, -0.041, 0.519, 0.904],
    },
}

# Order matches paper table columns
METRIC_ORDER = ['E(R)', 'std(R)', 'DD', 'Sharpe', 'Sortino', 'MDD', 'Calmar', '% +ve', 'Ave P/L']


def build_portfolio_returns(strategy_data):
    """Build NAV-based portfolio returns from per-contract return series."""
    all_s = [pd.Series(r, index=d) for d, r in strategy_data]
    port_df = pd.DataFrame(all_s).T.dropna()

    # Each contract starts at NAV=100, grow via cumprod
    navs = port_df.apply(lambda col: 100 * np.cumprod(1 + col))
    port_nav = navs.sum(axis=1)
    return port_nav.pct_change().dropna().values


def run_backtest(asset_class='Equity Index', mode='A'):
    tickers = ASSET_CLASSES[asset_class]
    paper = PAPER_T3[asset_class]

    sd = {'Long': [], 'Sign(R)': [], 'MACD': []}
    loaded = []

    for tk in tickers:
        df = load_clc_full(tk)
        if df is None:
            continue
        prices = df['Close'].values
        pct = get_pct_returns(prices)
        sigma = compute_ewma_vol(pct, span=60)
        t0, t1, _ = extract_test_period(df)
        if t0 is None:
            continue
        start = max(t0, 252)

        for pos, key in [(strategy_long_only(len(pct)), 'Long'),
                         (strategy_sign_r(pct), 'Sign(R)'),
                         (strategy_macd(prices), 'MACD')]:
            n = len(pct)
            rets = np.zeros(n)
            for t in range(2, n):
                if mode == 'A':
                    if sigma[t-1] <= 0 or sigma[t-2] <= 0:
                        continue
                    c = pos[t-1] * SIGMA_TGT_DAILY / sigma[t-1]
                    cp = pos[t-2] * SIGMA_TGT_DAILY / sigma[t-2]
                    rets[t] = c * pct[t] - BP * (prices[t-1] / prices[0]) * abs(c - cp)
                else:
                    rets[t] = pos[t-1] * pct[t] - BP * abs(pos[t-1] - pos[t-2])
            sd[key].append((df['Date'].iloc[start:t1+1], rets[start:t1+1]))
        loaded.append(tk)

    mode_label = 'A: pct + vol scaling' if mode == 'A' else 'B: pure pct, no vol scaling'

    print(f"\n{'='*90}")
    print(f"  Table 3 — {asset_class} ({len(loaded)}/{len(tickers)} contracts)")
    print(f"  {mode_label} | NAV-based portfolio")
    print(f"{'='*90}")

    for strat in ['Long', 'Sign(R)', 'MACD']:
        port_rets = build_portfolio_returns(sd[strat])
        ours = compute_all_metrics(port_rets)
        pp = paper[strat]

        print(f"\n  {strat}:")
        print(f"  {'Metric':8s} | {'Ours':>8s} | {'Paper':>8s} | {'%Err':>6s} | OK?")
        print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*4}")
        for i, m in enumerate(METRIC_ORDER):
            o = ours[m]
            p = pp[i]
            e = abs((o - p) / abs(p)) * 100 if p != 0 else 0
            ok = '✅' if e < 15 else ('⚠️' if e < 30 else '❌')
            print(f"  {m:8s} | {o:>+8.3f} | {p:>+8.3f} | {e:>5.1f}% | {ok}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-class', default='Equity Index')
    parser.add_argument('--mode', choices=['A', 'B'], default='A',
                        help='A=pct+vol scaling, B=pure pct')
    args = parser.parse_args()
    run_backtest(args.asset_class, args.mode)
