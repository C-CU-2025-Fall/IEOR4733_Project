"""
baseline_backtest.py — Baseline strategy backtest.

Two modes:
  A) Pct returns + per-contract vol scaling (σ_tgt/σ_t on pct returns)
  B) Pure pct returns, no vol scaling

Metrics use standard formulas:
  - DD = sqrt(1/n * sum(min(0, R_i)^2)), annualized
  - MDD = rolling 252-day max drawdown, averaged
  - Sortino = E(R) / DD
  - Calmar = E(R) / MDD
"""
import argparse
import numpy as np
import pandas as pd
from data_loader import load_clc_full, get_pct_returns, extract_test_period
from strategies import strategy_long_only, strategy_sign_r, strategy_macd
from vol_scaling import compute_ewma_vol
from config import ASSET_CLASSES, BP, TRADING_DAYS

TRADING = TRADING_DAYS
SIGMA_TGT_DAILY = 0.064


# ── Metrics ──────────────────────────────────────────────────────────────

def calc_er(r):
    return np.mean(r) * TRADING

def calc_std(r):
    return np.std(r) * np.sqrt(TRADING)

def calc_dd(r):
    """Downside deviation: sqrt(1/n * sum(min(0, R)^2)), annualized."""
    return np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * np.sqrt(TRADING)

def calc_sharpe(r):
    s = calc_std(r)
    return calc_er(r) / s if s > 0 else 0

def calc_sortino(r):
    d = calc_dd(r)
    return calc_er(r) / d if d > 0 else 0

def calc_mdd(r, window=252):
    """Rolling MDD: average of max drawdown over rolling windows."""
    mdds = []
    for i in range(len(r) - window + 1):
        w = r[i:i + window]
        wealth = np.cumprod(1 + w)
        peak = np.maximum.accumulate(wealth)
        dd = (peak - wealth) / peak
        mdds.append(float(np.max(dd)))
    return float(np.mean(mdds)) if mdds else 0.0

def calc_calmar(r):
    m = calc_mdd(r)
    return calc_er(r) / m if m > 0 else 0

def calc_pct_pos(r):
    return np.sum(r > 0) / len(r)

def calc_avg_pl(r):
    pos = r[r > 0]; neg = r[r < 0]
    if len(pos) > 0 and len(neg) > 0:
        return np.mean(pos) / abs(np.mean(neg))
    return 0.0

METRIC_NAMES = ['E(R)', 'std(R)', 'DD', 'Sharpe', 'Sortino', 'MDD', 'Calmar', '%+ve', 'Ave P/L']

def compute_all(r):
    return [calc_er(r), calc_std(r), calc_dd(r), calc_sharpe(r), calc_sortino(r),
            calc_mdd(r), calc_calmar(r), calc_pct_pos(r), calc_avg_pl(r)]


# ── Paper targets ────────────────────────────────────────────────────────

PAPER_T3 = {
    'Equity Index': {
        'Long':    [0.504, 0.928, 0.606, 0.543, 0.831, 0.127, 0.466, 0.541, 0.928],
        'Sign(R)': [0.168, 0.799, 0.526, 0.211, 0.319, 0.299, 0.075, 0.528, 0.928],
        'MACD':    [-0.068, 0.586, 0.385, -0.117, -0.178, 0.351, -0.041, 0.519, 0.904],
    },
}


# ── Backtest ─────────────────────────────────────────────────────────────

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
                    # Per-contract vol scaling on pct returns
                    if sigma[t-1] <= 0 or sigma[t-2] <= 0:
                        continue
                    c = pos[t-1] * SIGMA_TGT_DAILY / sigma[t-1]
                    cp = pos[t-2] * SIGMA_TGT_DAILY / sigma[t-2]
                    rets[t] = c * pct[t] - BP * (prices[t-1] / prices[0]) * abs(c - cp)
                else:
                    # Pure pct, no vol scaling
                    rets[t] = pos[t-1] * pct[t] - BP * abs(pos[t-1] - pos[t-2])
            sd[key].append((df['Date'].iloc[start:t1+1], rets[start:t1+1]))
        loaded.append(tk)

    mode_label = 'A: pct + vol scaling' if mode == 'A' else 'B: pure pct, no vol scaling'

    print(f"\n{'='*90}")
    print(f"  Table 3 — {asset_class} ({len(loaded)}/{len(tickers)} contracts)")
    print(f"  {mode_label}")
    print(f"{'='*90}")

    for strat in ['Long', 'Sign(R)', 'MACD']:
        all_s = [pd.Series(r, index=d) for d, r in sd[strat]]
        port_rets = pd.DataFrame(all_s).T.mean(axis=1).dropna().values

        ours = compute_all(port_rets)
        pp = paper[strat]

        print(f"\n  {strat}:")
        print(f"  {'Metric':8s} | {'Ours':>8s} | {'Paper':>8s} | {'%Err':>6s} | OK?")
        print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*4}")
        for i, m in enumerate(METRIC_NAMES):
            o, p = ours[i], pp[i]
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
