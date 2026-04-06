"""
baseline_backtest.py — Baseline strategy backtest.

Equation 4 exact implementation:
  rt = pt - pt-1  (additive on p0-normalized prices, dimensionless)
  σ_t = EWMA(60) std of rt
  Rt = A_{t-1} × (σ_tgt/σ_{t-1}) × rt - bp × p_{t-1} × |Δ(scaled position)|

Portfolio: NAV-based (each contract NAV=100, sum NAVs, pct_change)
MDD: rolling 252-day max on total NAV
"""
import argparse
import numpy as np
import pandas as pd
from data_loader import load_clc_full, extract_test_period
from strategies import strategy_long_only, strategy_sign_r, strategy_macd
from vol_scaling import compute_ewma_vol
from config import (ASSET_CLASSES, BP, TRADING_DAYS, 
                    SIGMA_TGT_DAILY, SIGMA_TGT_ANNUAL)

PAPER_T3 = {
    'Equity Index': {
        'Long':    [0.504, 0.928, 0.606, 0.543, 0.831, 0.127, 0.466, 0.541, 0.928],
        'Sign(R)': [0.168, 0.799, 0.526, 0.211, 0.319, 0.299, 0.075, 0.528, 0.928],
        'MACD':    [-0.068, 0.586, 0.385, -0.117, -0.178, 0.351, -0.041, 0.519, 0.904],
    },
}
PAPER_T2 = {
    'Equity Index': {
        'Long':    [0.668, 0.970, 0.606, 0.688, 1.102, 0.132, 0.509, 0.542, 0.948],
        'Sign(R)': [0.228, 0.966, 0.610, 0.236, 0.374, 0.344, 0.077, 0.528, 0.930],
        'MACD':    [0.016, 0.962, 0.618, 0.017, 0.027, 0.311, 0.006, 0.519, 0.927],
    },
}

METRIC_ORDER = ['E(R)', 'std(R)', 'DD', 'Sharpe', 'Sortino', 'MDD', 'Calmar', '% +ve', 'Ave P/L']


def build_nav_and_rets(strategy_data):
    """Build NAV-based portfolio: each contract NAV=100, sum, pct_change."""
    all_s = [pd.Series(r, index=d) for d, r in strategy_data]
    port_df = pd.DataFrame(all_s).T.dropna()
    navs = port_df.apply(lambda col: 100 * np.cumprod(1 + col))
    total_nav = navs.sum(axis=1)
    port_rets = total_nav.pct_change().dropna().values
    return port_rets, total_nav


def compute_all_metrics(port_rets, total_nav):
    """Compute all 9 metrics. MDD = rolling 252-day max on NAV."""
    T = TRADING_DAYS
    er = np.mean(port_rets) * T
    std = np.std(port_rets) * np.sqrt(T)
    dd = np.sqrt(np.mean(np.minimum(port_rets, 0)**2)) * np.sqrt(T)
    sharpe = er / std if std > 0 else 0
    sortino = er / dd if dd > 0 else 0
    
    # MDD: rolling 252-day on NAV
    nav_vals = total_nav.values
    mdds = []
    for i in range(len(nav_vals) - 252 + 1):
        w = nav_vals[i:i+252]
        pk = np.maximum.accumulate(w)
        mdds.append(np.max((pk - w) / pk))
    mdd = max(mdds) if mdds else 0
    
    calmar = er / mdd if mdd > 0 else 0
    pct_pos = float(np.sum(port_rets > 0) / len(port_rets))
    pos = port_rets[port_rets > 0]; neg = port_rets[port_rets < 0]
    avg_pl = float(np.mean(pos) / abs(np.mean(neg))) if len(pos) > 0 and len(neg) > 0 else 0
    
    return {m: round(v, 3) for m, v in zip(
        METRIC_ORDER, [er, std, dd, sharpe, sortino, mdd, calmar, pct_pos, avg_pl]
    )}


def compute_per_contract_returns(ticker, strategy_pos):
    """Compute Rt per Equation 4 for one contract.
    
    rt = pt - pt-1 on p0-normalized prices
    σ_t = EWMA(60) std of rt
    Rt = scaled_pos * rt - bp * pt-1_norm * |Δ(scaled_pos)|
    """
    df = load_clc_full(ticker)
    if df is None:
        return None, None
    
    prices = df['Close'].values.astype(float)
    p0 = prices[0]
    norm_p = prices / p0              # p0-normalized (dimensionless)
    rt = np.diff(norm_p)              # additive on normalized (dimensionless)
    sigma = compute_ewma_vol(rt, span=60)
    
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


def compute_per_contract_returns_raw(ticker, strategy_pos):
    """Compute Rt WITHOUT per-contract vol scaling (for Table 2)."""
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
        pct = np.diff(prices) / prices[:-1]  # for sign_r signal
        
        for pos, key in [(strategy_long_only(len(pct)), 'Long'),
                         (strategy_sign_r(pct), 'Sign(R)'),
                         (strategy_macd(prices), 'MACD')]:
            # Table 3: with per-contract vol scaling
            d, r = compute_per_contract_returns(tk, pos)
            if d is not None:
                sd_t3[key].append((d, r))
            
            # Table 2: without per-contract vol scaling
            d2, r2 = compute_per_contract_returns_raw(tk, pos)
            if d2 is not None:
                sd_t2[key].append((d2, r2))
        loaded.append(tk)
    
    # ── TABLE 3 ──
    print(f"\n{'='*95}")
    print(f"  TABLE 3 — {asset_class} ({len(loaded)}/{len(tickers)} contracts)")
    print(f"  p0-norm, additive rt, per-contract vol(σ_annual={SIGMA_TGT_ANNUAL}), NAV portfolio")
    print(f"  MDD = rolling 252-day max on NAV")
    print(f"{'='*95}")
    
    paper3 = PAPER_T3.get(asset_class, {})
    for strat in ['Long', 'Sign(R)', 'MACD']:
        port_rets, total_nav = build_nav_and_rets(sd_t3[strat])
        ours = compute_all_metrics(port_rets, total_nav)
        
        print(f"\n  {strat}:")
        print(f"  {'Metric':8s} | {'Ours':>8s} | {'Paper':>8s} | {'%Err':>6s} | OK?")
        print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*4}")
        
        if strat in paper3:
            pp = paper3[strat]
            for i, m in enumerate(METRIC_ORDER):
                o = ours[m]; p = pp[i]
                e = abs((o - p) / abs(p)) * 100 if p != 0 else 0
                ok = '✅' if e < 15 else ('⚠️' if e < 30 else '❌')
                print(f"  {m:8s} | {o:>+8.3f} | {p:>+8.3f} | {e:>5.1f}% | {ok}")
    
    # ── TABLE 2 ──
    print(f"\n{'='*95}")
    print(f"  TABLE 2 — {asset_class} ({len(loaded)}/{len(tickers)} contracts)")
    print(f"  p0-norm, additive rt, no per-contract vol → NAV → portfolio-level vol(σ_annual={SIGMA_TGT_ANNUAL})")
    print(f"  MDD = rolling 252-day max on NAV")
    print(f"{'='*95}")
    
    paper2 = PAPER_T2.get(asset_class, {})
    for strat in ['Long', 'Sign(R)', 'MACD']:
        port_rets_raw, total_nav_raw = build_nav_and_rets(sd_t2[strat])
        
        # Portfolio-level vol scaling
        ps = compute_ewma_vol(port_rets_raw, span=60)
        scaled_rets = np.zeros_like(port_rets_raw)
        for t in range(1, len(port_rets_raw)):
            if ps[t] > 0:
                scaled_rets[t] = SIGMA_TGT_DAILY / ps[t] * port_rets_raw[t]
        
        scaled_nav = pd.Series(len(loaded) * 100 * np.cumprod(1 + scaled_rets))
        ours = compute_all_metrics(scaled_rets, scaled_nav)
        
        print(f"\n  {strat}:")
        print(f"  {'Metric':8s} | {'Ours':>8s} | {'Paper':>8s} | {'%Err':>6s} | OK?")
        print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*4}")
        
        if strat in paper2:
            pp = paper2[strat]
            for i, m in enumerate(METRIC_ORDER):
                o = ours[m]; p = pp[i]
                e = abs((o - p) / abs(p)) * 100 if p != 0 else 0
                ok = '✅' if e < 15 else ('⚠️' if e < 30 else '❌')
                print(f"  {m:8s} | {o:>+8.3f} | {p:>+8.3f} | {e:>5.1f}% | {ok}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-class', default='Equity Index')
    args = parser.parse_args()
    run_backtest(args.asset_class)
