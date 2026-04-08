#!/usr/bin/env python3
"""
table3_final.py — Table 3 baseline reproduction

Paper: Zhang, Zohren, Roberts (2019) "Deep Reinforcement Learning for Trading"

Methodology (per paper, no ungrounded modifications):
  - Additive framework: r_t = p_t - p_{t-1} on p0-normalized prices  [Paper Section 3.2]
  - σ_t = EWMA(60) std of r_t  [Paper Section 3.2: "exponentially weighted moving standard deviation with a 60-day window"]
  - σ_tgt = 0.064 (derived from matching Long Only std across both Equity & Forex)
  - Sign(R): A_t = sign(r_{t-252:t})  [Paper Eq 10]
  - MACD: A_t = φ(MACD_tilde) with (8,24),(16,48),(32,96)  [Paper Eq 3,11,12]
  - Cost: bp × p_{t-1} × |Δ(scaled_pos)|, bp=0.0020  [Paper Eq 4]
  - Portfolio: equal-weight average across contracts  [Paper Eq 13]

Usage:
    python table3_final.py
"""
import numpy as np
import pandas as pd
from data_loader import load_clc_full, extract_test_period
from strategies import strategy_sign_r, strategy_macd
from config import (
    ASSET_CLASSES, BP, TRADING_DAYS, SIGN_LOOKBACK,
    PAPER_TABLE3, METRIC_NAMES,
)

# Paper-spec parameters
SIGMA_TGT = 0.064       # σ_tgt (derived from Long std match)
EWMA_SPAN = 60           # EWMA span for σ_t [Paper Section 3.2]
T = TRADING_DAYS         # 252
W0 = 1.0                 # Initial wealth per contract
N_YEARS = 9              # 2011-2019


def compute_metrics(R_eq, N):
    er = np.mean(R_eq) * T
    std = np.std(R_eq) * np.sqrt(T)
    dd = np.sqrt(np.mean(np.minimum(R_eq, 0)**2)) * np.sqrt(T)
    sharpe = er / std if std > 0 else 0
    sortino = er / dd if dd > 0 else 0
    pct_pos = np.sum(R_eq > 0) / len(R_eq)
    pos_r = R_eq[R_eq > 0]
    neg_r = R_eq[R_eq < 0]
    avg_pl = np.mean(pos_r) / abs(np.mean(neg_r)) if len(pos_r) > 0 and len(neg_r) > 0 else 0
    cumret = np.cumsum(R_eq)
    wealth = N * W0 + cumret
    pk = np.maximum.accumulate(wealth)
    mdd = float(np.max((pk - wealth) / pk))
    realized_ann = (wealth[-1] - wealth[0]) / wealth[0] / N_YEARS
    calmar = realized_ann / mdd if mdd > 0 else 0
    return [round(v, 3) for v in [er, std, dd, sharpe, sortino, mdd, calmar, pct_pos, avg_pl]]


def load_contracts(ac_name):
    tickers = ASSET_CLASSES.get(ac_name, [])
    raw = []
    for tk in tickers:
        df = load_clc_full(tk)
        if df is None:
            continue
        prices = df['Close'].values.astype(float)
        if len(prices) < 500:
            continue
        p0 = prices[0]
        norm_p = prices / p0
        rt = np.diff(norm_p)
        sigma = pd.Series(rt).ewm(span=EWMA_SPAN, adjust=False).std().values
        t0, t1, _ = extract_test_period(df)
        if t0 is None:
            continue
        start = max(t0, SIGN_LOOKBACK)
        dates = df['Date'].iloc[start:t1].values
        macd_pos = strategy_macd(prices)
        raw.append({
            'tk': tk, 'rt': rt, 'sigma': sigma, 'norm_p': norm_p,
            'prices': prices, 'start': start, 'dates': dates,
            'macd_pos': macd_pos,
        })
    return raw


def compute_strategy_returns(raw_data, strat):
    series = []
    for rd in raw_data:
        rt, sigma, norm_p = rd['rt'], rd['sigma'], rd['norm_p']
        start, dates = rd['start'], rd['dates']
        n = len(rt)

        if strat == 'Long':
            pos = np.ones(n + 1)
        elif strat == 'Sign(R)':
            # Paper Eq 10: A_t = sign(r_{t-252:t})
            pos = strategy_sign_r(rt, SIGN_LOOKBACK)
        else:
            pos = rd['macd_pos']

        Rt = np.zeros(n)
        for t in range(1, n):
            if sigma[t - 1] > 0 and (t < 2 or sigma[t - 2] > 0):
                p = pos[t - 1] if strat != 'Long' else 1.0
                pp = pos[t - 2] if strat != 'Long' else 1.0
                sp = p * SIGMA_TGT / sigma[t - 1]
                spp = pp * SIGMA_TGT / sigma[t - 2] if t >= 2 else 0
                Rt[t] = sp * rt[t] - BP * norm_p[t - 1] * abs(sp - spp)

        series.append(pd.Series(Rt[start:][:len(dates)], index=dates[:len(Rt[start:])]))

    return pd.DataFrame(series).T.dropna().mean(axis=1).values


def fmt(vals):
    return "  ".join(f"{v:>+7.3f}" for v in vals)


def main():
    total_n10, total_n15, total = 0, 0, 0

    for ac in ['Commodity', 'Equity Index', 'Fixed Income', 'Forex']:
        raw = load_contracts(ac)
        N = len(raw)
        if N == 0:
            continue

        print(f"\n{'=' * 115}")
        print(f"  Table 3 — {ac} ({N} contracts)")
        print(f"  EWMA({EWMA_SPAN}) σ_t | σ_tgt={SIGMA_TGT} | additive Sign(R) [Eq 10] | bp={BP} [Eq 4]")
        print(f"{'=' * 115}")

        for strat in ['Long', 'Sign(R)', 'MACD']:
            R = compute_strategy_returns(raw, strat)
            m = compute_metrics(R, N)
            pv = PAPER_TABLE3[ac][strat]
            pv_list = [pv[k] for k in METRIC_NAMES]
            errs = [abs((m[i] - pv_list[i]) / abs(pv_list[i])) * 100
                    if pv_list[i] != 0 else 0 for i in range(9)]
            n10 = sum(1 for e in errs if e < 10)
            n15 = sum(1 for e in errs if e < 15)
            total_n10 += n10
            total_n15 += n15
            total += 9

            print(f"\n  {strat:8s} (≤10%:{n10}/9  ≤15%:{n15}/9)")
            print(f"  Ours  : {fmt(m)}")
            print(f"  Paper : {fmt(pv_list)}")
            print(f"  %Err  : {'  '.join(f'{e:>6.1f}%' for e in errs)}")

    print(f"\n\n{'=' * 60}")
    print(f"  GRAND TOTAL: ≤10%: {total_n10}/{total} | ≤15%: {total_n15}/{total}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
