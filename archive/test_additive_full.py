"""
Full test: additive wealth framework for all 3 strategies.
W_0 = 1 per contract (p0-normalized price starts at 1).
Wealth = N × W_0 + cumsum(R_portfolio)  [equal-weight avg]
MDD on additive wealth.
"""
import numpy as np
import pandas as pd
from data_loader import load_clc_full, extract_test_period
from strategies import strategy_long_only, strategy_sign_r, strategy_macd
from vol_scaling import compute_ewma_vol
from config import ASSET_CLASSES, BP, TRADING_DAYS

SIGMA_TGT = 0.064
T = TRADING_DAYS
W0 = 1.0  # per contract

T3_paper = {
    'Long':   [0.504, 0.928, 0.606, 0.543, 0.831, 0.127, 0.466, 0.541, 0.928],
    'Sign(R)': [0.168, 0.799, 0.526, 0.211, 0.319, 0.299, 0.075, 0.528, 0.928],
    'MACD':   [-0.068, 0.586, 0.385, -0.117, -0.178, 0.351, -0.041, 0.519, 0.904],
}

all_rets = {'Long': [], 'Sign(R)': [], 'MACD': []}

for tk in ASSET_CLASSES['Equity Index']:
    df = load_clc_full(tk)
    if df is None:
        continue
    prices = df['Close'].values.astype(float)
    p0 = prices[0]
    norm_p = prices / p0
    rt = np.diff(norm_p)
    sigma = compute_ewma_vol(rt, span=60)
    t0, t1, _ = extract_test_period(df)
    if t0 is None:
        continue
    start = max(t0, 252)
    pct = np.diff(prices) / prices[:-1]

    positions = {
        'Long': strategy_long_only(len(pct)),
        'Sign(R)': strategy_sign_r(pct),
        'MACD': strategy_macd(prices),
    }

    for strat_name, pos in positions.items():
        n = len(rt)
        Rt = np.zeros(n)
        for t in range(1, n):
            if sigma[t-1] > 0 and (t < 2 or sigma[t-2] > 0):
                sp = pos[t-1] * SIGMA_TGT / sigma[t-1]
                spp = pos[t-2] * SIGMA_TGT / sigma[t-2] if t >= 2 else 0
                Rt[t] = sp * rt[t] - BP * norm_p[t-1] * abs(sp - spp)
        dates = df['Date'].iloc[start:t1].values[:len(Rt[start:t1])]
        all_rets[strat_name].append((dates, Rt[start:t1]))

n_contracts = len(all_rets['Long'])
print(f"Contracts: {n_contracts}/10  |  σ_tgt_daily={SIGMA_TGT}  |  W_0={W0}/contract")
print(f"Framework: ADDITIVE (cumsum), NOT multiplicative (cumprod)")

METRIC_ORDER = ['E(R)', 'std(R)', 'DD', 'Sharpe', 'Sortino', 'MDD', 'Calmar', '% +ve', 'Ave P/L']

for strat in ['Long', 'Sign(R)', 'MACD']:
    # Equal-weight average of per-contract returns
    series = [pd.Series(r, index=d) for d, r in all_rets[strat]]
    df_all = pd.DataFrame(series).T.dropna()
    R_eq = df_all.mean(axis=1).values

    # Metrics on raw R_eq
    er = np.mean(R_eq) * T
    std = np.std(R_eq) * np.sqrt(T)
    dd = np.sqrt(np.mean(np.minimum(R_eq, 0)**2)) * np.sqrt(T)
    sharpe = er / std if std > 0 else 0
    sortino = er / dd if dd > 0 else 0
    pct_pos = np.sum(R_eq > 0) / len(R_eq)
    pos_r = R_eq[R_eq > 0]
    neg_r = R_eq[R_eq < 0]
    avg_pl = np.mean(pos_r) / abs(np.mean(neg_r)) if len(pos_r) > 0 and len(neg_r) > 0 else 0

    # MDD on additive wealth: W = N*W0 + cumsum(R_eq)
    cumret = np.cumsum(R_eq)
    wealth = n_contracts * W0 + cumret
    pk = np.maximum.accumulate(wealth)
    mdd = float(np.max((pk - wealth) / pk))

    # Rolling 252d MDD
    mdds = []
    for i in range(len(wealth) - 252 + 1):
        w = wealth[i:i+252]
        p = np.maximum.accumulate(w)
        mdds.append(float(np.max((p - w) / p)))
    mdd_roll = max(mdds) if mdds else 0

    calmar = er / mdd_roll if mdd_roll > 0 else 0
    calmar_full = er / mdd if mdd > 0 else 0

    ours = [round(v, 3) for v in [er, std, dd, sharpe, sortino, mdd_roll, calmar, pct_pos, avg_pl]]
    paper = T3_paper[strat]

    print(f"\n{'='*95}")
    print(f"  {strat} | Table 3 | Equity Index")
    print(f"{'='*95}")
    print(f"  {'Metric':8s} | {'Ours':>8s} | {'Paper':>8s} | {'%Err':>6s} | OK?")
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*5}")
    for i, m in enumerate(METRIC_ORDER):
        o = ours[i]
        p = paper[i]
        e = abs((o - p) / abs(p)) * 100 if p != 0 else 0
        ok = '✅' if e < 15 else ('⚠️' if e < 30 else '❌')
        print(f"  {m:8s} | {o:>+8.3f} | {p:>+8.3f} | {e:>5.1f}% | {ok}")

    print(f"  [MDD_full={mdd:.3f}  MDD_roll252={mdd_roll:.3f}  Calmar_full={calmar_full:.3f}]")
    print(f"  [CumPnL range: {cumret.min():.3f} to {cumret.max():.3f}  Wealth range: {wealth.min():.3f} to {wealth.max():.3f}]")
