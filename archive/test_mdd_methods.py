"""
Test different portfolio construction methods and MDD calculations.
Key question: how does the paper get std=0.93 AND MDD=0.127 simultaneously?
"""
import numpy as np
import pandas as pd
from data_loader import load_clc_full, extract_test_period
from strategies import strategy_long_only, strategy_sign_r, strategy_macd
from vol_scaling import compute_ewma_vol
from config import ASSET_CLASSES, BP, TRADING_DAYS

SIGMA_TGT = 0.064  # matches paper std ~ 0.93

T3_paper = {
    'Long':   [0.504, 0.928, 0.606, 0.543, 0.831, 0.127, 0.466, 0.541, 0.928],
    'Sign(R)': [0.168, 0.799, 0.526, 0.211, 0.319, 0.299, 0.075, 0.528, 0.928],
    'MACD':   [-0.068, 0.586, 0.385, -0.117, -0.178, 0.351, -0.041, 0.519, 0.904],
}

T = TRADING_DAYS


def compute_metrics(R, wealth=None, label=""):
    """Compute 9 metrics from daily returns R."""
    er = np.mean(R) * T
    std = np.std(R) * np.sqrt(T)
    dd = np.sqrt(np.mean(np.minimum(R, 0)**2)) * np.sqrt(T)
    sharpe = er / std if std > 0 else 0
    sortino = er / dd if dd > 0 else 0
    pct_pos = np.sum(R > 0) / len(R)
    pos_r = R[R > 0]
    neg_r = R[R < 0]
    avg_pl = np.mean(pos_r) / abs(np.mean(neg_r)) if len(pos_r) > 0 and len(neg_r) > 0 else 0

    # MDD
    if wealth is not None:
        wv = wealth.values if isinstance(wealth, pd.Series) else wealth
        mdds = []
        for i in range(len(wv) - 252 + 1):
            w = wv[i:i+252]
            pk = np.maximum.accumulate(w)
            mdds.append(float(np.max((pk - w) / pk)))
        mdd = max(mdds) if mdds else 0
    else:
        wl = np.cumprod(1 + R)
        pk = np.maximum.accumulate(wl)
        mdd = float(np.max((pk - wl) / pk))

    calmar = er / mdd if mdd > 0 else 0

    print(f"  {label}")
    print(f"    E(R)={er:+.3f}  std={std:.3f}  DD={dd:.3f}  Sharpe={sharpe:.3f}")
    print(f"    Sortino={sortino:.3f}  MDD={mdd:.3f}  Calmar={calmar:.3f}")
    print(f"    %+ve={pct_pos:.3f}  AveP/L={avg_pl:.3f}")
    return er, std, dd, sharpe, sortino, mdd, calmar, pct_pos, avg_pl


# Get per-contract returns
all_rets = {}
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

    for pos_fn, key in [(lambda n: strategy_long_only(n), 'Long'),
                        (lambda n: strategy_sign_r(n), 'Sign(R)'),  # needs pct
                        (lambda n: strategy_macd(prices), 'MACD')]:
        pass  # will handle below

    # Long Only
    pos = strategy_long_only(len(pct))
    n = len(rt)
    Rt = np.zeros(n)
    for t in range(1, n):
        if sigma[t-1] > 0 and (t < 2 or sigma[t-2] > 0):
            sp = pos[t-1] * SIGMA_TGT / sigma[t-1]
            spp = pos[t-2] * SIGMA_TGT / sigma[t-2] if t >= 2 else 0
            Rt[t] = sp * rt[t] - BP * norm_p[t-1] * abs(sp - spp)
    dates = df['Date'].iloc[start:t1].values[:len(Rt[start:t1])]
    all_rets.setdefault('Long', []).append((dates, Rt[start:t1]))

    # Sign(R)
    pos = strategy_sign_r(pct)
    Rt2 = np.zeros(n)
    for t in range(1, n):
        if sigma[t-1] > 0 and (t < 2 or sigma[t-2] > 0):
            sp = pos[t-1] * SIGMA_TGT / sigma[t-1]
            spp = pos[t-2] * SIGMA_TGT / sigma[t-2] if t >= 2 else 0
            Rt2[t] = sp * rt[t] - BP * norm_p[t-1] * abs(sp - spp)
    all_rets.setdefault('Sign(R)', []).append((dates, Rt2[start:t1]))

    # MACD
    pos = strategy_macd(prices)
    Rt3 = np.zeros(n)
    for t in range(1, n):
        if sigma[t-1] > 0 and (t < 2 or sigma[t-2] > 0):
            sp = pos[t-1] * SIGMA_TGT / sigma[t-1]
            spp = pos[t-2] * SIGMA_TGT / sigma[t-2] if t >= 2 else 0
            Rt3[t] = sp * rt[t] - BP * norm_p[t-1] * abs(sp - spp)
    all_rets.setdefault('MACD', []).append((dates, Rt3[start:t1]))

n_contracts = len(all_rets.get('Long', []))
print(f"Contracts: {n_contracts}/10")
print(f"σ_tgt_daily={SIGMA_TGT}  annual≈{SIGMA_TGT*np.sqrt(252):.3f}")
print()

for strat in ['Long', 'Sign(R)', 'MACD']:
    print("=" * 90)
    print(f"  {strat} — Paper: {T3_paper[strat]}")
    print("=" * 90)

    # Align dates
    series = [pd.Series(r, index=d) for d, r in all_rets[strat]]
    df_all = pd.DataFrame(series).T.dropna()

    # ── Method 1: Equal-weight average of returns ──
    R_eq = df_all.mean(axis=1).values
    # cumprod MDD
    wl = np.cumprod(1 + R_eq)
    pk = np.maximum.accumulate(wl)
    mdd_cumprod = float(np.max((pk - wl) / pk))
    # rolling 252d MDD on cumprod
    mdds = []
    for i in range(len(wl) - 252 + 1):
        w = wl[i:i+252]
        p = np.maximum.accumulate(w)
        mdds.append(float(np.max((p - w) / p)))
    mdd_roll_cumprod = max(mdds) if mdds else 0

    er = np.mean(R_eq) * T
    std = np.std(R_eq) * np.sqrt(T)
    dd = np.sqrt(np.mean(np.minimum(R_eq, 0)**2)) * np.sqrt(T)
    sharpe = er / std if std > 0 else 0
    sortino = er / dd if dd > 0 else 0
    calmar = er / mdd_roll_cumprod if mdd_roll_cumprod > 0 else 0
    pct_pos = np.sum(R_eq > 0) / len(R_eq)
    pos_r = R_eq[R_eq > 0]
    neg_r = R_eq[R_eq < 0]
    avg_pl = np.mean(pos_r) / abs(np.mean(neg_r)) if len(pos_r) > 0 and len(neg_r) > 0 else 0
    print(f"  Equal-weight avg + cumprod(1+R) MDD:")
    print(f"    E(R)={er:+.3f}  std={std:.3f}  DD={dd:.3f}  Sharpe={sharpe:.3f}")
    print(f"    Sortino={sortino:.3f}  MDD_full={mdd_cumprod:.3f}  MDD_roll252={mdd_roll_cumprod:.3f}  Calmar={calmar:.3f}")
    print(f"    %+ve={pct_pos:.3f}  AveP/L={avg_pl:.3f}")

    # ── Method 2: Equal-weight avg + cumsum(R) + MDD on cumulative PnL ──
    cumret = np.cumsum(R_eq)
    # MDD on cumsum starting from 0
    pk2 = np.maximum.accumulate(cumret)
    # Only compute MDD where peak > 0
    valid = pk2 > 0
    if np.any(valid):
        dd_vals = np.where(valid, (pk2 - cumret) / pk2, 0)
        mdd_cumsum_full = float(np.max(dd_vals))
    else:
        mdd_cumsum_full = 0
    # rolling 252d
    mdds2 = []
    for i in range(len(cumret) - 252 + 1):
        w = cumret[i:i+252]
        p = np.maximum.accumulate(w)
        valid_w = p > 0
        if np.any(valid_w):
            dd_w = np.where(valid_w, (p - w) / p, 0)
            mdds2.append(float(np.max(dd_w)))
        else:
            mdds2.append(0)
    mdd_roll_cumsum = max(mdds2) if mdds2 else 0
    print(f"  Equal-weight avg + cumsum(R) MDD (peak/trough on PnL):")
    print(f"    MDD_full={mdd_cumsum_full:.3f}  MDD_roll252={mdd_roll_cumsum:.3f}")

    # ── Method 3: NAV portfolio + cumprod ──
    navs = df_all.apply(lambda c: 100 * np.cumprod(1 + c))
    total_nav = navs.sum(axis=1)
    R_nav = total_nav.pct_change().dropna().values
    nav_vals = total_nav.values
    mdds3 = []
    for i in range(len(nav_vals) - 252 + 1):
        w = nav_vals[i:i+252]
        p = np.maximum.accumulate(w)
        mdds3.append(float(np.max((p - w) / p)))
    mdd_nav = max(mdds3) if mdds3 else 0
    er_nav = np.mean(R_nav) * T
    std_nav = np.std(R_nav) * np.sqrt(T)
    print(f"  NAV portfolio (cumprod per contract, sum, pct_change):")
    print(f"    E(R)={er_nav:+.3f}  std={std_nav:.3f}  MDD_roll252={mdd_nav:.3f}")

    # ── Method 4: Per-contract MDD, then average ──
    per_mdds = []
    for col in df_all.columns:
        rc = df_all[col].values
        wl_c = np.cumprod(1 + rc)
        pk_c = np.maximum.accumulate(wl_c)
        mdd_c = float(np.max((pk_c - wl_c) / pk_c))
        per_mdds.append(mdd_c)
    print(f"  Per-contract cumprod MDD (avg): {np.mean(per_mdds):.3f}  individual: {[round(m,3) for m in per_mdds]}")

    # ── Method 5: Per-contract cumsum MDD, then average ──
    per_mdds2 = []
    for col in df_all.columns:
        rc = df_all[col].values
        cs = np.cumsum(rc)
        pk_cs = np.maximum.accumulate(cs)
        valid_cs = pk_cs > 0
        if np.any(valid_cs):
            dd_cs = np.where(valid_cs, (pk_cs - cs) / pk_cs, 0)
            per_mdds2.append(float(np.max(dd_cs)))
        else:
            per_mdds2.append(0)
    print(f"  Per-contract cumsum MDD (avg): {np.mean(per_mdds2):.3f}  individual: {[round(m,3) for m in per_mdds2]}")

    print()
