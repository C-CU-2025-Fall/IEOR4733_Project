"""
Test: additive wealth (cumsum) vs multiplicative wealth (cumprod) for MDD
"""
import numpy as np
import pandas as pd
from data_loader import load_clc_full, extract_test_period
from strategies import strategy_long_only
from vol_scaling import compute_ewma_vol
from config import ASSET_CLASSES, BP, TRADING_DAYS

SIGMA_TGT = 0.064  # gives std ≈ 0.97 annual, matching paper

sd_additive = []
sd_multiplicative = []

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
    pos = strategy_long_only(len(pct))

    n = len(rt)
    Rt = np.zeros(n)
    for t in range(1, n):
        if sigma[t-1] > 0 and (t < 2 or sigma[t-2] > 0):
            sp = pos[t-1] * SIGMA_TGT / sigma[t-1]
            spp = pos[t-2] * SIGMA_TGT / sigma[t-2] if t >= 2 else 0
            Rt[t] = sp * rt[t] - BP * norm_p[t-1] * abs(sp - spp)

    dates = df['Date'].iloc[start:t1].values[:len(Rt[start:t1])]
    sd_additive.append((dates, Rt[start:t1]))
    sd_multiplicative.append((dates, Rt[start:t1]))

# Build portfolios
series_a = [pd.Series(r, index=d) for d, r in sd_additive]
port_df = pd.DataFrame(series_a).T.dropna()

# === Method 1: Multiplicative (cumprod) NAV ===
navs_mult = port_df.apply(lambda c: 100 * np.cumprod(1 + c))
total_nav_mult = navs_mult.sum(axis=1)
port_rets_mult = total_nav_mult.pct_change().dropna().values

# === Method 2: Additive (cumsum) wealth ===
# Each contract starts with wealth = 100
# Wealth_t = 100 + cumsum(R_t)
wealths_add = port_df.apply(lambda c: 100 + np.cumsum(c))
total_wealth_add = wealths_add.sum(axis=1)
# Portfolio "return" = change in total wealth / previous total wealth
port_rets_add = total_wealth_add.pct_change().dropna().values

# MDD for multiplicative NAV
nav_vals = total_nav_mult.values
mdds_mult = []
for i in range(len(nav_vals) - 252 + 1):
    w = nav_vals[i:i+252]
    pk = np.maximum.accumulate(w)
    mdds_mult.append(float(np.max((pk - w) / pk)))

# MDD for additive wealth
wealth_vals = total_wealth_add.values
mdds_add = []
for i in range(len(wealth_vals) - 252 + 1):
    w = wealth_vals[i:i+252]
    pk = np.maximum.accumulate(w)
    mdds_add.append(float(np.max((pk - w) / pk)))

# Full period MDD too
pk_mult = np.maximum.accumulate(nav_vals)
mdd_full_mult = float(np.max((pk_mult - nav_vals) / pk_mult))

pk_add = np.maximum.accumulate(wealth_vals)
mdd_full_add = float(np.max((pk_add - wealth_vals) / pk_add))

print("=" * 80)
print("LONG ONLY — Additive vs Multiplicative wealth comparison")
print(f"σ_tgt_daily = {SIGMA_TGT} (annual ≈ {SIGMA_TGT * np.sqrt(252):.3f})")
print("=" * 80)

T = TRADING_DAYS
for label, rets in [("Multiplicative (cumprod)", port_rets_mult),
                     ("Additive (cumsum)", port_rets_add)]:
    er = np.mean(rets) * T
    std = np.std(rets) * np.sqrt(T)
    dd = np.sqrt(np.mean(np.minimum(rets, 0)**2)) * np.sqrt(T)
    sharpe = er / std if std > 0 else 0
    pct_pos = np.sum(rets > 0) / len(rets)
    pos = rets[rets > 0]
    neg = rets[rets < 0]
    avg_pl = np.mean(pos) / abs(np.mean(neg)) if len(pos) > 0 and len(neg) > 0 else 0

    if "Multi" in label:
        mdd_roll = max(mdds_mult)
        mdd_full = mdd_full_mult
    else:
        mdd_roll = max(mdds_add)
        mdd_full = mdd_full_add

    sortino = er / dd if dd > 0 else 0
    calmar = er / mdd_roll if mdd_roll > 0 else 0

    print(f"\n  {label}:")
    print(f"  E(R)={er:+.3f}  std={std:.3f}  DD={dd:.3f}  Sharpe={sharpe:.3f}")
    print(f"  Sortino={sortino:.3f}  MDD(roll252)={mdd_roll:.3f}  MDD(full)={mdd_full:.3f}  Calmar={calmar:.3f}")
    print(f"  %+ve={pct_pos:.3f}  AveP/L={avg_pl:.3f}")

paper = [0.504, 0.928, 0.606, 0.543, 0.831, 0.127, 0.466, 0.541, 0.928]
print(f"\n  Paper Table 3 Long:")
print(f"  E(R)=+{paper[0]:.3f}  std={paper[1]:.3f}  DD={paper[2]:.3f}  Sharpe={paper[3]:.3f}")
print(f"  Sortino={paper[4]:.3f}  MDD={paper[5]:.3f}  Calmar={paper[6]:.3f}")
print(f"  %+ve={paper[7]:.3f}  AveP/L={paper[8]:.3f}")

print(f"\n  Total wealth (additive) range: {total_wealth_add.min():.1f} - {total_wealth_add.max():.1f}")
print(f"  Total NAV (multiplicative) range: {total_nav_mult.min():.1f} - {total_nav_mult.max():.1f}")
print(f"  Contracts: {len(sd_additive)}/10")
