"""
Test: MDD on additive wealth with different initial capital values.
Paper uses additive profits: r_t = p_t - p_{t-1}
Wealth accumulates: W_t = W_0 + cumsum(R_portfolio)
MDD = max(W_peak - W_trough) / W_peak
"""
import numpy as np
import pandas as pd
from data_loader import load_clc_full, extract_test_period
from strategies import strategy_long_only
from vol_scaling import compute_ewma_vol
from config import ASSET_CLASSES, BP, TRADING_DAYS

SIGMA_TGT = 0.064
T = TRADING_DAYS

# Get per-contract R_t for Long Only
per_contract = []
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
    per_contract.append((tk, dates, Rt[start:t1]))

print(f"Contracts: {len(per_contract)}/10")
print(f"σ_tgt_daily={SIGMA_TGT}  annual≈{SIGMA_TGT*np.sqrt(252):.3f}")

# Equal-weight average
series = [pd.Series(r, index=d) for _, d, r in per_contract]
df_all = pd.DataFrame(series).T.dropna()
R_eq = df_all.mean(axis=1).values

# Metrics on raw R_eq (additive profits)
er = np.mean(R_eq) * T
std = np.std(R_eq) * np.sqrt(T)
dd = np.sqrt(np.mean(np.minimum(R_eq, 0)**2)) * np.sqrt(T)
sharpe = er / std
sortino = er / dd
pct_pos = np.sum(R_eq > 0) / len(R_eq)
pos_r = R_eq[R_eq > 0]
neg_r = R_eq[R_eq < 0]
avg_pl = np.mean(pos_r) / abs(np.mean(neg_r))

print(f"\nRaw R_eq stats (what paper reports):")
print(f"  E(R)={er:+.3f}  std={std:.3f}  DD={dd:.3f}  Sharpe={sharpe:.3f}")
print(f"  Sortino={sortino:.3f}  %+ve={pct_pos:.3f}  AveP/L={avg_pl:.3f}")
print(f"  Paper: E(R)=+0.504  std=0.928  DD=0.606  Sharpe=0.543  Sortino=0.831  MDD=0.127")

# Cumulative PnL
cumret = np.cumsum(R_eq)
print(f"\n  Cumulative PnL range: {cumret.min():.3f} to {cumret.max():.3f}")
print(f"  Daily R_eq range: {R_eq.min():.4f} to {R_eq.max():.4f}")
print(f"  Daily R_eq mean: {R_eq.mean():.5f}  std: {R_eq.std():.5f}")

# Test MDD with different initial capitals (per contract)
print(f"\nMDD on additive wealth W_t = W_0 + cumsum(R_port) [equal-weight avg]")
print(f"{'W_0':>6s} {'W_0×10':>8s} {'MDD_full':>10s} {'MDD_roll252':>12s} {'Calmar':>8s}")
print("-" * 50)

for w0_per_contract in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
    w0_total = w0_per_contract * len(per_contract)
    wealth = w0_total + cumret  # additive wealth
    # Full period MDD
    pk = np.maximum.accumulate(wealth)
    mdd_full = float(np.max((pk - wealth) / pk))
    # Rolling 252d MDD
    mdds = []
    for i in range(len(wealth) - 252 + 1):
        w = wealth[i:i+252]
        p = np.maximum.accumulate(w)
        mdds.append(float(np.max((p - w) / p)))
    mdd_roll = max(mdds) if mdds else 0
    calmar = er / mdd_roll if mdd_roll > 0 else 0
    print(f"{w0_per_contract:>6.1f} {w0_total:>8.1f} {mdd_full:>10.3f} {mdd_roll:>12.3f} {calmar:>8.3f}")

# For comparison: cumprod MDD
wl = np.cumprod(1 + R_eq)
pk_c = np.maximum.accumulate(wl)
mdd_cumprod = float(np.max((pk_c - wl) / pk_c))
print(f"\n  cumprod(1+R) MDD: {mdd_cumprod:.3f}")

# What if MDD = max cumulative loss / cumulative peak PnL? (no initial capital)
pk_pnl = np.maximum.accumulate(cumret)
# Only where peak > 0
valid = pk_pnl > 0
if np.any(valid):
    dd_pnl = np.where(valid, (pk_pnl - cumret) / pk_pnl, 0)
    mdd_pnl = float(np.max(dd_pnl))
else:
    mdd_pnl = 0
print(f"  MDD on cumsum PnL only (no initial capital): {mdd_pnl:.3f}")

# What if MDD = max drawdown in absolute PnL / initial price (=1 for p0-normalized)?
max_dd_absolute = float(np.max(pk_pnl - cumret))
print(f"  Max absolute drawdown in PnL: {max_dd_absolute:.3f}")
print(f"  If divided by cumsum final value {cumret[-1]:.3f}: {max_dd_absolute/cumret[-1]:.3f}")
