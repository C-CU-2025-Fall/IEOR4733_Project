"""Table 2 vs Table 3 comparison: Long Only, Equity Index, additive framework."""
import numpy as np, pandas as pd
from data_loader import load_clc_full, extract_test_period
from strategies import strategy_long_only
from vol_scaling import compute_ewma_vol
from config import ASSET_CLASSES, BP, TRADING_DAYS

SIGMA = 0.0625  # best fit from calibration
T = TRADING_DAYS
W0 = 1.0
N_YEARS = 9

paper = {
    'Table 3': {'Long': [0.504, 0.928, 0.606, 0.543, 0.831, 0.127, 0.466, 0.541, 0.928]},
    'Table 2': {'Long': [0.668, 0.970, 0.606, 0.688, 1.102, 0.132, 0.509, 0.542, 0.948]},
}
METRIC = ['E(R)','std(R)','DD','Sharpe','Sortino','MDD','Calmar','% +ve','Ave P/L']

# Load all equity index contracts
raw = []
for tk in ASSET_CLASSES['Equity Index']:
    df = load_clc_full(tk)
    if df is None: continue
    prices = df['Close'].values.astype(float)
    p0 = prices[0]; norm_p = prices / p0
    rt = np.diff(norm_p)
    sigma = compute_ewma_vol(rt, span=60)
    t0, t1, _ = extract_test_period(df)
    if t0 is None: continue
    start = max(t0, 252)
    dates = df['Date'].iloc[start:t1].values
    pos = strategy_long_only(len(prices) - 1)
    raw.append({'rt': rt, 'sigma': sigma, 'norm_p': norm_p, 'pos': pos,
                'start': start, 't1': t1, 'dates': dates})

N = len(raw)
print(f"Contracts: {N}/10  |  σ_tgt_daily={SIGMA}  |  Additive cumsum framework\n")

# === Table 3: per-contract vol scaling (Equation 4) ===
series_t3 = []
for rd in raw:
    rt, sigma, norm_p, pos = rd['rt'], rd['sigma'], rd['norm_p'], rd['pos']
    start, dates = rd['start'], rd['dates']
    n = len(rt); Rt = np.zeros(n)
    for t in range(1, n):
        if sigma[t-1] > 0 and (t < 2 or sigma[t-2] > 0):
            sp = pos[t-1] * SIGMA / sigma[t-1]
            spp = pos[t-2] * SIGMA / sigma[t-2] if t >= 2 else 0
            Rt[t] = sp * rt[t] - BP * norm_p[t-1] * abs(sp - spp)
    series_t3.append(pd.Series(Rt[start:][:len(dates)], index=dates[:len(Rt[start:])]))

R_t3 = pd.DataFrame(series_t3).T.dropna().mean(axis=1).values

def metrics(R_eq, label):
    er = np.mean(R_eq) * T
    std = np.std(R_eq) * np.sqrt(T)
    dd = np.sqrt(np.mean(np.minimum(R_eq, 0)**2)) * np.sqrt(T)
    sharpe = er / std if std > 0 else 0
    sortino = er / dd if dd > 0 else 0
    pct_pos = np.sum(R_eq > 0) / len(R_eq)
    pos_r = R_eq[R_eq > 0]; neg_r = R_eq[R_eq < 0]
    avg_pl = np.mean(pos_r) / abs(np.mean(neg_r)) if len(pos_r) > 0 and len(neg_r) > 0 else 0
    cumret = np.cumsum(R_eq); wealth = N * W0 + cumret
    pk = np.maximum.accumulate(wealth)
    mdd = float(np.max((pk - wealth) / pk))
    realized_ann = (wealth[-1] - wealth[0]) / wealth[0] / N_YEARS
    calmar = realized_ann / mdd if mdd > 0 else 0
    return [round(v, 3) for v in [er, std, dd, sharpe, sortino, mdd, calmar, pct_pos, avg_pl]]

ours_t3 = metrics(R_t3, 'Table 3')

# === Table 2: add portfolio-level vol scaling ===
# Portfolio-level vol scaling: scale R_eq by σ_tgt_port / σ_t_port
# σ_t_port = EWMA(60) std of R_eq (the portfolio return series)
sigma_port = pd.Series(R_t3).ewm(span=60).std().values
sigma_tgt_port = SIGMA  # same target? Or target ~1.0?

# Actually, let me try: Table 2 scales the portfolio to target annual std ~ 1.0
# sigma_tgt_port_daily such that annualized ≈ 1.0 → daily ≈ 1/√252 ≈ 0.063
# But actually the paper says "portfolio-level volatility targeting"
# This means: take the portfolio returns (from Table 3), compute EWMA vol, scale to target

# Let's try σ_tgt_port that gives std ≈ 0.970 (Table 2 paper value)
# R_t2[t] = (σ_tgt_port / σ_port[t-1]) * R_t3[t]

for sigma_port_tgt in [0.0625, 0.064, 0.065, 0.066, 0.067, 0.068]:
    R_t2 = np.zeros(len(R_t3))
    for t in range(1, len(R_t3)):
        if sigma_port[t-1] > 0 and not np.isnan(sigma_port[t-1]):
            R_t2[t] = (sigma_port_tgt / sigma_port[t-1]) * R_t3[t]
    m = metrics(R_t2, f'σ_port={sigma_port_tgt}')
    if abs(m[1] - 0.970) < 0.005:  # close to paper std
        print(f"  σ_port_tgt={sigma_port_tgt} → std={m[1]:.3f} (paper=0.970)")
        ours_t2 = m
        break
else:
    # scan wider
    best = None; best_err = 999
    for sp in np.arange(0.060, 0.075, 0.001):
        R_t2 = np.zeros(len(R_t3))
        for t in range(1, len(R_t3)):
            if sigma_port[t-1] > 0 and not np.isnan(sigma_port[t-1]):
                R_t2[t] = (sp / sigma_port[t-1]) * R_t3[t]
        m = metrics(R_t2, '')
        err = abs(m[1] - 0.970)
        if err < best_err:
            best_err = err; best = (sp, m)
    sp, ours_t2 = best
    print(f"  Best σ_port_tgt={sp:.3f} → std={ours_t2[1]:.3f} (paper=0.970, err={best_err:.3f})")

# Print comparison
for table, ours in [('Table 3', ours_t3), ('Table 2', ours_t2)]:
    p = paper[table]['Long']
    print(f"\n{'='*80}")
    print(f"  {table} — Long Only — Equity Index")
    print(f"{'='*80}")
    print(f"  {'Metric':8s} | {'Ours':>8s} | {'Paper':>8s} | {'%Err':>6s} | OK?")
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*4}")
    n_ok = 0
    for i, m in enumerate(METRIC):
        o = ours[i]; pp = p[i]
        e = abs((o - pp) / abs(pp)) * 100 if pp != 0 else 0
        ok = '✅' if e < 15 else ('⚠️' if e < 30 else '❌')
        if e < 15: n_ok += 1
        print(f"  {m:8s} | {o:>+8.3f} | {pp:>+8.3f} | {e:>5.1f}% | {ok}")
    print(f"  → {n_ok}/9 ✅")
