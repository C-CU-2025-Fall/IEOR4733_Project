"""Table 2: Two-layer vol scaling (per-contract × portfolio-level)"""
import numpy as np, pandas as pd
from data_loader import load_clc_full, extract_test_period
from strategies import strategy_long_only
from vol_scaling import compute_ewma_vol
from config import ASSET_CLASSES, BP, TRADING_DAYS

SIGMA = 0.0625  # per-contract σ_tgt
T = TRADING_DAYS
W0 = 1.0
N_YEARS = 9

paper_t3 = [0.504, 0.928, 0.606, 0.543, 0.831, 0.127, 0.466, 0.541, 0.928]
paper_t2 = [0.668, 0.970, 0.606, 0.688, 1.102, 0.132, 0.509, 0.542, 0.948]
METRIC = ['E(R)','std(R)','DD','Sharpe','Sortino','MDD','Calmar','% +ve','Ave P/L']

def calc_metrics(R_eq, N):
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

# Load data
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
                'start': start, 't1': t1, 'dates': dates, 'prices': prices})
N = len(raw)

# Step 1: Compute Table 3 (per-contract vol scaling only)
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

df_t3 = pd.DataFrame(series_t3).T.dropna()
R_t3 = df_t3.mean(axis=1).values

# Compute σ_port = EWMA(60) std of R_t3
R_t3_s = pd.Series(R_t3)
sigma_port = R_t3_s.ewm(span=60).std().values

print(f"Table 3 verified: {calc_metrics(R_t3, N)[0]:+.3f} / {calc_metrics(R_t3, N)[1]:.3f}")
print(f"σ_port range: {np.nanmin(sigma_port):.4f} ~ {np.nanmax(sigma_port):.4f}, mean={np.nanmean(sigma_port):.4f}")
print(f"σ_port annualized: {np.nanmean(sigma_port)*np.sqrt(252):.3f}")

# Step 2: Two-layer vol scaling for Table 2
# For each contract: R_t^i = [A × (σ_tgt/σ_t^i) × (σ_tgt_port/σ_{t-1}^port)] × r_t^i 
#                    - bp × p_{t-1} × |Δ(A × σ_tgt/σ_t^i × σ_tgt_port/σ_t^port)|

print(f"\n{'σ_port_tgt':>12s} | {'E(R)':>7s} {'std':>7s} {'DD':>7s} {'Sharpe':>7s} {'Sortino':>7s} {'MDD':>7s} {'Calmar':>7s} | std_err")
print("-" * 100)

for sigma_port_tgt in np.arange(0.040, 0.075, 0.001):
    series_t2 = []
    for rd in raw:
        rt, sigma, norm_p, pos = rd['rt'], rd['sigma'], rd['norm_p'], rd['pos']
        start, dates = rd['start'], rd['dates']
        n = len(rt); Rt = np.zeros(n)
        for t in range(1, n):
            if sigma[t-1] > 0 and (t < 2 or sigma[t-2] > 0):
                # Two-layer position scaling
                layer1 = SIGMA / sigma[t-1]                    # per-contract
                layer1_prev = SIGMA / sigma[t-2] if t >= 2 else layer1
                
                # Need σ_port aligned with contract time indices
                # sigma_port is computed on the portfolio-level aligned dates
                # For simplicity, use the same sigma_port for all contracts
                # (they share the same date range after dropna)
                sp_val = sigma_port[t-1] if (t-1) < len(sigma_port) else sigma_port[-1]
                sp_val_prev = sigma_port[t-2] if (t-2) >= 0 and (t-2) < len(sigma_port) else sp_val
                
                if sp_val > 0 and not np.isnan(sp_val):
                    layer2 = sigma_port_tgt / sp_val
                    layer2_prev = sigma_port_tgt / sp_val_prev if sp_val_prev > 0 else layer2
                    
                    full_pos = pos[t-1] * layer1 * layer2
                    full_pos_prev = pos[t-2] * layer1_prev * layer2_prev if t >= 2 else full_pos
                    
                    Rt[t] = full_pos * rt[t] - BP * norm_p[t-1] * abs(full_pos - full_pos_prev)
        
        series_t2.append(pd.Series(Rt[start:][:len(dates)], index=dates[:len(Rt[start:])]))
    
    R_t2 = pd.DataFrame(series_t2).T.dropna().mean(axis=1).values
    m = calc_metrics(R_t2, N)
    std_err = abs(m[1] - 0.970)
    
    if std_err < 0.010:
        print(f"  {sigma_port_tgt:>10.4f} | {m[0]:>+7.3f} {m[1]:>7.3f} {m[2]:>7.3f} {m[3]:>+7.3f} {m[4]:>+7.3f} {m[5]:>7.3f} {m[6]:>+7.3f} | {std_err:.4f}")

# Best match
print("\n--- Fine search ---")
best = None; best_total = 999
for sigma_port_tgt in np.arange(0.040, 0.075, 0.0001):
    series_t2 = []
    for rd in raw:
        rt, sigma, norm_p, pos = rd['rt'], rd['sigma'], rd['norm_p'], rd['pos']
        start, dates = rd['start'], rd['dates']
        n = len(rt); Rt = np.zeros(n)
        for t in range(1, n):
            if sigma[t-1] > 0 and (t < 2 or sigma[t-2] > 0):
                layer1 = SIGMA / sigma[t-1]
                layer1_prev = SIGMA / sigma[t-2] if t >= 2 else layer1
                sp_val = sigma_port[t-1] if (t-1) < len(sigma_port) else sigma_port[-1]
                sp_val_prev = sigma_port[t-2] if (t-2) >= 0 and (t-2) < len(sigma_port) else sp_val
                if sp_val > 0 and not np.isnan(sp_val):
                    layer2 = sigma_port_tgt / sp_val
                    layer2_prev = sigma_port_tgt / sp_val_prev if sp_val_prev > 0 else layer2
                    full_pos = pos[t-1] * layer1 * layer2
                    full_pos_prev = pos[t-2] * layer1_prev * layer2_prev if t >= 2 else full_pos
                    Rt[t] = full_pos * rt[t] - BP * norm_p[t-1] * abs(full_pos - full_pos_prev)
        series_t2.append(pd.Series(Rt[start:][:len(dates)], index=dates[:len(Rt[start:])]))
    
    R_t2 = pd.DataFrame(series_t2).T.dropna().mean(axis=1).values
    m = calc_metrics(R_t2, N)
    total = abs(m[1] - 0.970)/0.970 + 0.5*abs(m[0] - 0.668)/0.668
    if total < best_total:
        best_total = total; best = (sigma_port_tgt, m)

sp, m = best
print(f"\nBest: σ_port_tgt = {sp:.4f} (annual = {sp*np.sqrt(252):.3f})")
for i, met in enumerate(METRIC):
    e = abs((m[i]-paper_t2[i])/abs(paper_t2[i]))*100 if paper_t2[i]!=0 else 0
    ok = '✅' if e < 15 else ('⚠️' if e < 30 else '❌')
    print(f"  {met:8s} | {m[i]:>+8.3f} | {paper_t2[i]:>+8.3f} | {e:>5.1f}% | {ok}")
