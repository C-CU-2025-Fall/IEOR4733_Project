"""Run Long Only strategy for all 4 asset classes and save results"""
import numpy as np
import pandas as pd
from data_loader import load_clc_full
from metrics import compute_metrics
from config import ASSET_CLASSES, BP, TRADING_DAYS, PAPER_TABLE3, METRIC_NAMES

EWMA_SPAN = 60
T = TRADING_DAYS
W0 = 1.0
SIGMA_TGT = 0.064

def load_contracts(ac_name, test_start='2011-01-01', test_end='2019-12-31'):
    tickers = ASSET_CLASSES.get(ac_name, [])
    raw = []
    for tk in tickers:
        df = load_clc_full(tk)
        if df is None or len(df) < 500:
            continue
        prices = df['Close'].values.astype(float)
        p0 = prices[0]
        norm_p = prices / p0
        rt = np.zeros(len(norm_p))
        rt[1:] = norm_p[1:] - norm_p[:-1]
        sigma = pd.Series(rt).ewm(span=EWMA_SPAN, adjust=False).std().values
        mask_s = df['Date'] >= test_start
        mask_e = df['Date'] <= test_end
        if not mask_s.any() or not mask_e.any():
            continue
        t0 = mask_s.idxmax()
        t1 = len(df) - 1 - mask_e[::-1].values.argmax()
        start = max(t0, 252)
        dates = df['Date'].iloc[start:t1].values
        raw.append({
            'tk': tk, 'rt': rt, 'sigma': sigma, 'norm_p': norm_p,
            'prices': prices, 'start': start, 't1': t1, 'dates': dates,
        })
    return raw

def compute_contract_returns(rd, sigma_tgt):
    rt, sigma, norm_p = rd['rt'], rd['sigma'], rd['norm_p']
    n = len(rt)
    pos = np.ones(n)  # Long Only
    Rt = np.zeros(n)
    for t in range(1, n):
        if sigma[t - 1] > 0 and (t < 2 or sigma[t - 2] > 0):
            a_prev = 1.0
            a_prev2 = 1.0 if t >= 2 else 0.0
            sp = a_prev * sigma_tgt / sigma[t - 1]
            spp = a_prev2 * sigma_tgt / sigma[t - 2] if t >= 2 else 0.0
            Rt[t] = sp * rt[t] - BP * norm_p[t - 1] * abs(sp - spp)
    return Rt

def compute_portfolio_returns(raw_data, sigma_tgt):
    series = []
    for rd in raw_data:
        Rt = compute_contract_returns(rd, sigma_tgt)
        start, t1, dates = rd['start'], rd['t1'], rd['dates']
        slc = Rt[start:t1]
        series.append(pd.Series(slc[:len(dates)], index=dates[:len(slc)]))
    return pd.DataFrame(series).T.dropna().mean(axis=1).values

# Run for all asset classes
results = []
for ac in ['Commodity', 'Equity Index', 'Fixed Income', 'Forex']:
    raw = load_contracts(ac)
    N = len(raw)
    if N == 0:
        continue
    
    R = compute_portfolio_returns(raw, SIGMA_TGT)
    m = compute_metrics(R, N)
    pv = PAPER_TABLE3[ac]['Long']
    
    pv_vals = [pv[k] for k in METRIC_NAMES]
    errs = [abs((m[i] - pv_vals[i]) / abs(pv_vals[i])) * 100 if pv_vals[i] != 0 else 0 for i in range(9)]
    
    results.append({
        'Asset_Class': ac,
        'N_Contracts': N,
        'E(R)': m[0], 'std(R)': m[1], 'DD': m[2], 'Sharpe': m[3],
        'Sortino': m[4], 'MDD': m[5], 'Calmar': m[6], '%_ve': m[7], 'Ave_P/L': m[8],
        'Paper_E(R)': pv['E(R)'], 'Paper_std': pv['std(R)'], 'Paper_Sharpe': pv['Sharpe'],
        'Err_E(R)': errs[0], 'Err_std': errs[1], 'Err_Sharpe': errs[3],
        'Contracts': ', '.join([rd['tk'] for rd in raw])
    })
    
    print(f"\n{'='*80}")
    print(f"  {ac} ({N} contracts)")
    print(f"{'='*80}")
    print(f"  Ours  : E(R)={m[0]:+.3f}  std={m[1]:.3f}  Sharpe={m[3]:+.3f}  MDD={m[5]:.3f}")
    print(f"  Paper : E(R)={pv['E(R)']:+.3f}  std={pv['std(R)']:.3f}  Sharpe={pv['Sharpe']:+.3f}  MDD={pv['MDD']:.3f}")
    print(f"  %Err  : E(R)={errs[0]:.1f}%  std={errs[1]:.1f}%  Sharpe={errs[3]:.1f}%")
    print(f"  Contracts: {', '.join([rd['tk'] for rd in raw])}")

# Save results
df = pd.DataFrame(results)
df.to_csv('tests/results/long_only_table3_results.csv', index=False)
print(f"\n\nResults saved to tests/results/long_only_table3_results.csv")
