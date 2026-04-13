"""
Run full baseline replication for all 4 asset classes × 3 strategies
Generate complete alignment table with all 9 metrics
"""
import numpy as np
import pandas as pd
from data_loader import load_clc_full
from strategies import strategy_sign_r, strategy_macd
from metrics import compute_metrics
from config import ASSET_CLASSES, BP, TRADING_DAYS, PAPER_TABLE3, METRIC_NAMES

EWMA_SPAN = 60
T = TRADING_DAYS
SIGMA_TGT = 0.064  # Per-contract vol scaling

def load_contracts(ac_name, test_start='2011-01-01', test_end='2019-12-31'):
    if ac_name == 'All':
        # Combine all contracts from all asset classes
        tickers = []
        for ac_contracts in ASSET_CLASSES.values():
            tickers.extend(ac_contracts)
    else:
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
            'macd_pos': strategy_macd(norm_p),
        })
    return raw

def compute_contract_returns(rd, strat, sigma_tgt):
    rt, sigma, norm_p = rd['rt'], rd['sigma'], rd['norm_p']
    n = len(rt)
    
    if strat == 'Long':
        pos = np.ones(n)
    elif strat == 'Sign(R)':
        pos = strategy_sign_r(rt, 252)
    else:
        pos = rd['macd_pos']
    
    Rt = np.zeros(n)
    for t in range(1, n):
        if sigma[t - 1] > 0 and (t < 2 or sigma[t - 2] > 0):
            a_prev = pos[t - 1]
            a_prev2 = pos[t - 2] if t >= 2 else 0.0
            sp = a_prev * sigma_tgt / sigma[t - 1]
            spp = a_prev2 * sigma_tgt / sigma[t - 2] if t >= 2 else 0.0
            Rt[t] = sp * rt[t] - BP * norm_p[t - 1] * abs(sp - spp)
    return Rt

def compute_portfolio_returns(raw_data, strat, sigma_tgt):
    series = []
    for rd in raw_data:
        Rt = compute_contract_returns(rd, strat, sigma_tgt)
        start, t1, dates = rd['start'], rd['t1'], rd['dates']
        slc = Rt[start:t1]
        series.append(pd.Series(slc[:len(dates)], index=dates[:len(slc)]))
    return pd.DataFrame(series).T.dropna().mean(axis=1).values

# Run for all asset classes and strategies
results = []
asset_classes = ['Commodity', 'Equity Index', 'Fixed Income', 'Forex']
strategies = ['Long', 'Sign(R)', 'MACD']

print("="*140)
print("FULL BASELINE REPLICATION — Table 3 (Per-Contract Vol Scaling)")
print("="*140)

for ac in asset_classes:
    raw = load_contracts(ac)
    N = len(raw)
    if N == 0:
        continue
    
    print(f"\n{'='*140}")
    print(f"  {ac} ({N} contracts)")
    print(f"{'='*140}")
    print(f"  Contracts: {', '.join([rd['tk'] for rd in raw])}")
    
    for strat in strategies:
        R = compute_portfolio_returns(raw, strat, SIGMA_TGT)
        m = compute_metrics(R, N)
        pv = PAPER_TABLE3[ac][strat]
        pv_vals = [pv[k] for k in METRIC_NAMES]
        
        errs = [abs((m[i] - pv_vals[i]) / abs(pv_vals[i])) * 100 if pv_vals[i] != 0 else 0 for i in range(9)]
        
        results.append({
            'Asset_Class': ac,
            'Strategy': strat,
            'N_Contracts': N,
            'E(R)': m[0], 'std(R)': m[1], 'DD': m[2], 'Sharpe': m[3],
            'Sortino': m[4], 'MDD': m[5], 'Calmar': m[6], '%_ve': m[7], 'Ave_P/L': m[8],
            'Paper_E(R)': pv['E(R)'], 'Paper_std': pv['std(R)'], 'Paper_Sharpe': pv['Sharpe'],
            'Err_E(R)': errs[0], 'Err_std': errs[1], 'Err_Sharpe': errs[3],
            'All_Errs': ','.join([f'{e:.1f}' for e in errs])
        })
        
        # Print detailed comparison
        print(f"\n  {strat:8s} (≤10%: {sum(1 for e in errs if e<10)}/9  ≤15%: {sum(1 for e in errs if e<15)}/9)")
        print(f"  Ours  : {'  '.join(f'{v:+7.3f}' for v in m)}")
        print(f"  Paper : {'  '.join(f'{v:+7.3f}' for v in [pv[k] for k in METRIC_NAMES])}")
        print(f"  %Err  : {'  '.join(f'{e:>6.1f}%' for e in errs)}")

# Save results
df = pd.DataFrame(results)
df.to_csv('tests/results/full_baseline_table3.csv', index=False)

# =============================================================================
# Additional: "All" portfolio (all 50 contracts combined)
# =============================================================================
print("\n\n" + "="*140)
print("  ALL CONTRACTS (50 contracts combined)")
print("="*140)

all_raw = load_contracts('All')
N_all = len(all_raw)
print(f"\n  Total contracts: {N_all}")
print(f"  Contracts: {', '.join([rd['tk'] for rd in all_raw])}")

# Portfolio-level vol scaling to match Table 2 methodology
sigma_tgt_portfolio = SIGMA_TGT / np.sqrt(N_all)

for strat in strategies:
    R_all = compute_portfolio_returns(all_raw, strat, SIGMA_TGT)  # Per-contract scaling
    m_all = compute_metrics(R_all, N_all)
    
    print(f"\n  {strat:8s}")
    print(f"  Ours  : {'  '.join(f'{v:+7.3f}' for v in m_all)}")
    
    results.append({
        'Asset_Class': 'All',
        'Strategy': strat,
        'N_Contracts': N_all,
        'E(R)': m_all[0], 'std(R)': m_all[1], 'DD': m_all[2], 'Sharpe': m_all[3],
        'Sortino': m_all[4], 'MDD': m_all[5], 'Calmar': m_all[6], '%_ve': m_all[7], 'Ave_P/L': m_all[8],
        'Paper_E(R)': None, 'Paper_std': None, 'Paper_Sharpe': None,
        'Err_E(R)': None, 'Err_std': None, 'Err_Sharpe': None,
        'All_Errs': 'N/A'
    })

# Generate summary table
print("\n\n" + "="*140)
print("SUMMARY TABLE — All 9 Metrics Comparison")
print("="*140)

# Create formatted summary
summary_lines = []
for ac in asset_classes:
    ac_results = df[df['Asset_Class'] == ac]
    for _, row in ac_results.iterrows():
        strat = row['Strategy']
        pv = PAPER_TABLE3[ac][strat]
        pv_vals = [pv[k] for k in METRIC_NAMES]
        errs = [row['Err_E(R)'], row['Err_std'], 
                abs((row['DD'] - pv_vals[2])/pv_vals[2])*100 if pv_vals[2]!=0 else 0,
                row['Err_Sharpe'],
                abs((row['Sortino'] - pv_vals[4])/pv_vals[4])*100 if pv_vals[4]!=0 else 0,
                abs((row['MDD'] - pv_vals[5])/pv_vals[5])*100 if pv_vals[5]!=0 else 0,
                abs((row['Calmar'] - pv_vals[6])/pv_vals[6])*100 if pv_vals[6]!=0 else 0,
                abs((row['%_ve'] - pv_vals[7])/pv_vals[7])*100 if pv_vals[7]!=0 else 0,
                abs((row['Ave_P/L'] - pv_vals[8])/pv_vals[8])*100 if pv_vals[8]!=0 else 0]
        
        summary_lines.append({
            'Asset': ac, 'Strategy': strat, 'N': row['N_Contracts'],
            'E(R)': f"{row['E(R)']:+.3f}", 'std': f"{row['std(R)']:.3f}",
            'DD': f"{row['DD']:.3f}", 'Sharpe': f"{row['Sharpe']:+.3f}",
            'Sortino': f"{row['Sortino']:+.3f}", 'MDD': f"{row['MDD']:.3f}",
            'Calmar': f"{row['Calmar']:+.3f}", '%+ve': f"{row['%_ve']:.3f}", 'AveP/L': f"{row['Ave_P/L']:.3f}",
            'Err_avg': np.mean(errs),
            'Err<10': sum(1 for e in errs if e < 10),
            'Err<15': sum(1 for e in errs if e < 15),
        })

summary_df = pd.DataFrame(summary_lines)
print("\n" + summary_df.to_string(index=False))

# Save formatted summary
summary_df.to_csv('tests/results/baseline_alignment_summary.csv', index=False)

# Grand total
total_under10 = summary_df['Err<10'].sum()
total_under15 = summary_df['Err<15'].sum()
print(f"\n{'='*140}")
print(f"GRAND TOTAL: ≤10%: {total_under10}/108 | ≤15%: {total_under15}/108")
print(f"{'='*140}")

print(f"\nOutput files:")
print(f"  - tests/results/full_baseline_table3.csv")
print(f"  - tests/results/baseline_alignment_summary.csv")
