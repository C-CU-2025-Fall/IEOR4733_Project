"""
simulation_recorder.py — Record per-contract daily simulation data

For each (asset_class, strategy, contract), saves a CSV with:
  - date, close, norm_p
  - position (A), sigma, scaled_pos
  - r_t (raw additive return)
  - R_gross (position × r_t), cost, R_net (R_gross - cost)
  - cumret_gross, cumret_net

This allows:
  1. Reverse-engineering paper's MDD/Calmar from actual wealth paths
  2. Decomposing cost per contract per flip
  3. Validating signal alignment contract by contract
"""
import numpy as np
import pandas as pd
from data_loader import load_clc_full
from strategies import strategy_sign_r, strategy_macd
from config import ASSET_CLASSES, BP, SIGN_LOOKBACK

SIGMA_TGT = 0.064   # σ_tgt per contract [Paper Eq 4]

T = 252


def record_simulation(asset_class, strategy, output_dir="simulation_data"):
    """Run simulation for one (asset_class, strategy) and save per-contract CSVs + portfolio CSV."""
    contracts = ASSET_CLASSES.get(asset_class, [])
    all_dates = None
    portfolio_rows = []
    
    for tk in contracts:
        df = load_clc_full(tk)
        if df is None:
            continue
        prices = df['Close'].values.astype(float)
        dates = df['Date'].values
        n = len(prices)
        if n < 500:
            continue
        
        # Normalize
        p0 = prices[0]
        norm_p = prices / p0
        
        # Additive returns
        rt = np.zeros(n)
        rt[1:] = norm_p[1:] - norm_p[:-1]
        
        # EWMA sigma
        sigma = pd.Series(rt).ewm(span=60, adjust=False).std().values
        
        # Position signal
        if strategy == 'Long':
            pos = np.ones(n)
        elif strategy == 'Sign(R)':
            pos = strategy_sign_r(rt, SIGN_LOOKBACK)
        elif strategy == 'MACD':
            pos = strategy_macd(prices)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Compute daily R_t
        R_gross = np.zeros(n)
        cost_arr = np.zeros(n)
        scaled_pos = np.zeros(n)
        
        for t in range(1, n):
            if sigma[t-1] > 0 and (t < 2 or sigma[t-2] > 0):
                a_prev = pos[t-1]
                a_prev2 = pos[t-2] if t >= 2 else 0
                sp = a_prev * SIGMA_TGT / sigma[t-1]
                spp = a_prev2 * SIGMA_TGT / sigma[t-2] if t >= 2 else 0
                scaled_pos[t] = sp
                R_gross[t] = sp * rt[t]
                cost_arr[t] = BP * norm_p[t-1] * abs(sp - spp)
        
        R_net = R_gross - cost_arr
        cumret_gross = np.cumsum(R_gross)
        cumret_net = np.cumsum(R_net)
        
        # Save per-contract CSV
        contract_df = pd.DataFrame({
            'date': dates,
            'close': prices,
            'norm_p': norm_p,
            'position': pos,
            'sigma': sigma,
            'scaled_pos': scaled_pos,
            'r_t': rt,
            'R_gross': R_gross,
            'cost': cost_arr,
            'R_net': R_net,
            'cumret_gross': cumret_gross,
            'cumret_net': cumret_net,
        })
        
        fname = f"{output_dir}/{asset_class.replace(' ', '_')}_{strategy}_{tk}.csv"
        contract_df.to_csv(fname, index=False)
        print(f"  Saved {fname} ({len(contract_df)} rows)")
        
        # Collect for portfolio
        if all_dates is None:
            all_dates = dates
        portfolio_rows.append(R_net)
    
    if not portfolio_rows:
        return None
    
    # Portfolio: equal-weight average
    min_len = min(len(r) for r in portfolio_rows)
    R_port_net = np.mean([r[:min_len] for r in portfolio_rows], axis=0)
    R_port_gross_list = []
    for tk2 in contracts:
        fname = f"{output_dir}/{asset_class.replace(' ', '_')}_{strategy}_{tk2}.csv"
        try:
            cdf = pd.read_csv(fname)
            R_port_gross_list.append(cdf['R_gross'].values[:min_len])
        except:
            pass
    R_port_gross = np.mean(R_port_gross_list, axis=0) if R_port_gross_list else R_port_net
    
    port_df = pd.DataFrame({
        'date': all_dates[:min_len],
        'R_net': R_port_net,
        'R_gross': R_port_gross,
        'cumret_net': np.cumsum(R_port_net),
        'cumret_gross': np.cumsum(R_port_gross),
    })
    
    port_fname = f"{output_dir}/{asset_class.replace(' ', '_')}_{strategy}_portfolio.csv"
    port_df.to_csv(port_fname, index=False)
    print(f"  Saved {port_fname} ({len(port_df)} rows, {len(portfolio_rows)} contracts)")
    
    return port_df


if __name__ == "__main__":
    import sys
    asset = sys.argv[1] if len(sys.argv) > 1 else "Equity Index"
    strat = sys.argv[2] if len(sys.argv) > 2 else "Long"
    record_simulation(asset, strat)
    print("Done.")
