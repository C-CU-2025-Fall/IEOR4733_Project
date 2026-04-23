#!/usr/bin/env python3
"""
DQN Walk-Forward Full Backtest (2010-2019)
- Round 1 models for 2010-2014
- Round 2 models for 2015-2019
"""
import sys, numpy as np, pandas as pd, torch, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ASSET_CLASSES, PAPER_TABLE3, METRIC_NAMES, EWMA_SPAN, BP
from baseline_run import compute_metrics
from data_loader import load_clc_full
from strategy_dqn import DQNAgent, build_all_features, get_feature_window, WARMUP

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'dqn_walkforward')

PERIODS = {
    'R1': {'start': '2010-01-01', 'end': '2014-12-31', 'round': 1},
    'R2': {'start': '2015-01-01', 'end': '2019-12-31', 'round': 2},
}

def load_model(ticker, round_num):
    path = os.path.join(MODEL_DIR, f"{ticker}_r{round_num}.pt")
    if not os.path.exists(path):
        return None
    agent = DQNAgent()
    agent.q_net.to('cpu')
    ckpt = torch.load(path, map_location='cpu')
    agent.q_net.load_state_dict(ckpt['q'])
    agent.target.load_state_dict(ckpt['t'])
    agent.q_net.eval()
    return agent

def get_contract_returns(ticker, sigma_tgt):
    """Get full 2010-2019 returns for one contract using walk-forward models."""
    df = load_clc_full(ticker, source='RAD', start_date='2005-01-01', anchor_date='2010-01-01', data_dir='IEOR4733_Project/data/CLC')
    if df is None:
        return None
    
    prices_all = df['Close'].to_numpy()
    returns_all = np.zeros(len(prices_all)); returns_all[1:] = prices_all[1:] - prices_all[:-1]
    sigma_all = pd.Series(returns_all).ewm(span=EWMA_SPAN, adjust=False).std().values
    dates = pd.to_datetime(df['Date'])
    
    positions = np.zeros(len(prices_all))
    
    for period_name, period in PERIODS.items():
        round_num = period['round']
        agent = load_model(ticker, round_num)
        if agent is None:
            continue
        
        # Generate positions for this period
        for idx in range(WARMUP, len(prices_all)):
            date = dates.iloc[idx]
            if date < pd.Timestamp(period['start']) or date > pd.Timestamp(period['end']):
                continue
            
            features = build_all_features(prices_all, returns_all, sigma_all)
            state = get_feature_window(features, idx)
            if state is None:
                continue
            state_t = torch.from_numpy(state).unsqueeze(0).float()
            with torch.no_grad():
                action = agent.q_net(state_t).argmax().item()
            positions[idx] = float(action - 1)
    
    # Compute returns with vol scaling
    n = len(prices_all)
    Rt = np.zeros(n)
    for t in range(1, n):
        if sigma_all[t-1] > 0:
            a_prev = positions[t-1]
            a_prev2 = positions[t-2] if t >= 2 else 0.0
            sp = a_prev * sigma_tgt / sigma_all[t-1]
            spp = a_prev2 * sigma_tgt / sigma_all[t-2] if t >= 2 else 0.0
            gross = sp * returns_all[t]
            tc = BP * prices_all[t-1] * abs(sp - spp)
            Rt[t] = gross - tc
    
    # Filter to 2010-2019
    mask = (dates >= '2010-01-01') & (dates <= '2019-12-31')
    return Rt[mask]

def backtest_full(asset='Forex', sigma_tgt=0.058):
    print(f"\n{'='*70}")
    print(f"DQN Walk-Forward Full Backtest (2010-2019)")
    print(f"Asset: {asset}, Sigma_tgt: {sigma_tgt}")
    print(f"{'='*70}\n")
    
    tickers = ASSET_CLASSES.get(asset, [])
    contract_returns = []
    
    for tk in tickers:
        Rt = get_contract_returns(tk, sigma_tgt)
        if Rt is None or len(Rt) < 100:
            print(f"  ❌ {tk}: failed")
            continue
        contract_returns.append(Rt)
        print(f"  ✅ {tk}: len={len(Rt)}, mean={Rt.mean():.4f}, std={Rt.std():.4f}")
    
    if not contract_returns:
        print("No contracts succeeded!")
        return
    
    # Portfolio - simple mean (note: this doesn't align dates properly)
    # For proper alignment, use the correlation-aware version
    max_len = min(len(R) for R in contract_returns)
    R_matrix = np.column_stack([R[:max_len] for R in contract_returns])
    R_portfolio = R_matrix.mean(axis=1)
    
    m = compute_metrics(R_portfolio, n_contracts=len(contract_returns))
    metrics_dqn = dict(zip(METRIC_NAMES, m))
    paper = PAPER_TABLE3[asset]['Long']
    
    print(f"\n{'='*70}")
    print(f"{asset} DQN Walk-Forward ({len(contract_returns)} contracts, {max_len} days)")
    print(f"{'='*70}")
    for met in ['E(R)', 'std(R)', 'DD', 'Sharpe', 'Sortino', 'MDD', 'Calmar', '% +ve', 'Ave P/L']:
        ours, p = metrics_dqn[met], paper[met]
        err = abs((ours - p) / abs(p)) * 100 if p != 0 else 0
        ok = "✅" if err <= 15 else "❌"
        print(f"  {met:10}: {ours:+8.4f} vs {p:+8.4f}  err={err:6.1f}%  {ok}")
    
    return metrics_dqn

if __name__ == '__main__':
    backtest_full('Forex', 0.058)
