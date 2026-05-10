#!/usr/bin/env python3
"""
DQN Walk-Forward Full Backtest (2011-2019) - CUDA Accelerated
- Round 1 models for 2011-2014
- Round 2 models for 2015-2019
- Proper date alignment for portfolio
- GPU-accelerated inference
"""
import sys, numpy as np, pandas as pd, torch, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ASSET_CLASSES, PAPER_TABLE3, METRIC_NAMES, EWMA_SPAN, BP
from baseline_run import compute_metrics
from data_loader import load_clc_full
from strategy_dqn import DQNAgent, build_all_features, get_feature_window, WARMUP

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'dqn_walkforward')

PERIODS = {
    'R1': {'start': '2011-01-01', 'end': '2014-12-31', 'round': 1},
    'R2': {'start': '2015-01-01', 'end': '2019-12-31', 'round': 2},
}

def load_model_to_device(ticker, round_num, device):
    path = os.path.join(MODEL_DIR, f"{ticker}_r{round_num}.pt")
    if not os.path.exists(path):
        return None
    agent = DQNAgent()
    agent.q_net.to(device)
    ckpt = torch.load(path, map_location=device)
    agent.q_net.load_state_dict(ckpt['q'])
    agent.target.load_state_dict(ckpt['t'])
    agent.q_net.eval()
    return agent

def get_scaled_returns_for_period(ticker, round_num, sigma_tgt, period_start, period_end):
    """Get volatility-scaled returns for one contract in one period using GPU."""
    df = load_clc_full(ticker, source='RAD', start_date='2005-01-01', anchor_date='2010-01-01', data_dir='IEOR4733_Project/data/CLC')
    if df is None:
        return None, None
    
    prices = df['Close'].to_numpy()
    returns = np.zeros(len(prices)); returns[1:] = prices[1:] - prices[:-1]
    sigma = pd.Series(returns).ewm(span=EWMA_SPAN, adjust=False).std().values
    features = build_all_features(prices, returns, sigma)
    dates = pd.to_datetime(df['Date'])
    
    agent = load_model_to_device(ticker, round_num, DEVICE)
    if agent is None:
        return None, None
    
    scaled_returns = []
    valid_dates = []
    last_action = 0
    
    period_start_ts = pd.Timestamp(period_start)
    period_end_ts = pd.Timestamp(period_end)
    
    for idx in range(WARMUP, len(prices)):
        date = dates.iloc[idx]
        if date < period_start_ts or date > period_end_ts:
            # Still need to track last_action for continuity
            state = get_feature_window(features, idx)
            if state is not None:
                state_t = torch.from_numpy(state).unsqueeze(0).float().to(DEVICE)
                with torch.no_grad():
                    action = agent.q_net(state_t).argmax().item()
                last_action = int(action)
            continue
        
        state = get_feature_window(features, idx)
        if state is None:
            continue
        
        # GPU inference
        state_t = torch.from_numpy(state).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            action = agent.q_net(state_t).argmax().item()
        
        if sigma[idx-1] > 0:
            vol_scale = sigma_tgt / sigma[idx-1]
            sp = action * vol_scale
            spp = last_action * (sigma_tgt / sigma[idx-2]) if idx >= 2 else 0
            gross = sp * returns[idx]
            tc = BP * prices[idx-1] * abs(sp - spp)
            scaled_returns.append(gross - tc)
            valid_dates.append(date)
        
        last_action = action
    
    return np.array(scaled_returns), valid_dates

def backtest_full_aligned(asset='Forex', sigma_tgt=0.058):
    print(f"\n{'='*70}")
    print(f"DQN Walk-Forward Full Backtest (2011-2019) - CUDA Accelerated")
    print(f"Device: {DEVICE}")
    print(f"Asset: {asset}, Sigma_tgt: {sigma_tgt}")
    print(f"{'='*70}\n")
    
    tickers = ASSET_CLASSES.get(asset, [])
    all_returns = {}
    
    t0 = time.time()
    for tk in tickers:
        # Round 1: 2011-2014
        R1, d1 = get_scaled_returns_for_period(tk, 1, sigma_tgt, '2011-01-01', '2014-12-31')
        # Round 2: 2015-2019
        R2, d2 = get_scaled_returns_for_period(tk, 2, sigma_tgt, '2015-01-01', '2019-12-31')
        
        if R1 is None or R2 is None:
            print(f"  ❌ {tk}: failed")
            continue
        
        # Concatenate R1 + R2 with dates
        all_d = list(d1) + list(d2)
        all_R = np.concatenate([R1, R2])
        all_returns[tk] = pd.Series(all_R, index=all_d)
        print(f"  ✅ {tk}: {len(all_R)} days, mean={all_R.mean():.6f}, std={all_R.std():.6f}")
    
    t1 = time.time()
    print(f"\nInference time: {t1-t0:.1f}s")
    
    if not all_returns:
        print("No contracts succeeded!")
        return
    
    # Align all contracts by date
    df = pd.DataFrame(all_returns)
    df = df.dropna()
    print(f"\nAligned trading days: {len(df)}")
    
    # Portfolio: equal-weight
    R_portfolio = df.mean(axis=1)
    
    m = compute_metrics(R_portfolio, n_contracts=len(all_returns))
    metrics_dqn = dict(zip(METRIC_NAMES, m))
    paper = PAPER_TABLE3[asset]['Long']
    
    print(f"\n{'='*70}")
    print(f"{asset} DQN Walk-Forward ({len(all_returns)} contracts)")
    print(f"{'='*70}")
    
    n_ok = 0
    for met in ['E(R)', 'std(R)', 'DD', 'Sharpe', 'Sortino', 'MDD', 'Calmar', '% +ve', 'Ave P/L']:
        ours, p = metrics_dqn[met], paper[met]
        err = abs((ours - p) / abs(p)) * 100 if p != 0 else 0
        ok = "✅" if err <= 15 else "❌"
        if err <= 15:
            n_ok += 1
        print(f"  {met:10}: {ours:+9.4f} vs {p:+9.4f}  err={err:6.1f}%  {ok}")
    
    print(f"\n{'='*70}")
    print(f"Aligned: {n_ok}/9 metrics within 15% error")
    print(f"{'='*70}\n")
    
    return metrics_dqn

if __name__ == '__main__':
    backtest_full_aligned('Forex', 0.058)
