#!/usr/bin/env python3
"""
Per-contract DQN Walk-Forward Backtest

Usage:
    python run_dqn_backtest.py --round 1 --asset Forex --sigma 0.058
    python run_dqn_backtest.py --all --asset Forex
"""
import os, sys, argparse, numpy as np, pandas as pd, torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import ASSET_CLASSES, PAPER_TABLE3, METRIC_NAMES, EWMA_SPAN, SOURCE_OVERRIDES, BP
from baseline_run import load_contracts, compute_portfolio_returns, compute_metrics
from data_loader import load_clc_full
from strategy_dqn import DQNAgent, build_all_features, get_feature_window, WARMUP, DEVICE

MODEL_DIR = os.path.join(ROOT, 'models', 'dqn_walkforward')

ROUNDS = {
    1: {'train': '2005-2009', 'test': '2010-2014', 'test_start': '2010-01-01', 'test_end': '2014-12-31'},
    2: {'train': '2005-2014', 'test': '2015-2019', 'test_start': '2015-01-01', 'test_end': '2019-12-31'},
}

def load_model(ticker, round_num):
    model_path = os.path.join(MODEL_DIR, f"{ticker}_r{round_num}.pt")
    if not os.path.exists(model_path):
        return None
    agent = DQNAgent()
    agent.q_net.to('cpu')
    ckpt = torch.load(model_path, map_location='cpu')
    agent.q_net.load_state_dict(ckpt['q'])
    agent.target.load_state_dict(ckpt['t'])
    agent.q_net.eval()
    return agent

def infer_positions(ticker, round_num, sigma_tgt):
    source = SOURCE_OVERRIDES.get(ticker, 'RAD')
    round_info = ROUNDS[round_num]
    df = load_clc_full(ticker, source=source, start_date='2005-01-01', anchor_date=round_info['test_start'])
    if df is None:
        return None, None
    
    prices = df['Close'].to_numpy(dtype=float)
    returns = np.zeros(len(prices)); returns[1:] = prices[1:] - prices[:-1]
    sigma = pd.Series(returns).ewm(span=EWMA_SPAN, adjust=False).std().values
    features = build_all_features(prices, returns, sigma)
    
    agent = load_model(ticker, round_num)
    if agent is None:
        return None, None
    
    positions = np.zeros(len(prices))
    for idx in range(WARMUP, len(prices)):
        state = get_feature_window(features, idx)
        state_t = torch.from_numpy(state).unsqueeze(0).float()
        with torch.no_grad():
            action = agent.q_net(state_t).argmax().item()
        positions[idx] = float(action - 1)
    
    # Filter to test period
    test_mask = (df['Date'] >= round_info['test_start']) & (df['Date'] <= round_info['test_end'])
    return positions[test_mask.values], df[test_mask].reset_index(drop=True)

def compute_returns(prices, returns, sigma, positions, sigma_tgt):
    n = len(prices)
    Rt = np.zeros(n)
    for t in range(1, n):
        if sigma[t-1] > 0:
            a_prev = positions[t-1]
            a_prev2 = positions[t-2] if t >= 2 else 0.0
            sp = a_prev * sigma_tgt / sigma[t-1]
            spp = a_prev2 * sigma_tgt / sigma[t-2] if t >= 2 else 0.0
            gross = sp * returns[t]
            tc = BP * prices[t-1] * abs(sp - spp)
            Rt[t] = gross - tc
    return Rt

def backtest(round_num, asset='Forex', sigma_tgt=0.058):
    print(f"\n{'='*70}")
    print(f"DQN Per-Contract Backtest - Round {round_num}")
    print(f"Sigma_tgt: {sigma_tgt}")
    print(f"{'='*70}\n")
    
    tickers = ASSET_CLASSES.get(asset, [])
    models = [tk for tk in tickers if os.path.exists(os.path.join(MODEL_DIR, f"{tk}_r{round_num}.pt"))]
    print(f"Models: {len(models)}/{len(tickers)}\n")
    
    if not models:
        return None
    
    # DQN backtest
    contract_returns = []
    for tk in models:
        positions, df = infer_positions(tk, round_num, sigma_tgt)
        if positions is None or len(positions) < 100:
            continue
        
        prices = df['Close'].to_numpy()
        returns = np.zeros(len(prices)); returns[1:] = prices[1:] - prices[:-1]
        sigma = pd.Series(returns).ewm(span=EWMA_SPAN, adjust=False).std().values
        
        Rt = compute_returns(prices, returns, sigma, positions, sigma_tgt)
        contract_returns.append(Rt)
        print(f"  ✅ {tk}: pos={positions.mean():+.2f}±{positions.std():.2f}, len={len(Rt)}")
    
    # Portfolio
    max_len = min(len(R) for R in contract_returns)
    R_matrix = np.column_stack([R[:max_len] for R in contract_returns])
    R_portfolio = R_matrix.mean(axis=1)
    
    m = compute_metrics(R_portfolio, n_contracts=len(contract_returns))
    metrics_dqn = dict(zip(METRIC_NAMES, m))
    paper = PAPER_TABLE3[asset]['Long']
    
    print(f"\n{asset} DQN ({len(contract_returns)} contracts):")
    for met in ['E(R)', 'std(R)', 'DD', 'Sharpe', 'Sortino', 'MDD', 'Calmar', '% +ve', 'Ave P/L']:
        ours, p = metrics_dqn[met], paper[met]
        err = abs((ours - p) / abs(p)) * 100 if p != 0 else 0
        st = '✅' if err < 15 else '❌'
        print(f"  {met:8s}: {ours:+.4f} vs {p:+.4f} err={err:.1f}% {st}")
    
    # Long Only baseline
    round_info = ROUNDS[round_num]
    raw = load_contracts(asset, test_start=round_info['test_start'], test_end=round_info['test_end'])
    R_long = compute_portfolio_returns(raw, 'Long', sigma_tgt=sigma_tgt)
    m_long = compute_metrics(R_long, n_contracts=len(raw))
    metrics_long = dict(zip(METRIC_NAMES, m_long))
    
    print(f"\n{asset} Long Only ({len(raw)} contracts):")
    for met in ['E(R)', 'std(R)', 'Sharpe', 'MDD']:
        ours, p = metrics_long[met], paper[met]
        err = abs((ours - p) / abs(p)) * 100 if p != 0 else 0
        st = '✅' if err < 15 else '❌'
        print(f"  {met:8s}: {ours:+.4f} vs {p:+.4f} err={err:.1f}% {st}")
    
    return {'dqn': metrics_dqn, 'long': metrics_long}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--round', type=int, choices=[1, 2])
    p.add_argument('--all', action='store_true')
    p.add_argument('--asset', default='Forex')
    p.add_argument('--sigma', type=float, default=0.058)
    a = p.parse_args()
    
    if a.all:
        for r in [1, 2]:
            backtest(r, a.asset, a.sigma)
    elif a.round:
        backtest(a.round, a.asset, a.sigma)
    else:
        p.print_help()
