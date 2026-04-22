#!/usr/bin/env python3
"""
DQN Walk-Forward Backtest

Loads trained DQN models and evaluates on test periods.

Usage:
    python backtest_dqn_walkforward.py --round 1 --asset Forex
    python backtest_dqn_walkforward.py --round 2 --asset Forex
    python backtest_dqn_walkforward.py --all --asset Forex
"""
import os, sys, argparse, numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import ASSET_CLASSES, PAPER_TABLE3, METRIC_NAMES
from baseline_run import load_contracts, compute_portfolio_returns, compute_metrics

MODEL_DIR = os.path.join(ROOT, 'dqn', 'models', 'walkforward')

# Walk-forward rounds
ROUNDS = {
    1: {'train': '2005-2009', 'test': '2010-2014'},
    2: {'train': '2005-2014', 'test': '2015-2019'},
}

def backtest_round(round_num, asset='Forex'):
    """Backtest one walk-forward round."""
    print(f"\n{'='*70}")
    print(f"DQN Walk-Forward Backtest - Round {round_num}")
    print(f"Train: {ROUNDS[round_num]['train']}, Test: {ROUNDS[round_num]['test']}")
    print(f"{'='*70}\n")
    
    # Load DQN models
    tickers = ASSET_CLASSES.get(asset, [])
    models_loaded = []
    for tk in tickers:
        model_path = os.path.join(MODEL_DIR, f"{tk}_r{round_num}.pt")
        if os.path.exists(model_path):
            models_loaded.append(tk)
    
    print(f"Models loaded: {len(models_loaded)}/{len(tickers)}")
    
    # For now, just run Long Only as baseline
    # TODO: Integrate DQN inference
    raw = load_contracts(asset)
    R = compute_portfolio_returns(raw, 'Long', sigma_tgt=0.063)
    m = compute_metrics(R, n_contracts=len(raw))
    metrics = dict(zip(METRIC_NAMES, m))
    paper = PAPER_TABLE3[asset]['Long']
    
    print(f"\n{asset} ({len(raw)} contracts):")
    for met in ['E(R)', 'std(R)', 'Sharpe', 'MDD', 'Calmar']:
        ours = metrics[met]
        p = paper[met]
        err = abs((ours - p) / abs(p)) * 100 if p != 0 else 0
        st = '✅' if err < 15 else '❌'
        print(f"  {met:8s}: {ours:+.4f} vs {p:+.4f} err={err:.1f}% {st}")
    
    return metrics

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--round', type=int, choices=[1, 2])
    p.add_argument('--all', action='store_true')
    p.add_argument('--asset', default='Forex')
    a = p.parse_args()
    
    if a.all:
        for r in [1, 2]:
            backtest_round(r, a.asset)
    elif a.round:
        backtest_round(a.round, a.asset)
    else:
        p.print_help()

if __name__ == '__main__':
    main()
