#!/usr/bin/env python3
"""
Walk-Forward DQN Training for Forex.

Usage:
    python train_dqn_walkforward.py --round 1 --asset Forex --episodes 100
    python train_dqn_walkforward.py --round 2 --asset Forex --episodes 100
"""
import os, sys, time, argparse, gc
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from strategy_dqn import DQNAgent, ContractEnv, build_all_features
from config import ASSET_CLASSES, EWMA_SPAN
from data_loader import load_clc_full

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = os.path.join(ROOT, 'models', 'dqn_walkforward')
os.makedirs(MODEL_DIR, exist_ok=True)

# Walk-forward rounds
ROUNDS = {
    1: {'train_start': '2005-01-01', 'train_end': '2009-12-31', 'test_start': '2010-01-01', 'test_end': '2014-12-31'},
    2: {'train_start': '2005-01-01', 'train_end': '2014-12-31', 'test_start': '2015-01-01', 'test_end': '2019-12-31'},
}

def load_round_data(ticker, round_num):
    """Load preprocessed walk-forward data."""
    round_info = ROUNDS[round_num]
    path = os.path.join(ROOT, 'data', 'dqn_walkforward', f"{ticker}_r{round_num}_train.npz")
    
    if not os.path.exists(path):
        print(f"  {ticker}: ❌ data not found")
        return None
    
    data = np.load(path, allow_pickle=True)
    return {
        'prices': data['prices'],
        'returns': data['returns'],
        'sigma': data['sigma'],
        'features': data['features'],
        'dates': data['dates'],
        'source': str(data['source']),
    }

def train_contract(ticker, round_num, episodes=50):
    """Train DQN for one contract (CUDA accelerated)."""
    round_info = ROUNDS[round_num]
    data = load_round_data(ticker, round_num)
    
    if data is None:
        return False
    
    env = ContractEnv(data['prices'], data['returns'], data['sigma'], data['features'])
    agent = DQNAgent()
    
    report_interval = max(1, episodes // 5)
    t0 = time.time()
    
    for ep in range(episodes):
        state = env.reset()
        total_r = 0
        for _ in range(500):
            a = agent.act(state, eps=0.3)
            ns, r, done = env.step(a)
            agent.push(state, a, r, ns, float(done))
            agent.learn()
            state = ns
            total_r += r
            if done:
                break
        
        if (ep + 1) % report_interval == 0:
            elapsed = time.time() - t0
            print(f"  {ticker} ep {ep+1}/{episodes} r={total_r:+.2f} ({elapsed:.0f}s)")
    
    # Save model
    path = os.path.join(MODEL_DIR, f"{ticker}_r{round_num}.pt")
    agent.save(path)
    
    elapsed = time.time() - t0
    print(f"  {ticker}: ✅ saved ({elapsed:.1f}s)")
    
    del agent, env, data
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    return True

def train_asset_class(asset_name, round_num, episodes=100):
    """Train all contracts in one asset class."""
    tickers = ASSET_CLASSES.get(asset_name, [])
    
    print(f"\n{'='*60}")
    print(f"Walk-Forward Round {round_num}: {asset_name}")
    print(f"Train: {ROUNDS[round_num]['train_start']} ~ {ROUNDS[round_num]['train_end']}")
    print(f"Test: {ROUNDS[round_num]['test_start']} ~ {ROUNDS[round_num]['test_end']}")
    print(f"{'='*60}\n")
    
    t0 = time.time()
    ok = 0
    
    for i, tk in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] {tk}:")
        if train_contract(tk, round_num, episodes):
            ok += 1
    
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"{asset_name} Round {round_num}: {ok}/{len(tickers)} trained")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Models: {MODEL_DIR}/")
    print(f"{'='*60}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--round', type=int, required=True, choices=[1, 2])
    p.add_argument('--asset', default='Forex')
    p.add_argument('--episodes', type=int, default=100)
    a = p.parse_args()
    
    train_asset_class(a.asset, a.round, a.episodes)
