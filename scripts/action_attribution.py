#!/usr/bin/env python3
import argparse, sys, numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

def main():
    p = argparse.ArgumentParser(description='Print DQN action distribution from audit.npz')
    p.add_argument('--audit', required=True, help='Path to audit.npz file')
    p.add_argument('--ticker', default=None, help='Filter to specific contract ticker')
    args = p.parse_args()
    
    data = np.load(args.audit, allow_pickle=True)
    positions = data['positions']
    tickers = [str(t) for t in data['tickers']]
    ticker_list = [args.ticker] if args.ticker else tickers
    
    print(f'{"Ticker":>6} {"Long%":>8} {"Short%":>9} {"Flat%":>8} {"Total":>8} {"L/S":>8}')
    print('-' * 50)
    for i, t in enumerate(tickers):
        if t not in ticker_list:
            continue
        pos = positions[:, i]
        total = len(pos)
        long_pct = (pos > 0.5).mean() * 100
        short_pct = (pos < -0.5).mean() * 100
        flat_pct = (abs(pos) < 0.5).mean() * 100
        ls_ratio = long_pct / short_pct if short_pct > 0 else float('inf')
        print(f'{t:>6} {long_pct:>7.1f}% {short_pct:>8.1f}% {flat_pct:>7.1f}% {total:>8} {ls_ratio:>7.2f}')

if __name__ == '__main__':
    main()
