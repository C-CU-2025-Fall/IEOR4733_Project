#!/usr/bin/env python3
import argparse, sys, numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from config import TRADING_DAYS

def main():
    p = argparse.ArgumentParser(description='Compute rolling Sharpe from audit.npz')
    p.add_argument('--audit', required=True, help='Path to audit.npz file')
    p.add_argument('--window', type=int, default=TRADING_DAYS, help=f'Rolling window (default: {TRADING_DAYS})')
    p.add_argument('--step', type=int, default=21, help='Step size in days (default: 21 = monthly)')
    p.add_argument('--output', default=None, help='Save rolling Sharpe to .npy file')
    args = p.parse_args()
    
    data = np.load(args.audit, allow_pickle=True)
    R = data['portfolio_returns']
    dates = data['dates']
    
    w = args.window
    step = args.step
    rs = []
    rs_dates = []
    for i in range(0, len(R) - w, step):
        r_slice = R[i:i + w]
        s = r_slice.mean() / (r_slice.std() + 1e-10) * np.sqrt(TRADING_DAYS)
        rs.append(s)
        rs_dates.append(str(dates[i + w - 1]))
    
    rs_arr = np.array(rs)
    print(f'Window: {w} days | Step: {step} days | Points: {len(rs)}')
    print(f'Mean: {rs_arr.mean():+.4f} | Std: {rs_arr.std():.4f} | Min: {rs_arr.min():+.4f} | Max: {rs_arr.max():+.4f}')
    print(f'Positive: {(rs_arr > 0).mean() * 100:.1f}%')
    
    if args.output:
        np.save(args.output, rs_arr)
        print(f'Saved to {args.output}')

if __name__ == '__main__':
    main()
