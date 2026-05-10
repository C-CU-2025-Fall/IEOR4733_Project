#!/usr/bin/env python3
import argparse, json, sys, numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from drl.dqn.backtest.engine import portfolio_metrics

BUNDLES_R1 = {
    42: '20260505T102356_s42', 43: '20260505T102358_s43', 44: '20260505T102402_s44',
    45: '20260505T105344_s45', 46: '20260505T105347_s46',
    47: None, 48: None, 49: None, 50: None, 51: None,
}

def find_bundle(seed, round_num):
    model_root = REPO / 'drl' / 'dqn' / 'models' / 'Forex' / f'r{round_num}'
    for d in sorted(model_root.glob(f'*_s{seed}'), reverse=True):
        if (d / 'checkpoint.pt').exists():
            return str(d)
    return None

def main():
    p = argparse.ArgumentParser(description='Full backtest for gamma=0.6 with audit trail')
    p.add_argument('--asset', default='Forex')
    p.add_argument('--seeds', nargs='+', type=int, default=list(range(42, 52)))
    p.add_argument('--rounds', nargs='+', type=int, default=[1, 2])
    p.add_argument('--train-round', type=int, default=1, help='Which training round checkpoint to use')
    p.add_argument('--out-dir', default=None)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    
    out_dir = Path(args.out_dir) if args.out_dir else REPO / 'results' / 'gamma06_audit' / args.asset
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for seed in args.seeds:
        bundle = find_bundle(seed, args.train_round)
        if bundle is None:
            print(f'SKIP seed={seed}: no bundle found for r{args.train_round}')
            continue
        
        for rn in args.rounds:
            audit_path = out_dir / f'r{rn}_s{seed}_audit.npz'
            metrics_path = out_dir / f'r{rn}_s{seed}_metrics.json'
            
            if audit_path.exists() and metrics_path.exists():
                print(f'SKIP r{rn}_s{seed}: already exists')
                continue
            
            print(f'Backtest r{rn} s{seed}...', end=' ', flush=True)
            try:
                metrics = portfolio_metrics(
                    args.asset, 'DQN',
                    round_num=rn,
                    checkpoint_bundle=bundle,
                    device=args.device,
                    progress=False,
                    save_audit_to=str(audit_path),
                )
                with open(metrics_path, 'w') as f:
                    json.dump({'gamma': 0.6, 'seed': seed, 'round': rn, 'train_round': args.train_round, **metrics}, f)
                print(f'OK ({len(metrics)} metrics, audit saved)')
            except Exception as e:
                print(f'FAIL: {e}')
    
    print(f'Done. Results in {out_dir}/')

if __name__ == '__main__':
    main()
