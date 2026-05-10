#!/usr/bin/env python3
"""Parallel runner: launch N seeds simultaneously using ProcessPoolExecutor, collect aggregate results."""
import argparse
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def run_seed(args_dict):
    """Worker function that runs one seed. Training output -> log file."""
    import sys, os
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    
    save_dir = _Path(args_dict.get('save_dir', '/tmp'))
    logs_dir = save_dir / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"seed_{args_dict['seed']}.log"
    
    real_stdout = sys.stdout
    with open(log_path, 'w', buffering=1) as log_f:
        log_f.write(f"SEED={args_dict['seed']} LOG={log_path}\n")
        log_f.write(f"HP: gamma={args_dict.get('gamma')} tau={args_dict.get('tau')} cycles={args_dict.get('cycles')}\n")
        log_f.write("=" * 60 + "\n")
        log_f.flush()
        sys.stdout = log_f
        try:
            from scripts.roll_train import run
            run(**args_dict)
        finally:
            sys.stdout = real_stdout
    
    print(f"  [done] seed={args_dict['seed']} log={log_path}", file=real_stdout, flush=True)
    return args_dict['seed']


def load_seed_results(save_dir: Path, seed: int) -> dict:
    """Load results from a seed's window directories."""
    seed_dir = save_dir / str(seed)
    if not seed_dir.exists():
        return None
    
    results = {
        'seed': seed,
        'windows': [],
        'dq_sharpes': [],
        'lo_sharpes': [],
    }
    
    # Find all window directories
    window_dirs = sorted(seed_dir.glob('window_*'))
    
    for window_dir in window_dirs:
        # Load diagnostics.json if it exists
        diag_path = window_dir / 'diagnostics.json'
        if diag_path.exists():
            try:
                with open(diag_path) as f:
                    diag = json.load(f)
                    window_data = {
                        'window': window_dir.name,
                        'diagnostics': diag,
                    }
                    # Extract Sharpe ratios if available
                    if isinstance(diag, dict):
                        if 'dq_sharpe' in diag:
                            window_data['dq_sharpe'] = diag['dq_sharpe']
                            results['dq_sharpes'].append(diag['dq_sharpe'])
                        if 'lo_sharpe' in diag:
                            window_data['lo_sharpe'] = diag['lo_sharpe']
                            results['lo_sharpes'].append(diag['lo_sharpe'])
                    results['windows'].append(window_data)
            except (json.JSONDecodeError, IOError):
                pass
    
    return results if results['windows'] else None


def main():
    p = argparse.ArgumentParser(description='Parallel DQN training across multiple seeds')
    
    # Existing arguments
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456],
                   help='List of seeds to run')
    p.add_argument('--gamma', type=float, default=0.5,
                   help='Gamma parameter')
    p.add_argument('--tau', type=int, default=500,
                   help='Tau parameter')
    p.add_argument('--sigma-tgt', type=float, default=0.10,
                   help='Target sigma')
    p.add_argument('--train-years', type=int, default=2,
                   help='Number of years for training')
    p.add_argument('--test-months', type=int, default=6,
                   help='Number of months for testing')
    p.add_argument('--device', default='cuda',
                   help='Device to use (cuda/cpu)')
    p.add_argument('--cycles', type=int, default=50,
                   help='Number of training cycles')
    p.add_argument('--eps-start', type=float, default=0.01,
                   help='Starting epsilon')
    p.add_argument('--eps-end', type=float, default=0.01,
                   help='Ending epsilon')
    
    # New arguments
    p.add_argument('--save-dir', default='results/v4',
                   help='Base directory for saving results')
    p.add_argument('--window-limit', type=int, default=None,
                   help='Limit number of windows to process (None = all)')
    
    args = p.parse_args()
    
    seeds = args.seeds
    print(f"Launching {len(seeds)} parallel seeds: {seeds}")
    print(f"HP: gamma={args.gamma} tau={args.tau} sigma_tgt={args.sigma_tgt} "
          f"train={args.train_years}yr test={args.test_months}mo cycles={args.cycles} "
          f"eps={args.eps_start}->{args.eps_end}")
    print(f"Save dir: {args.save_dir}")
    print("=" * 60, flush=True)
    
    # Determine max workers
    max_workers = min(len(seeds), 3)
    print(f"Using {max_workers} workers (CPU count: {multiprocessing.cpu_count()})")
    
    # Prepare save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Track start time
    t0 = time.time()
    
    ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = []
        for seed in seeds:
            hp = {
                'seed': seed,
                'gamma': args.gamma,
                'tau': args.tau,
                'sigma_tgt': args.sigma_tgt,
                'train_years': args.train_years,
                'test_months': args.test_months,
                'device': args.device,
                'cycles': args.cycles,
                'eps_start': args.eps_start,
                'eps_end': args.eps_end,
                'save_dir': str(save_dir / str(seed)),
                'window_limit': args.window_limit,
            }
            future = executor.submit(run_seed, hp)
            futures.append(future)
        
        # Wait for all to complete and collect results
        from tqdm import tqdm
        completed_seeds = []
        with tqdm(total=len(futures), desc="Training seeds", unit="seed") as pbar:
            for future in as_completed(futures):
                try:
                    seed = future.result()
                    completed_seeds.append(seed)
                    pbar.set_postfix({"seed": seed, "elapsed": f"{time.time()-t0:.0f}s", "log": f"seed_{seed}.log"})
                except Exception as e:
                    print(f"\n[seed unknown] Failed with error: {e}", flush=True)
                pbar.update(1)
    
    total_time = time.time() - t0
    print(f"\nAll {len(completed_seeds)} processes completed in {total_time:.1f}s")
    
    # Aggregate results from saved files
    print("\n" + "=" * 60)
    print("AGGREGATING RESULTS")
    print("=" * 60)
    
    summary = {
        'seeds': {},
        'aggregate': {},
        'config': {
            'gamma': args.gamma,
            'tau': args.tau,
            'sigma_tgt': args.sigma_tgt,
            'train_years': args.train_years,
            'test_months': args.test_months,
            'cycles': args.cycles,
            'eps_start': args.eps_start,
            'eps_end': args.eps_end,
            'device': args.device,
            'window_limit': args.window_limit,
        },
        'runtime_seconds': total_time,
    }
    
    all_dq_sharpes = []
    all_lo_sharpes = []
    
    for seed in completed_seeds:
        seed_results = load_seed_results(save_dir, seed)
        if seed_results:
            summary['seeds'][str(seed)] = seed_results
            
            # Collect Sharpe ratios for aggregation
            if seed_results['dq_sharpes']:
                seed_mean_dq = np.mean(seed_results['dq_sharpes'])
                print(f"  seed={seed}: DQN Sharpe = {seed_mean_dq:+.3f} "
                      f"(across {len(seed_results['dq_sharpes'])} windows)")
                all_dq_sharpes.extend(seed_results['dq_sharpes'])
            
            if seed_results['lo_sharpes']:
                seed_mean_lo = np.mean(seed_results['lo_sharpes'])
                print(f"  seed={seed}: Long Sharpe = {seed_mean_lo:+.3f} "
                      f"(across {len(seed_results['lo_sharpes'])} windows)")
                all_lo_sharpes.extend(seed_results['lo_sharpes'])
    
    # Compute aggregate statistics
    if all_dq_sharpes:
        dq_mean = np.mean(all_dq_sharpes)
        dq_std = np.std(all_dq_sharpes)
        summary['aggregate']['dq_sharpe_mean'] = float(dq_mean)
        summary['aggregate']['dq_sharpe_std'] = float(dq_std)
        summary['aggregate']['dq_sharpe_n_windows'] = len(all_dq_sharpes)
    
    if all_lo_sharpes:
        lo_mean = np.mean(all_lo_sharpes)
        lo_std = np.std(all_lo_sharpes)
        summary['aggregate']['lo_sharpe_mean'] = float(lo_mean)
        summary['aggregate']['lo_sharpe_std'] = float(lo_std)
        summary['aggregate']['lo_sharpe_n_windows'] = len(all_lo_sharpes)
    
    # Save summary.json
    summary_path = save_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    
    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    if all_dq_sharpes:
        print(f"DQN  Sharpe: {np.mean(all_dq_sharpes):+.3f} ± {np.std(all_dq_sharpes):.3f} "
              f"({len(all_dq_sharpes)} windows across {len(completed_seeds)} seeds)")
    else:
        print("DQN  Sharpe: No data")
    
    if all_lo_sharpes:
        print(f"Long Sharpe: {np.mean(all_lo_sharpes):+.3f} ± {np.std(all_lo_sharpes):.3f} "
              f"({len(all_lo_sharpes)} windows across {len(completed_seeds)} seeds)")
    else:
        print("Long Sharpe: No data")
    
    print(f"\nTotal runtime: {total_time:.1f}s")


if __name__ == "__main__":
    main()
