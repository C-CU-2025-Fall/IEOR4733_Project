#!/usr/bin/env python3
"""Batch train DQN models in parallel with burn-in features."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drl_shared.spec import universe_tickers

# All 50 contracts
ALL_CONTRACTS = sorted(set(
    list(universe_tickers('Forex')) + 
    list(universe_tickers('Fixed Income')) + 
    list(universe_tickers('Equity Index')) + 
    list(universe_tickers('Commodity'))
))

def train_contract(ticker: str) -> dict:
    """Train a single contract and return results."""
    result = {
        'ticker': ticker,
        'status': 'running',
        'best_reward': None,
        'best_ep': None,
        'early_stop': False,
        'error': None,
    }
    
    try:
        cmd = [
            sys.executable,
            str(REPO_ROOT / 'drl/dqn/train/train_dqn_walkforward.py'),
            '--ticker', ticker,
            '--round', '1',
            '--episodes', '200',
            '--early-stop', '10',
        ]
        
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout per contract
            cwd=str(REPO_ROOT),
        )
        
        output = proc.stdout + proc.stderr
        
        # Parse results
        if 'Early stop' in output:
            result['early_stop'] = True
            for line in output.split('\n'):
                if 'Early stop' in line:
                    # Parse: "Early stop @ ep59 (best=-0.53 @ ep49)"
                    parts = line.split('best=')
                    if len(parts) > 1:
                        result['best_reward'] = float(parts[1].split('@')[0].strip())
                    parts2 = line.split('@ ep')
                    if len(parts2) > 1:
                        result['best_ep'] = int(parts2[1].strip().rstrip(')'))
        
        result['status'] = 'completed'
        result['output'] = output
        
    except subprocess.TimeoutExpired:
        result['status'] = 'timeout'
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
    
    return result


def main():
    max_workers = min(10, len(ALL_CONTRACTS))
    print(f"Training {len(ALL_CONTRACTS)} contracts with {max_workers} workers")
    print(f"Contracts: {ALL_CONTRACTS}")
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(train_contract, t): t for t in ALL_CONTRACTS}
        
        for i, future in enumerate(as_completed(futures), 1):
            ticker = futures[future]
            try:
                result = future.result()
                results.append(result)
                
                status = result['status']
                if status == 'completed':
                    reward = result.get('best_reward')
                    ep = result.get('best_ep')
                    early = 'ES' if result.get('early_stop') else 'Full'
                    print(f"[{i}/{len(ALL_CONTRACTS)}] {ticker}: {status} | {early} | best={reward:.4f} @ ep{ep}")
                else:
                    print(f"[{i}/{len(ALL_CONTRACTS)}] {ticker}: {status} | {result.get('error', 'N/A')}")
                    
            except Exception as e:
                print(f"[{i}/{len(ALL_CONTRACTS)}] {ticker}: exception {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Training Summary")
    print(f"{'='*70}")
    
    completed = [r for r in results if r['status'] == 'completed']
    print(f"Completed: {len(completed)}/{len(ALL_CONTRACTS)}")
    
    if completed:
        rewards = [r['best_reward'] for r in completed if r['best_reward'] is not None]
        if rewards:
            print(f"Reward range: {min(rewards):.4f} ~ {max(rewards):.4f}")
            print(f"Mean reward: {sum(rewards)/len(rewards):.4f}")
        
        early_stops = [r for r in completed if r.get('early_stop')]
        print(f"Early stops: {len(early_stops)}/{len(completed)}")
    
    # Save results
    import json
    from datetime import datetime
    results_file = REPO_ROOT / f'batch_train_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
