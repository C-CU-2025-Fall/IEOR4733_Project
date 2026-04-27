#!/usr/bin/env python3
"""Batch per-contract multi-seed DQN training with parallel workers.

Usage:
  python scripts/run_all_percontract.py --parallel 4 --episodes 200 --seeds 5
  python scripts/run_all_percontract.py --parallel 4 --asset Forex  # single asset class
  python scripts/run_all_percontract.py --parallel 4 --tickers AN BN CN  # specific contracts
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from config import ASSET_CLASSES
from drl_shared.spec import current_source_policy


def get_all_tickers() -> list[str]:
    policy = current_source_policy()
    excluded = set(policy["excluded_contracts"])
    all_tickers = []
    for tickers in ASSET_CLASSES.values():
        all_tickers.extend(t for t in tickers if t not in excluded)
    return sorted(set(all_tickers))


def get_asset_tickers(asset: str) -> list[str]:
    policy = current_source_policy()
    excluded = set(policy["excluded_contracts"])
    return [t for t in ASSET_CLASSES.get(asset, []) if t not in excluded]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=4, help="Max concurrent training jobs")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--early-stop", type=int, default=0, help="Early stopping patience (0=disabled)")
    parser.add_argument("--asset", default=None, help="Single asset class (e.g. Forex)")
    parser.add_argument("--tickers", nargs="*", default=None, help="Specific tickers")
    parser.add_argument("--skip-existing", action="store_true", help="Skip contracts with best_seed.json for both rounds")
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.asset:
        tickers = get_asset_tickers(args.asset)
    else:
        tickers = get_all_tickers()

    # Build task queue: ticker × round
    tasks = []
    for ticker in tickers:
        for round_num in [1, 2]:
            # Skip if both rounds have best_seed.json and --skip-existing
            if args.skip_existing:
                best_file = REPO / "drl" / "dqn" / "models" / ticker / f"r{round_num}" / "best_seed.json"
                if best_file.exists():
                    continue
            tasks.append((ticker, round_num))

    print(f"DQN Per-Contract Training: {len(tasks)} tasks ({len(tickers)} tickers × 2 rounds)")
    print(f"Config: {args.seeds} seeds × {args.episodes} eps | parallel={args.parallel} | device={args.device}")
    print(f"{'='*60}")

    running: list[tuple[subprocess.Popen, str, int, str]] = []
    remaining = list(tasks)
    completed = 0
    failed = 0
    t0 = time.time()

    while remaining or running:
        # Fill slots
        while len(running) < args.parallel and remaining:
            ticker, round_num = remaining.pop(0)
            log_file = f"/tmp/dqn_per_{ticker}_r{round_num}.log"
            cmd = [
                sys.executable,
                str(REPO / "drl" / "dqn" / "train" / "_train_single_contract.py"),
                "--ticker", ticker,
                "--round", str(round_num),
                "--episodes", str(args.episodes),
                "--seeds", str(args.seeds),
                "--device", args.device,
            ]
            if args.early_stop > 0:
                cmd.extend(["--early-stop", str(args.early_stop)])

            proc = subprocess.Popen(
                cmd,
                stdout=open(log_file, "w"),
                stderr=subprocess.STDOUT,
                cwd=str(REPO),
            )
            running.append((proc, ticker, round_num, log_file))
            elapsed = time.time() - t0
            print(f"[{time.strftime('%H:%M:%S')}] ({elapsed:.0f}s) Started: {ticker} r{round_num} (PID {proc.pid}) | queue={len(remaining)} running={len(running)}")

        if not running:
            break

        time.sleep(15)

        still_running = []
        for proc, ticker, round_num, log in running:
            ret = proc.poll()
            if ret is not None:
                elapsed = time.time() - t0
                if ret == 0:
                    completed += 1
                    print(f"[{time.strftime('%H:%M:%S')}] ({elapsed:.0f}s) ✅ {ticker} r{round_num} | done={completed} fail={failed} left={len(remaining)+len(running)-1}")
                else:
                    failed += 1
                    print(f"[{time.strftime('%H:%M:%S')}] ({elapsed:.0f}s) ❌ {ticker} r{round_num} (exit {ret})")
                    try:
                        with open(log) as f:
                            lines = f.readlines()
                        for line in lines[-3:]:
                            print(f"    {line.rstrip()}")
                    except Exception:
                        pass
            else:
                still_running.append((proc, ticker, round_num, log))

        running = still_running

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Completed: {completed} | Failed: {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
