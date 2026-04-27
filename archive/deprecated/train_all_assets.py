#!/usr/bin/env python3
"""Train DQN for all asset classes with concurrency control.

Order: Fixed Income + Forex → Equity Index → Commodity
Max concurrent: 2 GPU training jobs
"""
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

# Training queue: (asset_name, round_num)
# FI + FX first (parallel), then Equity, then Commodity
QUEUE = [
    # Batch 1: FI + FX in parallel
    ("Fixed Income", 1),
    ("Forex", 1),
    # Batch 2: FI + FX round 2
    ("Fixed Income", 2),
    ("Forex", 2),
    # Batch 3: Equity Index
    ("Equity Index", 1),
    ("Equity Index", 2),
    # Batch 4: Commodity (most contracts = slowest, last)
    ("Commodity", 1),
    ("Commodity", 2),
]

MAX_CONCURRENT = 2
EPISODES = 150
DEVICE = "cuda"


def run_training(asset: str, round_num: int) -> subprocess.Popen:
    log_file = f"/tmp/dqn_{asset.replace(' ', '_').lower()}_r{round_num}.log"
    return subprocess.Popen(
        [
            sys.executable,
            str(REPO / "drl" / "dqn" / "train" / "train_dqn_walkforward.py"),
            "--asset", asset,
            "--round", str(round_num),
            "--episodes", str(EPISODES),
            "--device", DEVICE,
        ],
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
        cwd=str(REPO),
    )


def main():
    running: list[tuple[subprocess.Popen, str, int, str]] = []  # (proc, asset, round, log)
    remaining = list(QUEUE)
    completed = 0
    failed = 0

    print(f"DQN Training Pipeline — {len(QUEUE)} jobs, max {MAX_CONCURRENT} concurrent")
    print(f"{'='*60}")

    while remaining or running:
        # Fill slots
        while len(running) < MAX_CONCURRENT and remaining:
            asset, round_num = remaining.pop(0)
            proc = run_training(asset, round_num)
            log = f"/tmp/dqn_{asset.replace(' ', '_').lower()}_r{round_num}.log"
            running.append((proc, asset, round_num, log))
            print(f"[{time.strftime('%H:%M:%S')}] Started: {asset} r{round_num} (PID {proc.pid})")

        if not running:
            break

        # Poll
        time.sleep(10)

        still_running = []
        for proc, asset, round_num, log in running:
            ret = proc.poll()
            if ret is not None:
                if ret == 0:
                    completed += 1
                    print(f"[{time.strftime('%H:%M:%S')}] ✅ Done: {asset} r{round_num}")
                else:
                    failed += 1
                    print(f"[{time.strftime('%H:%M:%S')}] ❌ FAILED: {asset} r{round_num} (exit {ret})")
                    # Print last 5 lines of log
                    try:
                        with open(log) as f:
                            lines = f.readlines()
                        for line in lines[-5:]:
                            print(f"    {line.rstrip()}")
                    except Exception:
                        pass
            else:
                still_running.append((proc, asset, round_num, log))

        running = still_running

    print(f"\n{'='*60}")
    print(f"Complete: {completed}/{completed+failed} | Failed: {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
