#!/usr/bin/env python3
"""Batch DQN training: 3 gammas × N seeds × R1+R2 × 4 asset classes.

Usage:
    python scripts/dqn_batch_train.py                    # 3γ × 3 seed × 2 rounds
    python scripts/dqn_batch_train.py --extra-seeds      # +3 more seeds
    python scripts/dqn_batch_train.py --dry-run          # print plan only
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "drl" / "dqn" / "train" / "train_dqn_walkforward.py"

GAMMAS = [0.5, 0.6, 0.7]
SEEDS = [42, 123, 456]
EXTRA_SEEDS = [789, 1024, 2048]
ROUNDS = [1, 2]
ASSETS = ["Forex", "Equity Index", "Commodity", "Fixed Income"]
PARALLEL = 4  # max concurrent processes


def run_one(args_list):
    """Run a single training command."""
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(SCRIPT)] + args_list,
        capture_output=True, text=True, timeout=7200,
    )
    elapsed = time.time() - t0
    return args_list, result.returncode, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-seeds", action="store_true", help="Add 3 more seeds")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    parser.add_argument("--parallel", type=int, default=PARALLEL)
    args = parser.parse_args()

    seeds = SEEDS + (EXTRA_SEEDS if args.extra_seeds else [])
    total = len(GAMMAS) * len(seeds) * len(ROUNDS) * len(ASSETS)

    # Build all commands
    commands = []
    for gamma in GAMMAS:
        for seed in seeds:
            for rnd in ROUNDS:
                for asset in ASSETS:
                    cmd_args = [
                        "--asset", asset,
                        "--round", str(rnd),
                        "--gamma", str(gamma),
                        "--seed", str(seed),
                        "--device", "cuda",
                        "--episodes", "100",
                    ]
                    commands.append(cmd_args)

    print(f"DQN Batch Training Plan")
    print(f"  Gammas: {GAMMAS}")
    print(f"  Seeds:  {seeds}")
    print(f"  Rounds: {ROUNDS}")
    print(f"  Assets: {ASSETS}")
    print(f"  Total runs: {total}")
    print(f"  Parallel: {args.parallel}")
    print()

    if args.dry_run:
        for i, cmd in enumerate(commands):
            gamma = cmd[cmd.index("--gamma") + 1]
            seed = cmd[cmd.index("--seed") + 1]
            rnd = cmd[cmd.index("--round") + 1]
            asset = cmd[cmd.index("--asset") + 1]
            print(f"  [{i+1:>3d}/{total}] γ={gamma} seed={seed} R{rnd} {asset}")
        return

    # Run in batches of PARALLEL
    completed = 0
    failed = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        # Submit all
        futures = {executor.submit(run_one, cmd): cmd for cmd in commands}

        for future in as_completed(futures):
            cmd = futures[future]
            try:
                cmd_args, rc, elapsed = future.result()
            except Exception as e:
                completed += 1
                failed += 1
                gamma = cmd[cmd.index("--gamma") + 1]
                seed = cmd[cmd.index("--seed") + 1]
                rnd = cmd[cmd.index("--round") + 1]
                asset = cmd[cmd.index("--asset") + 1]
                print(f"  [{completed}/{total}] FAIL γ={gamma} s={seed} R{rnd} {asset}: {e}")
                continue

            completed += 1
            gamma = cmd_args[cmd_args.index("--gamma") + 1]
            seed = cmd_args[cmd_args.index("--seed") + 1]
            rnd = cmd_args[cmd_args.index("--round") + 1]
            asset = cmd_args[cmd_args.index("--asset") + 1]
            status = "✅" if rc == 0 else f"❌ rc={rc}"
            elapsed_min = elapsed / 60
            total_elapsed = (time.time() - t_start) / 60
            eta_min = (total_elapsed / completed) * (total - completed)

            print(f"  [{completed}/{total}] {status} γ={gamma} s={seed} R{rnd} {asset:>15s} "
                  f"({elapsed_min:.1f}min) total={total_elapsed:.0f}min eta={eta_min:.0f}min")

            if rc != 0:
                failed += 1

    print(f"\nDone: {completed} total, {failed} failed in {(time.time()-t_start)/60:.0f}min")


if __name__ == "__main__":
    main()
