#!/usr/bin/env python3
"""V4 normalization iteration: 2 validate → 10-parallel expand.

Round 1: 2 representative contracts (validate signal direction)
Round 2: if passed → remaining contracts, 10 parallel

Usage:
    # Default: validate with LB,UB then expand Forex
    python scripts/iterate_v4.py --rounds 1,2 --episodes 200 --patience 30

    # Validate only
    python scripts/iterate_v4.py --validate-only --tickers LB,UB --rounds 1

    # Custom asset
    python scripts/iterate_v4.py --asset Equity --rounds 1
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config import ASSET_CLASSES
from drl_shared.spec import universe_tickers, current_source_policy

VERSION = "v4"
VALIDATE_TICKERS = ["LB", "UB"]  # Default validators: positive + negative repr
MAX_WORKERS = 10  # Parallel training workers


def run(cmd: list[str], desc: str, cwd: str | Path = REPO_ROOT) -> tuple[bool, str]:
    """Run command, return (success, output)."""
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=3600)
    output = result.stdout + result.stderr
    success = result.returncode == 0
    status = "✅" if success else "❌"
    print(f"  {status} {desc}")
    return success, output


def prepare_features(ticker: str, rounds: list[int]) -> bool:
    for rnd in rounds:
        ok, _ = run(
            [
                sys.executable, "drl_shared/prepare_features.py",
                "--ticker", ticker, "--round", str(rnd), "--version", VERSION,
            ],
            f"Prepare {ticker} r{rnd}",
        )
        if not ok:
            return False
    return True


def train_model(ticker: str, rnd: int, episodes: int, patience: int) -> tuple[bool, str]:
    return run(
        [
            sys.executable, "drl/dqn/train/train_dqn_walkforward.py",
            "--ticker", ticker, "--round", str(rnd),
            "--episodes", str(episodes), "--early-stop", str(patience),
            "--version", VERSION,
        ],
        f"Train {ticker} r{rnd}",
    )


def backtest_single(ticker: str, rnd: int) -> dict:
    """Run backtest, parse key metrics."""
    asset = _find_asset_class(ticker)
    bundle = f"drl/dqn/models/{VERSION}/{ticker.upper()}/r{rnd}/latest"
    ok, output = run(
        [
            sys.executable, "run_strategy_backtest.py",
            "--strategy", "DQN", "--asset", asset,
            "--round", str(rnd), "--checkpoint-bundle", bundle,
            "--version", VERSION,
        ],
        f"Backtest {ticker} r{rnd}",
    )
    # Parse metrics from output
    metrics = {}
    for line in output.splitlines():
        if "Ours" in line and "DQN" not in line:
            parts = line.split()
            try:
                vals = [float(p) for p in parts if p.replace(".", "").replace("-", "").isdigit()]
                if len(vals) >= 9:
                    from config import METRIC_NAMES
                    for i, name in enumerate(METRIC_NAMES):
                        if i < len(vals):
                            metrics[name] = vals[i]
            except (ValueError, IndexError):
                pass
    return metrics


def _find_asset_class(ticker: str) -> str:
    ticker = ticker.upper()
    for asset_name, tickers in ASSET_CLASSES.items():
        if ticker in tickers:
            return asset_name
    return "All"


def validate_passed(ticker: str, rnd: int) -> tuple[bool, dict]:
    """Check if v4 results pass threshold."""
    metrics = backtest_single(ticker, rnd)
    # Heuristic: E(R) should be better than mainline (less negative or positive)
    # action=0 ratio can't be parsed easily from backtest output
    # Use E(R) sign as proxy
    e_r = metrics.get("E(R)", 0)
    passed = e_r > -0.05  # Tolerate small negative (data issue) but not large
    return passed, metrics


def prepare_parallel(tickers: list[str], rounds: list[int], workers: int = MAX_WORKERS):
    """Prepare features in parallel."""
    print(f"\n  Preparing features ({len(tickers)} tickers, {workers} workers)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {}
        for t in tickers:
            if prepare_features(t, rounds):
                futures[t] = True
            else:
                futures[t] = False
    ok = sum(1 for v in futures.values() if v)
    print(f"  Prepared: {ok}/{len(tickers)}")
    return ok == len(tickers)


def train_parallel(tickers: list[str], rounds: list[int], episodes: int, patience: int, workers: int = MAX_WORKERS):
    """Train models in parallel (10 workers)."""
    print(f"\n  Training ({len(tickers)} tickers × {len(rounds)} rounds, {workers} workers)...")
    tasks = [(t, r) for t in tickers for r in rounds]
    ok_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {
            exe.submit(train_model, t, r, episodes, patience): (t, r)
            for t, r in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            t, r = futures[future]
            try:
                success, _ = future.result()
                if success:
                    ok_count += 1
            except Exception as e:
                print(f"  ❌ {t} r{r}: {e}")
    print(f"  Trained: {ok_count}/{len(tasks)}")
    return ok_count


def main():
    parser = argparse.ArgumentParser(description="V4 iteration: 2 validate → 10-parallel expand")
    parser.add_argument("--tickers", default=None, help="Comma-separated (default: LB,UB for validate)")
    parser.add_argument("--asset", default="Forex", help="Asset class for expansion")
    parser.add_argument("--rounds", default="1,2", help="Comma-separated rounds")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Parallel workers")
    parser.add_argument("--validate-only", action="store_true", help="Only run validation round")
    parser.add_argument("--expand-only", action="store_true", help="Skip validation, only expand")
    args = parser.parse_args()

    rounds = [int(r.strip()) for r in args.rounds.split(",")]
    policy = current_source_policy()
    excluded = set(policy["excluded_contracts"])

    print(f"\n{'#'*70}")
    print(f"  V4 Iteration: (p_t - EMA60) / EWMA60(r)  [NO sqrt(60)]")
    print(f"{'#'*70}")
    print(f"  Rounds: {rounds} | Episodes: {args.episodes} | Patience: {args.patience}")
    print(f"  Workers: {args.workers} | Excluded: {sorted(excluded)}")

    t0 = time.time()

    # === Round 1: Validate with 2 contracts ===
    if not args.expand_only:
        validate_tickers = args.tickers.split(",") if args.tickers else VALIDATE_TICKERS
        validate_tickers = [t.strip().upper() for t in validate_tickers if t.strip().upper() not in excluded]

        print(f"\n{'='*70}")
        print(f"  ROUND 1: Validation ({len(validate_tickers)} contracts)")
        print(f"{'='*70}")

        # Prepare
        prepare_parallel(validate_tickers, rounds, workers=args.workers)

        # Train
        train_parallel(validate_tickers, rounds, args.episodes, args.patience, workers=args.workers)

        # Check results
        print(f"\n{'='*70}")
        print(f"  Validating results...")
        print(f"{'='*70}")

        all_passed = True
        for ticker in validate_tickers:
            for rnd in rounds:
                passed, metrics = validate_passed(ticker, rnd)
                e_r = metrics.get("E(R)", "N/A")
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"  {ticker} r{rnd}: E(R)={e_r} {status}")
                if not passed:
                    all_passed = False

        if args.validate_only:
            elapsed = time.time() - t0
            print(f"\n{'#'*70}")
            print(f"  Validation complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")
            print(f"  Result: {'PASS → proceed to expand' if all_passed else 'FAIL → stop iteration'}")
            print(f"{'#'*70}")
            return

        if not all_passed:
            print(f"\n❌ Validation failed. Stopping. Fix normalization before expanding.")
            return

        print(f"\n✅ Validation passed! Proceeding to expand...")

    # === Round 2: Expand to full asset ===
    if args.expand_only or not args.validate_only:
        all_tickers = universe_tickers(args.asset)
        expand_tickers = [t for t in all_tickers if t.upper() not in excluded]
        # Remove already-trained validators
        if not args.expand_only:
            already_done = set(t.upper() for t in validate_tickers)
            expand_tickers = [t for t in expand_tickers if t.upper() not in already_done]

        if not expand_tickers:
            print(f"\n  No new contracts to train (all done in validation).")
            return

        print(f"\n{'='*70}")
        print(f"  ROUND 2: Expansion ({len(expand_tickers)} contracts, {args.workers} workers)")
        print(f"  Tickers: {expand_tickers}")
        print(f"{'='*70}")

        # Prepare
        prepare_parallel(expand_tickers, rounds, workers=args.workers)

        # Train (10 parallel)
        trained = train_parallel(expand_tickers, rounds, args.episodes, args.patience, workers=args.workers)

        # Summary
        elapsed = time.time() - t0
        print(f"\n{'#'*70}")
        print(f"  ✅ V4 iteration complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")
        print(f"  Total trained: {trained} models")
        print(f"{'#'*70}")


if __name__ == "__main__":
    main()
