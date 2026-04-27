#!/usr/bin/env python3
"""Parallel asset-class DQN training scheduler."""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import ASSET_CLASSES
from drl.dqn.train.train_dqn_walkforward import parse_rounds, train_asset_round


def _train_one(task: tuple[str, int, int, int, bool, str, float]):
    asset_name, round_num, episodes, early_stop, no_early_stop, device, sigma_tgt = task
    checkpoint, bundle = train_asset_round(
        asset_name,
        round_num,
        episodes=episodes,
        early_stop_patience=early_stop,
        no_early_stop=no_early_stop,
        device=device,
        sigma_tgt=sigma_tgt,
    )
    return {
        "asset": asset_name,
        "round": round_num,
        "checkpoint": str(checkpoint),
        "bundle": str(bundle),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="All", help='Asset class or "All"')
    parser.add_argument("--round", default="both", help='1, 2, or "both"')
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--early-stop", type=int, default=20)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--sigma-tgt", type=float, default=0.058)
    args = parser.parse_args()

    assets = list(ASSET_CLASSES) if args.asset in (None, "", "All") else [args.asset]
    rounds = parse_rounds(args.round)
    tasks = [
        (asset, rn, args.episodes, args.early_stop, args.no_early_stop, args.device, args.sigma_tgt)
        for rn in rounds
        for asset in assets
    ]
    print(f"Scheduling {len(tasks)} asset-class DQN tasks | parallel={args.parallel} | device={args.device}")

    results = []
    failures = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.parallel))) as executor:
        future_map = {executor.submit(_train_one, task): task for task in tasks}
        for future in as_completed(future_map):
            asset, rn, *_ = future_map[future]
            try:
                result = future.result()
                results.append(result)
                print(f"OK {asset} r{rn}: {result['bundle']}")
            except Exception as exc:
                failures.append({"asset": asset, "round": rn, "error": str(exc)})
                print(f"FAIL {asset} r{rn}: {exc}")

    print("\nSummary")
    print(f"  succeeded: {len(results)}")
    print(f"  failed   : {len(failures)}")
    for failure in failures:
        print(f"  - {failure['asset']} r{failure['round']}: {failure['error']}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
