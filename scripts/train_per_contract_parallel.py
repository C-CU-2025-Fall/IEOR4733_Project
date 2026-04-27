#!/usr/bin/env python3
"""Per-contract parallel DQN training (fresh, no checkpoint reuse)."""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from config import ASSET_CLASSES
from drl_shared.spec import universe_tickers
from drl.dqn.spec import RETRAIN_ROUNDS, current_source_policy


def _train_one(task: dict) -> dict:
    """Train a single contract+round in a subprocess to avoid GPU memory leaks."""
    import subprocess
    ticker = task["ticker"]
    round_num = task["round"]
    episodes = task["episodes"]
    device = task["device"]
    sigma_tgt = task["sigma_tgt"]
    early_stop = task.get("early_stop", 0)

    script = REPO / "drl" / "dqn" / "train" / "_train_single_contract.py"
    cmd = [
        sys.executable, str(script),
        "--ticker", ticker,
        "--round", str(round_num),
        "--episodes", str(episodes),
        "--device", device,
        "--sigma-tgt", str(sigma_tgt),
    ]
    if early_stop > 0:
        cmd.extend(["--early-stop", str(early_stop)])
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO), timeout=1200)
    elapsed = time.time() - t0
    return {
        "ticker": ticker,
        "round": round_num,
        "returncode": result.returncode,
        "elapsed": round(elapsed, 1),
        "stdout_tail": (result.stdout[-500:] if result.stdout else ""),
        "stderr_tail": (result.stderr[-500:] if result.stderr else ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Per-contract parallel DQN training")
    parser.add_argument("--asset", default="Forex", help='Asset class name or "All"')
    parser.add_argument("--round", default="both", help="1, 2, or both")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--sigma-tgt", type=float, default=None)
    parser.add_argument("--early-stop", type=int, default=0, help="Early stopping patience (0=disabled)")
    args = parser.parse_args()

    # Resolve tickers
    if args.asset == "All":
        tickers = []
        for asset_name in ASSET_CLASSES:
            tickers.extend(universe_tickers(asset_name))
    else:
        tickers = universe_tickers(args.asset)

    # Resolve rounds
    if args.round == "both":
        rounds = sorted(RETRAIN_ROUNDS)
    else:
        rounds = [int(args.round)]

    sigma_tgt = args.sigma_tgt or 0.0063  # SIGMA_TGT_DAILY default
    early_stop = args.early_stop

    policy = current_source_policy()
    excluded = set(policy.get("excluded_contracts", []))
    tickers = [t for t in tickers if t.upper() not in excluded]

    tasks = [
        {"ticker": t, "round": r, "episodes": args.episodes, "device": args.device, "sigma_tgt": sigma_tgt, "early_stop": early_stop}
        for t in tickers for r in rounds
    ]

    print(f"Per-contract DQN: {len(tasks)} tasks | {len(tickers)} contracts × {len(rounds)} rounds")
    print(f"  episodes={args.episodes} | parallel={args.parallel} | device={args.device}")
    print(f"  tickers: {tickers}")
    print()

    ok = fail = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.parallel) as pool:
        future_map = {pool.submit(_train_one, task): task for task in tasks}
        for future in as_completed(future_map):
            task = future_map[future]
            try:
                r = future.result()
                status = "OK" if r["returncode"] == 0 else "FAIL"
                if r["returncode"] != 0:
                    fail += 1
                    print(f"  {status} {r['ticker']} r{r['round']} ({r['elapsed']}s): {r['stderr_tail'][-200:]}")
                else:
                    ok += 1
                    print(f"  {status} {r['ticker']} r{r['round']} ({r['elapsed']}s)")
            except Exception as e:
                fail += 1
                print(f"  FAIL {task['ticker']} r{task['round']}: {e}")

    elapsed = time.time() - t0
    print(f"\nDone: {ok}/{len(tasks)} ok, {fail} fail, {elapsed:.0f}s total")


if __name__ == "__main__":
    main()
