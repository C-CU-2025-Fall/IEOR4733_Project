#!/usr/bin/env python3
"""Parallel v3 DQN training: 50 contracts × 2 rounds, 16 workers."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config import ASSET_CLASSES
from drl.dqn.spec import ACTIVE_MODEL_VERSION


def get_done():
    """Return set of (ticker, round_dir) that already have valid checkpoints."""
    done = set()
    v2_root = REPO_ROOT / "drl" / "dqn" / "models" / "v3"
    if v2_root.is_dir():
        for ticker in os.listdir(v2_root):
            tpath = v2_root / ticker
            if not tpath.is_dir():
                continue
            for rnd_dir in os.listdir(tpath):
                rpath = tpath / rnd_dir
                if not rpath.is_dir():
                    continue
                for run_id in os.listdir(rpath):
                    if (rpath / run_id / "checkpoint.pt").exists():
                        done.add((ticker.upper(), rnd_dir))
    return done


def train_one(task):
    """Train a single contract/round. Returns (ticker, round, dt, success, err)."""
    ticker, round_num, asset = task
    sys.path.insert(0, str(REPO_ROOT))
    from drl.dqn.train.train_dqn_walkforward import train_contract_round
    t0 = time.time()
    try:
        ckpt, bundle = train_contract_round(
            ticker, round_num,
            episodes=100, early_stop_patience=5,
            model_version="v3", device="cuda",
        )
        return ticker, round_num, asset, time.time() - t0, True, ""
    except Exception as e:
        return ticker, round_num, asset, time.time() - t0, False, str(e)[:120]


def main():
    done = get_done()

    tasks = []
    for asset, tickers in ASSET_CLASSES.items():
        for tk in tickers:
            for rnd in ["r1", "r2"]:
                if (tk.upper(), rnd) not in done:
                    tasks.append((tk.upper(), int(rnd[1]), asset))

    total = len(tasks)
    n_workers = 16
    print(f"v3 DQN Parallel Training")
    print(f"Tasks: {total} remaining | Done: {len(done)} | Workers: {n_workers}")
    print(f"Started: {time.strftime('%H:%M:%S')}")
    if total == 0:
        print("All done!")
        return

    t0 = time.time()
    ok = fail = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(train_one, t): t for t in tasks}
        for fut in as_completed(futs):
            tk, rnd, asset, dt, success, err = fut.result()
            done_n = ok + fail + 1
            if success:
                ok += 1
                print(f"  [{done_n}/{total}] ✅ {tk} r{rnd} ({asset}) {dt:.1f}s")
            else:
                fail += 1
                print(f"  [{done_n}/{total}] ❌ {tk} r{rnd} ({asset}) {dt:.1f}s — {err}")

            elapsed = time.time() - t0
            rate = done_n / elapsed
            eta = (total - done_n) / rate
            eta_str = f"{eta:.0f}s" if eta < 120 else f"{eta/60:.1f}min"
            print(f"    OK={ok} FAIL={fail} elapsed={elapsed/60:.1f}min ETA={eta_str}")

    total_t = time.time() - t0
    print(f"\nDone: {ok}/{total} in {total_t:.0f}s ({total_t/60:.1f}min), {fail} failed")


if __name__ == "__main__":
    main()
