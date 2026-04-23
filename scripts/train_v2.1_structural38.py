#!/usr/bin/env python3
"""Parallel v2.1 DQN training: STRUCTURAL_38 preset, 48 contracts × 2 rounds, 16 workers.

v2.1 = v2 features (EWMA60 close deviation) + STRUCTURAL_38 preset + sigma=0.06
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config import ASSET_CLASSES
from frontier_presets import STRUCTURAL_38_EXCLUDED, STRUCTURAL_38_OVERRIDES

EXCLUDED = set(STRUCTURAL_38_EXCLUDED)
PRESET = "structural_38"
SIGMA_TGT = 0.06
MODEL_VERSION = "v2.1"
N_WORKERS = 8


def get_done():
    done = set()
    v_root = REPO_ROOT / "drl" / "dqn" / "models" / MODEL_VERSION
    if v_root.is_dir():
        for ticker in os.listdir(v_root):
            tpath = v_root / ticker
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
    ticker, round_num, asset = task
    sys.path.insert(0, str(REPO_ROOT))
    from drl.dqn.train.train_dqn_walkforward import train_contract_round
    t0 = time.time()
    try:
        ckpt, bundle = train_contract_round(
            ticker, round_num,
            episodes=100, early_stop_patience=5,
            model_version=MODEL_VERSION, device="cuda",
            preset=PRESET, sigma_tgt=SIGMA_TGT,
        )
        return ticker, round_num, asset, time.time() - t0, True, ""
    except Exception as e:
        return ticker, round_num, asset, time.time() - t0, False, str(e)[:120]


def main():
    done = get_done()
    tasks = []
    for asset, tickers in ASSET_CLASSES.items():
        for tk in tickers:
            if tk.upper() in EXCLUDED:
                continue
            for rnd in ["r1", "r2"]:
                if (tk.upper(), rnd) not in done:
                    tasks.append((tk.upper(), int(rnd[1]), asset))

    total = len(tasks)
    print(f"v2.1 DQN | STRUCTURAL_38 | sigma={SIGMA_TGT}")
    print(f"Tasks: {total} remaining | Done: {len(done)} | Workers: {N_WORKERS}")
    print(f"Excluded: {sorted(EXCLUDED)}")
    print(f"Started: {time.strftime('%H:%M:%S')}")
    if total == 0:
        print("All done!")
        return

    t0 = time.time()
    ok = fail = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = {pool.submit(train_one, t): t for t in tasks}
        for fut in as_completed(futs):
            tk, rnd, asset, dt, success, err = fut.result()
            done_n = ok + fail + 1
            if success:
                ok += 1
                print(f"  [{done_n}/{total}] OK {tk} r{rnd} ({asset}) {dt:.1f}s")
            else:
                fail += 1
                print(f"  [{done_n}/{total}] FAIL {tk} r{rnd} ({asset}) {dt:.1f}s -- {err}")

            elapsed = time.time() - t0
            rate = done_n / elapsed
            eta = (total - done_n) / rate
            eta_str = f"{eta:.0f}s" if eta < 120 else f"{eta/60:.1f}min"
            print(f"    OK={ok} FAIL={fail} elapsed={elapsed/60:.1f}min ETA={eta_str}")

    total_t = time.time() - t0
    print(f"\nDone: {ok}/{total} in {total_t:.0f}s ({total_t/60:.1f}min), {fail} failed")


if __name__ == "__main__":
    main()
