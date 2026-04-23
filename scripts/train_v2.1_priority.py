#!/usr/bin/env python3
"""Train v2.1 DQN in priority order: FI+FX first, then EQ, then Commodity."""
from __future__ import annotations

import os, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config import ASSET_CLASSES
from frontier_presets import STRUCTURAL_38_EXCLUDED

EXCLUDED = set(STRUCTURAL_38_EXCLUDED)
PRESET = "structural_38"
SIGMA_TGT = 0.06
MODEL_VERSION = "v2.1"

# Priority: FI + FX first, then EQ, then Commodity
ASSET_PRIORITY = ["Fixed Income", "Forex", "Equity Index", "Commodity"]

def get_done():
    done = set()
    v_root = REPO_ROOT / "drl" / "dqn" / "models" / MODEL_VERSION
    if v_root.is_dir():
        for ticker in os.listdir(v_root):
            tpath = v_root / ticker
            if not tpath.is_dir(): continue
            for rnd_dir in os.listdir(tpath):
                rpath = tpath / rnd_dir
                if not rpath.is_dir(): continue
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
            ticker, round_num, episodes=100, early_stop_patience=5,
            model_version=MODEL_VERSION, device="cuda",
            preset=PRESET, sigma_tgt=SIGMA_TGT,
        )
        return ticker, round_num, asset, time.time() - t0, True, ""
    except Exception as e:
        return ticker, round_num, asset, time.time() - t0, False, str(e)[:120]

def main():
    done = get_done()
    
    for asset in ASSET_PRIORITY:
        tickers = [t for t in ASSET_CLASSES[asset] if t.upper() not in EXCLUDED]
        tasks = []
        for tk in tickers:
            for rnd in ["r1", "r2"]:
                if (tk.upper(), rnd) not in done:
                    tasks.append((tk.upper(), int(rnd[1]), asset))
        
        if not tasks:
            print(f"{asset}: already done, skipping")
            continue
        
        n = len(tickers) * 2
        print(f"\n{'='*60}")
        print(f"  {asset}: {len(tasks)} tasks remaining ({n} total)")
        print(f"{'='*60}")
        
        t0 = time.time()
        ok = fail = 0
        
        with ProcessPoolExecutor(max_workers=8) as pool:
            futs = {pool.submit(train_one, t): t for t in tasks}
            for fut in as_completed(futs):
                tk, rnd, asset, dt, success, err = fut.result()
                done_n = ok + fail + 1
                if success:
                    ok += 1
                    print(f"  [{done_n}/{len(tasks)}] OK {tk} r{rnd} {dt:.1f}s")
                else:
                    fail += 1
                    print(f"  [{done_n}/{len(tasks)}] FAIL {tk} r{rnd} {dt:.1f}s -- {err}")
                elapsed = time.time() - t0
                rate = done_n / elapsed
                eta = (len(tasks) - done_n) / rate
                print(f"    OK={ok} FAIL={fail} {elapsed/60:.1f}min ETA={eta/60:.1f}min")
        
        total_t = time.time() - t0
        print(f"  {asset}: {ok}/{len(tasks)} in {total_t:.0f}s")
        
        # Update done set for next asset
        done = get_done()

if __name__ == "__main__":
    main()
