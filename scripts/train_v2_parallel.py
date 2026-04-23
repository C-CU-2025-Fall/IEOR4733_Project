#!/usr/bin/env python3
"""Train remaining v2 DQN models. 4 parallel processes, one per asset class."""
from __future__ import annotations

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config import ASSET_CLASSES

V2_ROOT = REPO_ROOT / "drl" / "dqn" / "models" / "v2"

def get_done():
    done = set()
    if V2_ROOT.is_dir():
        for ticker in os.listdir(V2_ROOT):
            tpath = V2_ROOT / ticker
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

def train_asset_class(asset, tickers):
    """Train all remaining contracts for one asset class. Runs in its own process."""
    done = get_done()
    tasks = []
    for tk in tickers:
        for rnd in ["r1", "r2"]:
            if (tk.upper(), rnd) not in done:
                tasks.append((tk.upper(), int(rnd[1])))
    
    if not tasks:
        return asset, 0, 0, 0
    
    t0 = time.time()
    ok = fail = 0
    
    # Import here so each process gets its own CUDA context
    sys.path.insert(0, str(REPO_ROOT))
    from drl.dqn.train.train_dqn_walkforward import train_contract_round
    
    for tk, rnd in tasks:
        try:
            train_contract_round(tk, rnd, episodes=50, early_stop_patience=3,
                               model_version="v2", device="cuda")
            ok += 1
        except Exception as e:
            print(f"  ❌ {tk} r{rnd}: {e}", flush=True)
            fail += 1
    
    dt = time.time() - t0
    return asset, ok, fail, dt

def main():
    done = get_done()
    print(f"Already done: {len(done)}/100")
    
    tasks_by_asset = {}
    for asset, tickers in ASSET_CLASSES.items():
        remaining = sum(1 for tk in tickers for rnd in ["r1","r2"] 
                       if (tk.upper(), rnd) not in done)
        if remaining > 0:
            tasks_by_asset[asset] = [t.upper() for t in tickers]
            print(f"  {asset}: {remaining} remaining")
    
    if not tasks_by_asset:
        print("All done!")
        return
    
    print(f"\nLaunching {len(tasks_by_asset)} parallel processes...")
    t0 = time.time()
    
    # Each asset class gets its own process with its own CUDA context
    # This avoids the GIL+CUDA serialization issue
    with ProcessPoolExecutor(max_workers=len(tasks_by_asset)) as pool:
        futs = {}
        for asset, tickers in tasks_by_asset.items():
            futs[pool.submit(train_asset_class, asset, tickers)] = asset
        
        for fut in as_completed(futs):
            asset, ok, fail, dt = fut.result()
            print(f"  ✅ {asset}: {ok} trained, {fail} failed, {dt:.0f}s")
    
    total = time.time() - t0
    print(f"\nTotal: {total:.0f}s ({total/60:.1f}min)")

if __name__ == "__main__":
    main()
