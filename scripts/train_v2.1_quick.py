#!/usr/bin/env python3
"""Quick v2.1 test: FI+FX only, 50ep, patience 3, sigma=0.058, STRUCTURAL_38 preset."""
from __future__ import annotations
import os, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config import ASSET_CLASSES
from frontier_presets import STRUCTURAL_38_EXCLUDED

EXCLUDED = set(STRUCTURAL_38_EXCLUDED)

def get_done():
    done = set()
    v_root = REPO_ROOT / "drl" / "dqn" / "models" / "v2.1"
    if v_root.is_dir():
        for t in os.listdir(v_root):
            tp = v_root / t
            if not tp.is_dir(): continue
            for r in os.listdir(tp):
                rp = tp / r
                if not rp.is_dir(): continue
                for run in os.listdir(rp):
                    if (rp / run / "checkpoint.pt").exists():
                        done.add((t.upper(), r))
    return done

def train_one(task):
    ticker, round_num, asset = task
    sys.path.insert(0, str(REPO_ROOT))
    from drl.dqn.train.train_dqn_walkforward import train_contract_round
    t0 = time.time()
    try:
        ckpt, bundle = train_contract_round(
            ticker, round_num, episodes=50, early_stop_patience=3,
            model_version="v2.1", device="cuda",
            preset="structural_38", sigma_tgt=0.058,
        )
        return ticker, round_num, asset, time.time()-t0, True, ""
    except Exception as e:
        return ticker, round_num, asset, time.time()-t0, False, str(e)[:120]

def main():
    done = get_done()
    tasks = []
    for asset in ["Fixed Income", "Forex"]:
        tickers = [t for t in ASSET_CLASSES[asset] if t.upper() not in EXCLUDED]
        for tk in tickers:
            for rnd in ["r1", "r2"]:
                if (tk.upper(), rnd) not in done:
                    tasks.append((tk.upper(), int(rnd[1]), asset))
    
    print(f"v2.1 quick test: FI+FX | 50ep | patience=3 | sigma=0.058")
    print(f"Tasks: {len(tasks)} | Done: {len(done)} | Workers: 8")
    if not tasks:
        print("All done!")
        return
    
    t0 = time.time()
    ok = fail = 0
    with ProcessPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(train_one, t): t for t in tasks}
        for fut in as_completed(futs):
            tk, rnd, asset, dt, success, err = fut.result()
            n = ok + fail + 1
            if success:
                ok += 1
                print(f"  [{n}/{len(tasks)}] OK {tk} r{rnd} ({asset}) {dt:.1f}s")
            else:
                fail += 1
                print(f"  [{n}/{len(tasks)}] FAIL {tk} r{rnd} ({asset}) {dt:.1f}s -- {err}")
            elapsed = time.time() - t0
            rate = n / elapsed
            eta = (len(tasks) - n) / rate
            print(f"    OK={ok} FAIL={fail} {elapsed/60:.1f}min ETA={eta/60:.1f}min")
    
    print(f"\nDone: {ok}/{len(tasks)} in {(time.time()-t0)/60:.1f}min")

if __name__ == "__main__":
    main()
