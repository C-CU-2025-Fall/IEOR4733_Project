#!/usr/bin/env python3
"""Parallel DQN training — mainline, 100ep, patience 5, 8 workers."""
from __future__ import annotations
import subprocess, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "drl" / "dqn" / "train" / "train_dqn_walkforward.py"

from config import ASSET_CLASSES
from frontier_presets import STRUCTURAL_38_EXCLUDED

EXCLUDED = set(STRUCTURAL_38_EXCLUDED)
TASKS = []
for asset_name, tickers in ASSET_CLASSES.items():
    for tk in tickers:
        if tk.upper() not in EXCLUDED:
            for rnd in [1, 2]:
                TASKS.append((tk.upper(), rnd))

print(f"Training {len(TASKS)} models: 100ep, patience=20, 8 workers")

def train_one(args):
    tk, rnd = args
    cmd = [
        sys.executable, str(SCRIPT),
        "--ticker", tk, "--round", str(rnd),
        "--episodes", "100", "--early-stop", "20",
        "--device", "cuda",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    return tk, rnd, r.returncode, r.stdout[-200:] if r.stdout else "", r.stderr[-200:] if r.stderr else ""

ok = fail = 0
with ProcessPoolExecutor(max_workers=8) as pool:
    futs = {pool.submit(train_one, t): t for t in TASKS}
    for f in as_completed(futs):
        tk, rnd, rc, out, err = f.result()
        if rc == 0:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL {tk} r{rnd}: {err}")
        done = ok + fail
        if done % 10 == 0 or done == len(TASKS):
            print(f"  [{done}/{len(TASKS)}] ok={ok} fail={fail}")

print(f"\nDone: {ok}/{len(TASKS)} ok, {fail} fail")
