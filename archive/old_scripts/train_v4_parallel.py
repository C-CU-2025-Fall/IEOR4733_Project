#!/usr/bin/env python3
"""Parallel DQN training for v4 features."""
import subprocess, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "drl" / "dqn" / "train" / "train_dqn_walkforward.py"

tickers = ['AN', 'BN', 'CN', 'DX', 'FN', 'JN', 'MP', 'NK', 'SN', 'LB', 'UB']
rounds = [1, 2]
TASKS = [(tk, rnd) for tk in tickers for rnd in rounds]

print(f"Training {len(TASKS)} models: v4, 100ep, patience=20, 8 workers")

def train_one(args):
    tk, rnd = args
    cmd = [
        sys.executable, str(SCRIPT),
        "--ticker", tk, "--round", str(rnd),
        "--episodes", "100", "--early-stop", "20",
        "--device", "cuda", "--version", "v4",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    return tk, rnd, r.returncode

ok = fail = 0
with ProcessPoolExecutor(max_workers=8) as pool:
    futs = {pool.submit(train_one, t): t for t in TASKS}
    for f in as_completed(futs):
        tk, rnd, rc = f.result()
        if rc == 0:
            ok += 1
        else:
            fail += 1
        done = ok + fail
        if done % 5 == 0 or done == len(TASKS):
            print(f"  [{done}/{len(TASKS)}] ok={ok} fail={fail}")

print(f"\nDone: {ok}/{len(TASKS)} ok, {fail} fail")
