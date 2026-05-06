#!/usr/bin/env python3
"""Batch backtest all gamma tuning models on r1 and r2."""
import json, sys, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from drl.dqn.backtest.engine import portfolio_metrics

BUNDLES = {
    (0.5, 42): "20260505T094637_s42",
    (0.5, 43): "20260505T094640_s43",
    (0.5, 44): "20260505T094644_s44",
    (0.5, 45): "20260505T094648_s45",
    (0.5, 46): "20260505T102353_s46",
    (0.6, 42): "20260505T102356_s42",
    (0.6, 43): "20260505T102358_s43",
    (0.6, 44): "20260505T102402_s44",
    (0.6, 45): "20260505T105344_s45",
    (0.6, 46): "20260505T105347_s46",
    (0.7, 42): "20260505T105350_s42",
    (0.7, 43): "20260505T105352_s43",
    (0.7, 44): "20260505T112209_s44",
    (0.7, 45): "20260505T112212_s45",
    (0.7, 46): "20260505T112214_s46",
}

OUT_DIR = REPO / "results" / "gamma_tuning"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ROOT = REPO / "drl" / "dqn" / "models" / "Forex" / "r1"

for (gamma, seed), bundle_name in sorted(BUNDLES.items()):
    bundle_dir = MODEL_ROOT / bundle_name
    if not (bundle_dir / "checkpoint.pt").exists():
        print(f"SKIP g={gamma} s={seed}: no checkpoint at {bundle_dir}")
        continue

    for round_num in [1, 2]:
        out_file = OUT_DIR / f"backtest_r{round_num}_{gamma}_{seed}.json"
        if out_file.exists():
            print(f"EXISTS {out_file.name}")
            continue

        print(f"Backtesting g={gamma} s={seed} r{round_num}...", end=" ", flush=True)
        try:
            metrics = portfolio_metrics(
                "Forex", "DQN",
                round_num=round_num,
                checkpoint_bundle=str(bundle_dir),
                device="cuda",
                progress=False,
            )
            with open(out_file, "w") as f:
                json.dump({"gamma": gamma, "seed": seed, "round": round_num, **metrics}, f)
            print(f"OK ({len(metrics)} metrics)")
        except Exception as e:
            print(f"FAIL: {e}")

print(f"\nDone. Results in {OUT_DIR}/")
