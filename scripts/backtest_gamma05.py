#!/usr/bin/env python3
"""Batch backtest all gamma=0.5 DQN models (seeded only), output comparison table."""
import json, subprocess, sys, re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BT_SCRIPT = REPO / "drl" / "dqn" / "backtest" / "backtest_dqn_walkforward.py"
MODEL_BASE = REPO / "drl" / "dqn" / "models"

ASSETS_MAP = {
    "Equity_Index": "Equity Index",
    "Commodity": "Commodity",
    "Fixed_Income": "Fixed Income",
    "Forex": "Forex",
}

results = []

for asset_dir, asset_name in ASSETS_MAP.items():
    for rnd in ["r1", "r2"]:
        rnd_int = int(rnd[1])
        base = MODEL_BASE / asset_dir / rnd
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            tc_path = d / "train_config.json"
            if not tc_path.exists():
                continue
            cfg = json.load(open(tc_path))
            if cfg.get("gamma") != 0.5:
                continue
            seed = cfg.get("seed")
            if seed is None:
                continue  # skip unseeded (no checkpoint.pt)

            # Check completed
            log = d / "train.log"
            if not log.exists():
                continue
            log_text = open(log).read()
            if "Stopped: early" not in log_text and "Stopped: complete" not in log_text:
                continue

            cmd = [
                sys.executable, str(BT_SCRIPT),
                "--strategy", "DQN",
                "--asset", asset_name,
                "--round", str(rnd_int),
                "--checkpoint-bundle", str(d),
                "--device", "cuda",
            ]
            print(f"Running: {asset_name} R{rnd_int} seed={seed}", file=sys.stderr)
            try:
                out = subprocess.check_output(cmd, text=True, timeout=300)
                metrics = {}
                for line in out.strip().split("\n"):
                    m = re.match(r'\s+(\S+)\s*:\s+([+-]?\d+\.\d+)\s+vs\s+([+-]?\d+\.\d+)\s+err=([+-]?\d+\.\d+)%', line)
                    if m:
                        metrics[m.group(1)] = float(m.group(2))
                        metrics[m.group(1) + "_paper"] = float(m.group(3))
                        metrics[m.group(1) + "_err%"] = float(m.group(4))
                results.append({
                    "asset": asset_name,
                    "round": f"R{rnd_int}",
                    "seed": str(seed),
                    **metrics,
                })
            except Exception as e:
                print(f"  ERROR: {e}", file=sys.stderr)

# Print summary
print("\n" + "=" * 120)
print("DQN Gamma=0.5 Backtest Summary (3 seeds × 4 assets × 2 rounds = 24 models)")
print("=" * 120)

metric_names = ["E(R)", "std(R)", "DD", "Sharpe", "Sortino", "MDD", "Calmar", "% +ve", "Ave P/L"]

for asset_name in ASSETS_MAP.values():
    for rnd_label in ["R1", "R2"]:
        sub = [r for r in results if r["asset"] == asset_name and r["round"] == rnd_label]
        if not sub:
            continue
        print(f"\n--- {asset_name} {rnd_label} ---")
        header = f"{'seed':>6s}"
        for m in metric_names:
            if m in sub[0]:
                header += f"  {m:>10s}"
        print(header)
        for r in sub:
            line = f"{r['seed']:>6s}"
            for m in metric_names:
                if m in r:
                    line += f"  {r[m]:>+10.4f}"
            print(line)
        # Average
        line = f"{'AVG':>6s}"
        for m in metric_names:
            vals = [r[m] for r in sub if m in r]
            if vals:
                line += f"  {sum(vals)/len(vals):>+10.4f}"
            else:
                line += f"  {'---':>10s}"
        print(line)

# Per-seed summary across all assets
print("\n" + "=" * 120)
print("Per-seed average across all asset classes (R1+R2)")
print("=" * 120)
for seed in ["42", "123", "456"]:
    sub = [r for r in results if r["seed"] == seed]
    if not sub:
        continue
    line = f"  seed={seed:>4s}: "
    for m in metric_names:
        vals = [r[m] for r in sub if m in r]
        if vals:
            line += f"{m}={sum(vals)/len(vals):+.4f}  "
    print(line)

# Overall best seed
print("\n--- Best seed by metric ---")
for m in ["E(R)", "Sharpe", "Sortino", "Calmar"]:
    seed_avgs = {}
    for seed in ["42", "123", "456"]:
        vals = [r[m] for r in results if r["seed"] == seed and m in r]
        if vals:
            seed_avgs[seed] = sum(vals) / len(vals)
    if seed_avgs:
        best = max(seed_avgs, key=seed_avgs.get)
        print(f"  {m}: seed={best} ({seed_avgs[best]:+.4f})  |  s42={seed_avgs.get('42',0):+.4f}  s123={seed_avgs.get('123',0):+.4f}  s456={seed_avgs.get('456',0):+.4f}")
