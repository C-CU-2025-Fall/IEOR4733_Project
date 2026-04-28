#!/usr/bin/env python3
"""Run Table 3 comparison: Long vs DQN (best + top3) for all asset classes.

Outputs results to stdout and saves JSON to scripts/results/table3_comparison.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from drl.dqn.backtest.engine import portfolio_metrics, current_dqn_policy

ASSETS = ["Forex", "Commodity", "Equity Index", "Fixed Income"]
METRICS_7 = ["E(R)", "std(R)", "DD", "Sharpe", "Sortino", "% +ve", "Ave P/L"]
SIGMA_TGT = 0.058


def run_one(asset: str, strategy: str, training_mode: str = "per_contract",
            ensemble_mode: str = "best", round_num: int | None = None) -> dict | None:
    """Run one backtest, return metrics dict or None on failure."""
    try:
        overrides, excluded = current_dqn_policy()
        return portfolio_metrics(
            asset, strategy,
            round_num=round_num,
            sigma_tgt=SIGMA_TGT,
            source_overrides=overrides,
            excluded_contracts=excluded,
            training_mode=training_mode,
            ensemble_mode=ensemble_mode,
        )
    except Exception as e:
        return {"error": str(e)}


def fmt(metric: str, val: float) -> str:
    if metric in ("E(R)", "std(R)", "DD", "Sharpe", "Sortino", "Calmar"):
        return f"{val:+.3f}"
    if metric == "MDD":
        return f"{val:+.4f}"
    if metric in ("% +ve", "Ave P/L"):
        return f"{val:+.3f}"
    return f"{val:.3f}"


def main():
    results = {}
    t0 = time.time()

    print("=" * 90)
    print("Table 3 Comparison: Long vs DQN (best) vs DQN (top3) — all asset classes")
    print(f"sigma_tgt={SIGMA_TGT} | preset=drl_shared structural_38")
    print("=" * 90)

    for asset in ASSETS:
        print(f"\n{'─' * 90}")
        print(f"  {asset}")
        print(f"{'─' * 90}")

        asset_results = {}

        # Long baseline
        long_m = run_one(asset, "Long")
        asset_results["Long"] = long_m
        results[asset] = asset_results

        if long_m and "error" not in long_m:
            print(f"  Long:  ", " | ".join(f"{m}={fmt(m, long_m[m])}" for m in METRICS_7))
        else:
            print(f"  Long:  FAILED — {long_m}")
            continue

        # DQN asset-class best seed
        dqn_best = run_one(asset, "DQN", training_mode="asset_class", ensemble_mode="best")
        asset_results["DQN_best"] = dqn_best

        if dqn_best and "error" not in dqn_best:
            print(f"  DQN_b: ", " | ".join(f"{m}={fmt(m, dqn_best[m])}" for m in METRICS_7))
        else:
            print(f"  DQN_b: FAILED — {dqn_best.get('error','?') if dqn_best else '?'}")

        # DQN asset-class top3
        dqn_top3 = run_one(asset, "DQN", training_mode="asset_class", ensemble_mode="top3")
        asset_results["DQN_top3"] = dqn_top3

        if dqn_top3 and "error" not in dqn_top3:
            print(f"  DQN_t3:", " | ".join(f"{m}={fmt(m, dqn_top3[m])}" for m in METRICS_7))
        else:
            print(f"  DQN_t3: FAILED — {dqn_top3.get('error','?') if dqn_top3 else '?'}")

    elapsed = time.time() - t0

    # Save JSON
    out_dir = REPO / "scripts" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "table3_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 90}")
    print(f"Done in {elapsed:.1f}s — saved to {out_path}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
