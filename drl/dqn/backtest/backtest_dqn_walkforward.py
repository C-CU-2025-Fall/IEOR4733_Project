#!/usr/bin/env python3
"""Unified Table 2/3 strategy runner via the baseline backtest stack.

Supports:
  - Baseline strategies (Long, Sign(R), MACD) — no round distinction
  - DQN with per_contract (default) or asset_class training mode
  - DQN ensemble: best (default) or top3 (avg Q-values)
  - Arbitrary test periods via --test-start/--test-end for extensibility
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drl.dqn.backtest.engine import canonical_strategy_name, paper_reference, portfolio_metrics


def _sigma_from_bundle(bundle: str | None) -> float | None:
    if not bundle:
        return None
    manifest = Path(bundle) / "manifest.json"
    if not manifest.exists():
        return None
    with manifest.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    value = payload.get("sigma_tgt")
    return float(value) if value is not None else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified backtest runner (Table 2/3)")
    parser.add_argument("--strategy", default="Long", choices=["Long", "Sign(R)", "MACD", "DQN"])
    parser.add_argument("--asset", default="Forex")
    parser.add_argument("--round", type=int, default=None, help="DQN round. If omitted, stitch all rounds.")
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint path.")
    parser.add_argument("--checkpoint-bundle", default=None, help="Explicit bundle directory.")
    parser.add_argument("--run-id", default="latest")
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--progress", action="store_true", help="Show per-contract progress bars.")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--table", choices=["2", "3", "both"], default="3", help="Table 2 (train) or Table 3 (test) or both.")
    parser.add_argument("--training-mode", choices=["per_contract", "asset_class"], default="per_contract",
                        help="DQN training mode: per_contract (default) or asset_class.")
    parser.add_argument("--ensemble-mode", choices=["best", "top3"], default="best",
                        help="DQN ensemble: best seed (default) or top-3 avg Q-values.")
    parser.add_argument("--test-start", default=None, help="Override test period start (YYYY-MM-DD).")
    parser.add_argument("--test-end", default=None, help="Override test period end (YYYY-MM-DD).")
    args = parser.parse_args()

    strategy = canonical_strategy_name(args.strategy)
    effective_sigma = args.sigma
    if effective_sigma is None and strategy == "DQN":
        effective_sigma = _sigma_from_bundle(args.checkpoint_bundle)
    if effective_sigma is None:
        effective_sigma = 0.058

    tables = []
    if args.table in ("2", "both"):
        tables.append("2")
    if args.table in ("3", "both"):
        tables.append("3")

    for tbl in tables:
        metrics = portfolio_metrics(
            args.asset,
            strategy,
            round_num=args.round,
            checkpoint=args.checkpoint,
            checkpoint_bundle=args.checkpoint_bundle,
            run_id=args.run_id,
            device=args.device,
            progress=args.progress,
            batch_size=args.batch_size,
            sigma_tgt=effective_sigma,
            training_mode=args.training_mode,
            ensemble_mode=args.ensemble_mode,
            test_start=args.test_start,
            test_end=args.test_end,
        )
        paper = paper_reference(args.asset, strategy)

        print(f"\n{'=' * 70}")
        print(f"Table {tbl} — {strategy} — {args.asset}")
        print(f"sigma_tgt={effective_sigma}")
        if strategy == "DQN":
            print(f"training_mode={args.training_mode} | ensemble_mode={args.ensemble_mode}")
            print(f"device: {args.device}")
            if args.round is not None:
                print(f"round: r{args.round}")
            else:
                print("round: stitched (all rounds)")
            if args.test_start or args.test_end:
                print(f"test period: {args.test_start or 'auto'} ~ {args.test_end or 'auto'}")
        print(f"{'=' * 70}")
        for metric, ours in metrics.items():
            if paper is not None and metric in paper:
                target = paper[metric]
                err = abs((ours - target) / abs(target)) * 100 if target != 0 else 0.0
                print(f"  {metric:8s}: {ours:+.3f} vs {target:+.3f} err={err:.1f}%")
            else:
                print(f"  {metric:8s}: {ours:+.3f}")
