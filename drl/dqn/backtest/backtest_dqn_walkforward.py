#!/usr/bin/env python3
"""Table 3 strategy runner via the unified baseline backtest stack."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drl.dqn.backtest.engine import canonical_strategy_name, paper_reference, portfolio_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="Long", choices=["Long", "Sign(R)", "MACD", "DQN"])
    parser.add_argument("--asset", default="Forex")
    parser.add_argument("--round", type=int, default=None, help="Only for DQN. If omitted, stitch all rounds.")
    parser.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint path for DQN.")
    parser.add_argument("--sigma", type=float, default=0.058)
    args = parser.parse_args()

    strategy = canonical_strategy_name(args.strategy)
    metrics = portfolio_metrics(
        args.asset,
        strategy,
        round_num=args.round,
        checkpoint=args.checkpoint,
        sigma_tgt=args.sigma,
    )
    paper = paper_reference(args.asset, strategy)

    print(f"\n{'=' * 70}")
    print(f"Table 3 Backtest — {strategy} — {args.asset}")
    print(f"sigma_tgt={args.sigma}")
    if strategy == "DQN":
        if args.round is not None:
            print(f"DQN round override: r{args.round}")
        else:
            print("DQN using stitched round schedule (r1 + r2)")
    print(f"{'=' * 70}")
    for metric, ours in metrics.items():
        if paper is not None and metric in paper:
            target = paper[metric]
            err = abs((ours - target) / abs(target)) * 100 if target != 0 else 0.0
            print(f"  {metric:8s}: {ours:+.3f} vs {target:+.3f} err={err:.1f}%")
        else:
            print(f"  {metric:8s}: {ours:+.3f}")
