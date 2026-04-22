#!/usr/bin/env python3
"""Global strategy backtest CLI using the baseline-owned metric stack."""
from __future__ import annotations

import argparse

from baseline_run import EWMA_SPAN
from config import BP, METRIC_NAMES
from drl.dqn.backtest.engine import dqn_position_provider
from strategy_backtester import backtest_strategy_metrics, contract_count, paper_table3_reference

STRATEGIES = ("Long", "Sign(R)", "MACD", "DQN")


def _fmt(vals: list[float]) -> str:
    return "  ".join(f"{v:>+7.3f}" for v in vals)


def _pct_err(ours: float, paper: float) -> float:
    if paper == 0:
        return 0.0
    return abs((ours - paper) / abs(paper)) * 100.0


def _parse_exclusions(exclude_contracts: str | None) -> list[str]:
    if not exclude_contracts:
        return []
    tokens = [tk.strip().upper() for tk in exclude_contracts.split(",")]
    return sorted(set(tk for tk in tokens if tk))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="Long", choices=STRATEGIES)
    parser.add_argument("--asset", default="Forex")
    parser.add_argument("--sigma", type=float, default=0.058)
    parser.add_argument("--round", type=int, default=None, help="Only for DQN. If omitted, stitch all rounds.")
    parser.add_argument("--checkpoint", default=None, help="Only for DQN; optional explicit checkpoint path.")
    parser.add_argument("--exclude-contracts", default=None, help="Comma-separated exclusions, e.g. FB,ZA")
    args = parser.parse_args()

    excluded = _parse_exclusions(args.exclude_contracts)
    provider = None
    if args.strategy == "DQN":
        provider = dqn_position_provider(round_num=args.round, checkpoint=args.checkpoint)

    metrics_raw = backtest_strategy_metrics(
        asset_name=args.asset,
        strategy=args.strategy,
        sigma_tgt=args.sigma,
        position_provider=provider,
        excluded_contracts=excluded,
        round_output=False,
    )
    metrics = {k: round(v, 3) for k, v in metrics_raw.items()}
    paper = paper_table3_reference(args.asset, args.strategy)
    n_contracts = contract_count(args.asset, excluded_contracts=excluded)
    metric_names = list(METRIC_NAMES)
    excluded_str = ",".join(excluded) if excluded else "none"

    print(f"\n{'=' * 90}")
    print(f"  Table 3 — {args.asset} ({n_contracts} contracts)")
    print(f"  σ_tgt={args.sigma} | EWMA({EWMA_SPAN}) | bp={BP} | excluded={excluded_str}")
    print(f"  Metrics: {', '.join(metric_names)}")
    print(f"{'=' * 90}")
    print()

    if paper is not None:
        ours = [metrics[m] for m in metric_names]
        paper_vals = [paper[m] for m in metric_names]
        errs = [_pct_err(metrics_raw[m], paper[m]) for m in metric_names]
        n10 = sum(1 for e in errs if e < 10)
        n15 = sum(1 for e in errs if e < 15)
        print(f"  {args.strategy:8s} (≤10%:{n10}/{len(metric_names)}  ≤15%:{n15}/{len(metric_names)})")
        print(f"  Ours  : {_fmt(ours)}")
        print(f"  Paper : {_fmt(paper_vals)}")
        print(f"  %Err  : {'  '.join(f'{e:>6.1f}%' for e in errs)}")
        return

    print(f"  {args.strategy:8s} (paper reference unavailable)")
    print(f"  Ours  : {_fmt([metrics[m] for m in metric_names])}")
    if args.strategy == "DQN":
        print()
        if args.round is not None:
            print(f"DQN round override: r{args.round}")
        else:
            print("DQN using stitched round schedule (r1 + r2)")


if __name__ == "__main__":
    main()
