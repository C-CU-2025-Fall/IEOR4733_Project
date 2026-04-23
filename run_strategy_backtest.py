#!/usr/bin/env python3
"""Global strategy backtest CLI using the baseline-owned metric stack."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from baseline_run import EWMA_SPAN
from config import ASSET_CLASSES, BP, METRIC_NAMES
from drl.dqn.backtest.engine import current_dqn_policy, portfolio_metrics as dqn_portfolio_metrics
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="Long", choices=STRATEGIES)
    parser.add_argument("--asset", default="Forex")
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--round", type=int, default=None, help="Only for DQN. If omitted, stitch all rounds.")
    parser.add_argument("--checkpoint", default=None, help="Only for DQN; optional explicit checkpoint path.")
    parser.add_argument("--checkpoint-bundle", default=None, help="Only for DQN; explicit bundle directory.")
    parser.add_argument("--run-id", default="latest", help='Only for DQN; bundle run id or "latest".')
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto", help="Only for DQN; torch inference device.")
    parser.add_argument("--progress", action="store_true", help="Only for DQN; show inference progress bars with ETA.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Only for DQN; inference batch size.")
    parser.add_argument("--exclude-contracts", default=None, help="Comma-separated exclusions, e.g. FB,ZA")
    args = parser.parse_args()

    excluded = _parse_exclusions(args.exclude_contracts)
    effective_sigma = args.sigma
    if effective_sigma is None and args.strategy == "DQN":
        effective_sigma = _sigma_from_bundle(args.checkpoint_bundle)
    if effective_sigma is None:
        effective_sigma = 0.058

    if args.strategy == "DQN":
        if excluded:
            raise ValueError("Mainline DQN uses fixed structural_38 exclusions; remove --exclude-contracts.")
        resolved_overrides, resolved_excluded = current_dqn_policy()
        metrics_raw = dqn_portfolio_metrics(
            asset_name=args.asset,
            strategy=args.strategy,
            round_num=args.round,
            checkpoint=args.checkpoint,
            checkpoint_bundle=args.checkpoint_bundle,
            run_id=args.run_id,
            device=args.device,
            progress=args.progress,
            batch_size=args.batch_size,
            sigma_tgt=effective_sigma,
            excluded_contracts=resolved_excluded,
            source_overrides=resolved_overrides,
        )
        excluded = list(resolved_excluded or [])
        n_contracts = contract_count(args.asset, excluded_contracts=excluded, source_overrides=resolved_overrides)
    else:
        metrics_raw = backtest_strategy_metrics(
            asset_name=args.asset,
            strategy=args.strategy,
            sigma_tgt=effective_sigma,
            excluded_contracts=excluded,
            round_output=False,
        )
        n_contracts = contract_count(args.asset, excluded_contracts=excluded)
    metrics = {k: round(v, 3) for k, v in metrics_raw.items()}
    paper = paper_table3_reference(args.asset, args.strategy)
    metric_names = list(METRIC_NAMES)
    relevant_excluded = [tk for tk in excluded if tk in ASSET_CLASSES.get(args.asset, [])]
    excluded_str = ",".join(relevant_excluded) if relevant_excluded else "none"

    print(f"\n{'=' * 90}")
    print(f"  Table 3 — {args.asset} ({n_contracts} contracts)")
    print(f"  σ_tgt={effective_sigma} | EWMA({EWMA_SPAN}) | bp={BP} | excluded={excluded_str}")
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
        print(f"DQN device: {args.device}")
        if args.round is not None:
            print(f"DQN round override: r{args.round}")
        else:
            print("DQN using stitched round schedule (r1 + r2)")
        print(f"DQN run id: {args.run_id}")


if __name__ == "__main__":
    main()
