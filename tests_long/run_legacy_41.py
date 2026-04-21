#!/usr/bin/env python3
"""Legacy 41 data-selection run with Table 3 style output for Long only."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_strategy_report import run_single_strategy_report  # noqa: E402


<<<<<<< Updated upstream
def _fmt(vals):
    return "  ".join(f"{v:>+7.3f}" for v in vals)


def _print_table_style(s: dict) -> None:
    assets = ["Commodity", "Equity Index", "Fixed Income", "Forex", "All"]
    grand_n10, grand_n15 = 0, 0

    for asset in assets:
        res = s["results"][asset]
        metrics = res["metrics"]
        paper = fe.PAPER_TABLE3[asset]["Long"]

        ours = [metrics[m] for m in fe.METRIC_NAMES]
        pv = [paper[m] for m in fe.METRIC_NAMES]
        errs = [res["errors"][m] for m in fe.METRIC_NAMES]

        n10 = sum(1 for e in errs if e < 10)
        n15 = sum(1 for e in errs if e < 15)
        grand_n10 += n10
        grand_n15 += n15

        print(f"\n{'=' * 110}")
        print(f"  Table 3 — {asset}")
        print(f"  σ_tgt={fe.SIGMA} | EWMA(60) | bp=0.002")
        print(f"{'=' * 110}")
        print(f"\n  Long     (≤10%:{n10}/9  ≤15%:{n15}/9)")
        print(f"  Ours  : {_fmt(ours)}")
        print(f"  Paper : {_fmt(pv)}")
        print(f"  %Err  : {'  '.join(f'{e:>6.1f}%' for e in errs)}")

    print(f"\n{'=' * 60}")
    print(f"  GRAND TOTAL: ≤10%: {grand_n10}/45 | ≤15%: {grand_n15}/45")
    print(f"{'=' * 60}")


=======
# Same legacy scenario used in previous frontier scripts
LEGACY_EXPERIMENTAL_OVERRIDES = {
    "EN": "REV",
    "DT": "REV",
    "CC": "RAD",
    "LB": "REV",
    "JO": "REV",
    "ZH": "REV",
}
LEGACY_EXPERIMENTAL_EXCLUDED = {"FB", "ZA", "ZO", "EN", "ES"}
>>>>>>> Stashed changes
def main():
    parser = argparse.ArgumentParser(description="Legacy 41 Table 3-style report")
    parser.add_argument("--sigma", type=float, default=0.064, help="sigma target (default: 0.064)")
    parser.add_argument("--test-start", default="2011-01-01")
    parser.add_argument("--test-end", default="2019-12-31")
    parser.add_argument("--dataset", choices=["RAD", "NON", "REV"], default="RAD")
    args = parser.parse_args()

    run_single_strategy_report(
        strategy="Long",
        sigma_tgt=args.sigma,
        test_start=args.test_start,
        test_end=args.test_end,
        default_dataset=args.dataset,
        source_overrides=LEGACY_EXPERIMENTAL_OVERRIDES,
        excluded=LEGACY_EXPERIMENTAL_EXCLUDED,
        include_all=True,
        table_label="Table 3",
    )
<<<<<<< Updated upstream
    s = row["summary"]
    _print_table_style(s)
=======
>>>>>>> Stashed changes


if __name__ == "__main__":
    main()
