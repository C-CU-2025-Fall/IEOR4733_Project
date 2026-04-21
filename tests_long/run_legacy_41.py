#!/usr/bin/env python3
"""One-command reproduction of the current legacy experimental 41/45 frontier."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TESTS = ROOT / "tests_long"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import frontier_40plus_enumeration as fe  # noqa: E402


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


def main():
    row = fe.scenario(
        label="legacy_experimental / Equity:risk_price_non / annual_mean_sleeve / contract_equal_path",
        family="legacy_experimental",
        overrides=fe.LEGACY_EXPERIMENTAL_OVERRIDES,
        excluded=fe.LEGACY_EXPERIMENTAL_EXCLUDED,
        asset_capital_overrides={"Equity Index": "risk_price_non"},
        numerator_mode="annual_mean_sleeve",
        asset_path_mode="contract_equal_path",
        all_mode="contract_equal_path",
        same_rule=False,
        asset_specific=True,
        structural_heavy=True,
        experimental=True,
    )
    s = row["summary"]
    _print_table_style(s)


if __name__ == "__main__":
    main()
