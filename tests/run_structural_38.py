#!/usr/bin/env python3
"""One-command baseline reproduction for the retained structural-38 configuration."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_run import load_contracts, run_table  # noqa: E402
from config import PAPER_TABLE2, PAPER_TABLE3  # noqa: E402
from frontier_presets import STRUCTURAL_38_EXCLUDED, STRUCTURAL_38_OVERRIDES  # noqa: E402


OVERRIDES = STRUCTURAL_38_OVERRIDES
EXCLUDED = sorted(STRUCTURAL_38_EXCLUDED)
SIGMA = 0.06
TRADE_METRICS = ['E(R)', 'std(R)', 'DD', 'Sharpe', 'Sortino', '% +ve', 'Ave P/L']
PATH_METRICS = ['MDD', 'Calmar']


def run_baseline_tables(table: str, with_path_metrics: bool = False):
    asset_classes = ["Commodity", "Equity Index", "Fixed Income", "Forex"]
    tables = []
    if table in {"3", "both"}:
        tables.append(("Table 3", PAPER_TABLE3, None))
    if table in {"2", "both"}:
        tables.append(("Table 2", PAPER_TABLE2, 0.97))
    metric_names = list(TRADE_METRICS)
    if with_path_metrics:
        metric_names = metric_names[:5] + PATH_METRICS + metric_names[5:]

    for table_label, paper_table, port_vol in tables:
        total10 = total15 = totaln = 0
        for ac in asset_classes:
            raw = load_contracts(ac, excluded_contracts=EXCLUDED, source_overrides=OVERRIDES)
            n10, n15, n = run_table(
                raw,
                ac,
                SIGMA,
                paper_table,
                table_label,
                port_vol_target=port_vol,
                metric_names=metric_names,
                port_bridge="rolling252_lagged",
            )
            total10 += n10
            total15 += n15
            totaln += n
        print("\n" + "=" * 60)
        print(f"{table_label} TOTAL: <=10%: {total10}/{totaln} | <=15%: {total15}/{totaln}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", choices=["2", "3", "both"], default="both")
    parser.add_argument(
        "--with-path-metrics",
        action="store_true",
        help="Also show portfolio-path MDD and Calmar from the current unified backtest stack.",
    )
    args = parser.parse_args()
    print("Structural-38 baseline")
    print("=" * 60)
    print(f"excluded: {', '.join(EXCLUDED)}")
    print("source overrides:")
    for tk, src in sorted(OVERRIDES.items()):
        print(f"  {tk}: {src}")
    print()
    run_baseline_tables(args.table, with_path_metrics=args.with_path_metrics)


if __name__ == "__main__":
    main()
