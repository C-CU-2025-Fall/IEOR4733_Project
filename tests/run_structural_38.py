#!/usr/bin/env python3
"""One-command reproduction of the retained cleaner-doctrine 38/45 line."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARCHIVE_TESTS = ROOT / "archive" / "tests"
if str(ARCHIVE_TESTS) not in sys.path:
    sys.path.insert(0, str(ARCHIVE_TESTS))

from baseline_run import load_contracts, run_table  # noqa: E402
from config import PAPER_TABLE2, PAPER_TABLE3  # noqa: E402
from frontier_presets import STRUCTURAL_38_EXCLUDED, STRUCTURAL_38_OVERRIDES  # noqa: E402
import frontier_40plus_enumeration as fe  # noqa: E402


OVERRIDES = STRUCTURAL_38_OVERRIDES
EXCLUDED = sorted(STRUCTURAL_38_EXCLUDED)
SIGMA = 0.06
TRADE_METRICS = ['E(R)', 'std(R)', 'DD', 'Sharpe', 'Sortino', '% +ve', 'Ave P/L']


def run_baseline_tables(table: str):
    asset_classes = ["Commodity", "Equity Index", "Fixed Income", "Forex"]
    tables = []
    if table in {"3", "both"}:
        tables.append(("Table 3", PAPER_TABLE3, None))
    if table in {"2", "both"}:
        tables.append(("Table 2", PAPER_TABLE2, 0.97))

    for table_label, paper_table, port_vol in tables:
        total10 = total15 = totaln = 0
        for ac in asset_classes:
            raw = []
            if ac == "All":
                for name in ["Commodity", "Equity Index", "Fixed Income", "Forex"]:
                    raw.extend(load_contracts(name, excluded_contracts=EXCLUDED, source_overrides=OVERRIDES))
            else:
                raw = load_contracts(ac, excluded_contracts=EXCLUDED, source_overrides=OVERRIDES)
            n10, n15, n = run_table(
                raw,
                ac,
                SIGMA,
                paper_table,
                table_label,
                port_vol_target=port_vol,
                metric_names=list(TRADE_METRICS),
                port_bridge="rolling252_lagged",
                report_source="trade",
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
    args = parser.parse_args()

    row = fe.scenario(
        label="structural_history / Equity Index:risk_price_non / wealth_cagr / contract_equal_path",
        family="coherent_override",
        overrides=OVERRIDES,
        excluded=set(EXCLUDED),
        asset_capital_overrides={"Equity Index": "risk_price_non"},
        numerator_mode="wealth_cagr",
        asset_path_mode="contract_equal_path",
        all_mode="contract_equal_path",
        same_rule=False,
        asset_specific=True,
        structural_heavy=True,
        experimental=True,
    )
    s = row["summary"]
    print("Cleaner-doctrine 38/45 frontier")
    print("=" * 60)
    print(f"<=10: {s['score10']}/45")
    print(f"<=15: {s['score15']}/45")
    print(f"excluded: {', '.join(EXCLUDED)}")
    print("source overrides:")
    for tk, src in sorted(OVERRIDES.items()):
        if tk in {"DT", "CC", "LB", "JO", "ZH"}:
            print(f"  {tk}: {src}")
    print()
    run_baseline_tables(args.table)


if __name__ == "__main__":
    main()
