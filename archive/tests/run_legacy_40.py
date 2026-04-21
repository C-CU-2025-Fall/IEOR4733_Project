#!/usr/bin/env python3
"""One-command reproduction of the retained 40/45 cleaner experimental fallback."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TESTS = ROOT / "archive" / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from baseline_run import load_contracts, run_table  # noqa: E402
from config import METRIC_NAMES, PAPER_TABLE2, PAPER_TABLE3  # noqa: E402
from frontier_presets import LEGACY_40_EXCLUDED, LEGACY_40_OVERRIDES  # noqa: E402
import frontier_40plus_enumeration as fe  # noqa: E402


OVERRIDES = LEGACY_40_OVERRIDES
EXCLUDED = sorted(LEGACY_40_EXCLUDED)
SIGMA = 0.058


def run_baseline_tables(table: str):
    asset_classes = ["Commodity", "Equity Index", "Fixed Income", "Forex", "All"]
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
                metric_names=list(METRIC_NAMES),
                port_bridge="rolling252_lagged",
                report_source="RISK_PRICE_SIGMA0",
            )
            total10 += n10
            total15 += n15
            totaln += n
        print("\n" + "=" * 60)
        print(f"{table_label} TOTAL: <=10%: {total10}/{totaln} | <=15%: {total15}/{totaln}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", choices=["2", "3", "both"], default="3")
    args = parser.parse_args()

    row = fe.scenario(
        label="legacy40 / Equity:risk_price_non / annual_mean_sleeve / contract_equal_path",
        family="legacy_experimental",
        overrides=OVERRIDES,
        excluded=set(EXCLUDED),
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
    print("Cleaner experimental 40/45 fallback")
    print("=" * 60)
    print(f"<=10: {s['score10']}/45")
    print(f"<=15: {s['score15']}/45")
    print(f"excluded: {', '.join(EXCLUDED)}")
    print("source overrides:")
    for tk in ["CC", "DT", "EN", "JO", "LB", "ZH"]:
        if tk in OVERRIDES:
            print(f"  {tk}: {OVERRIDES[tk]}")
    print()
    run_baseline_tables(args.table)


if __name__ == "__main__":
    main()
