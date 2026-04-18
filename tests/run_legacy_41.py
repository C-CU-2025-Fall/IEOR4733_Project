#!/usr/bin/env python3
"""One-command reproduction of the current legacy experimental 41/45 frontier."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import frontier_40plus_enumeration as fe  # noqa: E402


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
    print("Legacy experimental 41/45 frontier")
    print("=" * 60)
    print(f"<=10: {s['score10']}/45")
    print(f"<=15: {s['score15']}/45")
    print()
    print("source overrides:")
    for tk, src in sorted(fe.LEGACY_EXPERIMENTAL_OVERRIDES.items()):
        print(f"  {tk}: {src}")
    print()
    print("excluded:")
    print("  " + ", ".join(sorted(fe.LEGACY_EXPERIMENTAL_EXCLUDED)))
    print()
    print("reporting:")
    print("  Equity Index capital anchor: risk_price_non")
    print("  numerator: annual_mean_sleeve")
    print("  asset path: contract_equal_path")
    print("  all mode: contract_equal_path")
    print()
    for asset in ["Commodity", "Equity Index", "Fixed Income", "Forex", "All"]:
        res = s["results"][asset]
        misses = ", ".join(res["misses15"]) or "none"
        print(f"{asset}: <=15 misses -> {misses}")


if __name__ == "__main__":
    main()
