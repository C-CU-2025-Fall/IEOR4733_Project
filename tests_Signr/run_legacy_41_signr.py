#!/usr/bin/env python3
"""One-command reproduction of the optimal Sign(R) legacy experimental frontier."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TESTS_SIGNR = ROOT / "tests_Signr"
if str(TESTS_SIGNR) not in sys.path:
    sys.path.insert(0, str(TESTS_SIGNR))

import frontier_40plus_enumeration_signr as fe  # noqa: E402


def main():
    # 使用 Sign(R)-specific 的最优配置（从 enumeration 搜索中发现）
    row = fe.scenario(
        label="legacy_experimental_SignR / Equity:risk_price_non / annual_mean_sleeve / contract_equal_path",
        family="legacy_experimental",
        overrides=fe.LEGACY_EXPERIMENTAL_OVERRIDES_SIGNR,
        excluded=fe.LEGACY_EXPERIMENTAL_EXCLUDED_SIGNR,
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
    print("Sign(R) Legacy experimental frontier (optimal Sign(R)-specific configuration)")
    print("=" * 60)
    print(f"4-asset <=10: {s['four10']}/36")
    print(f"4-asset <=15: {s['four15']}/36")
    print()
    print("source overrides (Sign(R)-specific):")
    for tk, src in sorted(fe.LEGACY_EXPERIMENTAL_OVERRIDES_SIGNR.items()):
        print(f"  {tk}: {src}")
    print()
    print("excluded:")
    print("  " + ", ".join(sorted(fe.LEGACY_EXPERIMENTAL_EXCLUDED_SIGNR)))
    print()
    print("reporting:")
    print("  Equity Index capital anchor: risk_price_non")
    print("  numerator: annual_mean_sleeve")
    print("  asset path: contract_equal_path")
    print("  all mode: contract_equal_path")
    print()
    for asset in ["Commodity", "Equity Index", "Fixed Income", "Forex"]:
        res = s["results"][asset]
        misses = ", ".join(res["misses15"]) or "none"
        print(f"{asset}: <=15 misses -> {misses}")


if __name__ == "__main__":
    main()
