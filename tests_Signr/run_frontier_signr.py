#!/usr/bin/env python3
"""
Sign(R) 策略数据源前沿搜索主程序。

遍历四个搜索族（clean_same_rule / coherent_override / structural_heavy /
legacy_experimental），对比 PAPER_TABLE3 Sign(R) 目标，找到最优数据选择组合。

用法：
    python tests_Sign(r)/run_frontier_signr.py            # 完整搜索（较慢）
    python tests_Sign(r)/run_frontier_signr.py --quick    # 仅运行 legacy_experimental 族
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SIGNR_DIR = ROOT / "tests_Sign(r)"
if str(SIGNR_DIR) not in sys.path:
    sys.path.insert(0, str(SIGNR_DIR))

import frontier_signr_enumeration as fe  # noqa: E402

MAX_SCORE = fe.MAX_SCORE  # 36


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

def _col(v: object, w: int) -> str:
    return str(v).ljust(w)


def print_summary_table(rows: list[dict], title: str, top_n: int = 10) -> None:
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    header = (
        f"{'Rank':<5} {'<=10':>6} {'<=15':>6} "
        f"{'AnnGap':>8} {'CalGap':>8}  Label"
    )
    print(header)
    print("-" * 90)
    for i, row in enumerate(rows[:top_n]):
        s = row["summary"]
        print(
            f"{i+1:<5} "
            f"{s['score10']:>4}/{MAX_SCORE}  "
            f"{s['score15']:>4}/{MAX_SCORE}  "
            f"{s['mean_ann_gap']:>8.1f}  "
            f"{s['mean_cal_gap']:>8.1f}  "
            f"{row['label']}"
        )


def print_best(row: dict, title: str = "Best scenario") -> None:
    print(f"\n{'#'*72}")
    print(f"  {title}")
    print(f"{'#'*72}")
    fe.print_scenario_detail(row)

    s = row["summary"]
    print("Data source overrides:")
    for tk, src in sorted(row["overrides"].items()):
        print(f"  {tk}: {src}")
    print()
    print("Excluded:", ", ".join(sorted(row["excluded"])))
    print()
    print("Reporting:")
    print(f"  capital_mode (default): {row['default_capital_mode']}")
    if row["asset_capital_overrides"]:
        for asset, mode in row["asset_capital_overrides"].items():
            print(f"  capital_mode ({asset}): {mode}")
    print(f"  numerator: {row['numerator_mode']}")
    print(f"  asset_path_mode: {row['asset_path_mode']}")
    print(f"  all_mode: {row['all_mode']}")
    print()
    print(f"4-asset score: <=10: {s['score10']}/{MAX_SCORE}   <=15: {s['score15']}/{MAX_SCORE}")
    print()
    print("Misses (<=15% tolerance):")
    for asset in fe.ASSETS4:
        misses = s["results"][asset]["misses15"]
        miss_str = ", ".join(misses) if misses else "none"
        print(f"  {asset}: {miss_str}")
    print()
    print("All (no paper target):")
    all_m = s["results"]["All"]["metrics"]
    print(
        f"  E(R)={all_m.get('E(R)', float('nan')):.3f}  "
        f"std={all_m.get('std(R)', float('nan')):.3f}  "
        f"Sharpe={all_m.get('Sharpe', float('nan')):.3f}  "
        f"MDD={all_m.get('MDD', float('nan')):.3f}  "
        f"Calmar={all_m.get('Calmar', float('nan')):.3f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Sign(R) frontier search")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="仅运行 legacy_experimental 族（以 Long 最优数据配置为基础）",
    )
    args = parser.parse_args()

    all_rows: list[dict] = []

    if args.quick:
        print(">> Mode: quick（仅 legacy_experimental 族）")
        rows = fe.search_legacy_experimental()
        all_rows.extend(rows)
        print_summary_table(rows, "Legacy-experimental family (Sign(R))", top_n=10)
    else:
        print(">> Mode: full search（四族全部枚举，耗时较长…）")

        print("\n[1/4] clean_same_rule …", flush=True)
        clean_rows = fe.search_clean_same_rule()
        all_rows.extend(clean_rows)
        print_summary_table(clean_rows, "clean_same_rule family (Sign(R))", top_n=5)

        print("\n[2/4] coherent_override …", flush=True)
        co_rows = fe.search_coherent_override()
        all_rows.extend(co_rows)
        print_summary_table(co_rows, "coherent_override family (Sign(R))", top_n=5)

        print("\n[3/4] structural_heavy …", flush=True)
        sh_rows = fe.search_structural_heavy()
        all_rows.extend(sh_rows)
        print_summary_table(sh_rows, "structural_heavy family (Sign(R))", top_n=5)

        print("\n[4/4] legacy_experimental …", flush=True)
        le_rows = fe.search_legacy_experimental()
        all_rows.extend(le_rows)
        print_summary_table(le_rows, "legacy_experimental family (Sign(R))", top_n=5)

    # ----------------------------------------------------------------
    # 全局最优
    # ----------------------------------------------------------------
    all_rows.sort(key=lambda r: r["summary"]["rank"])
    best = all_rows[0]

    print_best(best, title="Sign(R) — Global Best Scenario")

    # ----------------------------------------------------------------
    # 快速对比：每族的最佳代表
    # ----------------------------------------------------------------
    if not args.quick:
        print("\n--- 各族最佳代表汇总 ---")
        seen_families: dict[str, dict] = {}
        for r in all_rows:
            fam = r["family"]
            if fam not in seen_families:
                seen_families[fam] = r
        for fam, r in seen_families.items():
            s = r["summary"]
            print(
                f"  [{fam}]  <=15: {s['score15']}/{MAX_SCORE}   "
                f"label: {r['label']}"
            )
        print()


if __name__ == "__main__":
    main()
