#!/usr/bin/env python3
"""
Generate current-doctrine attribution reports for the sigma band 0.058-0.060.

This script replaces the old historical attribution workflow that was tied to
memory_5 and stale sigma values. It now writes three archived attribution reports:

1. archive/docs/er_attribution_report.md
2. archive/docs/equity_contract_contribution_report.md
3. archive/docs/all_row_contribution_report.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_run import compute_portfolio_returns, compute_reporting_mdd_calmar_risk_price_sigma0
from config import ASSET_CLASSES, EXCLUDED_CONTRACTS, PAPER_TABLE3, SOURCE_OVERRIDES
from metrics import compute_metrics
from repro_analysis import (
    METRIC_DEFINITIONS,
    contract_additive_components,
    evaluate_table,
    load_asset_contracts,
)


ER_DOC = ROOT / "archive" / "docs" / "er_attribution_report.md"
EQUITY_DOC = ROOT / "archive" / "docs" / "equity_contract_contribution_report.md"
ALL_DOC = ROOT / "archive" / "docs" / "all_row_contribution_report.md"

TEST_START = "2011-01-01"
TEST_END = "2019-12-31"
SIGMA_GRID = [0.058, 0.059, 0.060]
METRIC_DEF = METRIC_DEFINITIONS["additive_subset"]
ASSET_CLASSES_4 = ["Commodity", "Equity Index", "Fixed Income", "Forex"]
BASE_EXCLUDED = list(EXCLUDED_CONTRACTS)
METRICS_9 = ["E(R)", "std(R)", "DD", "Sharpe", "Sortino", "MDD", "Calmar", "% +ve", "Ave P/L"]


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def abs_gap(a, b):
    return abs(a - b)


def pct_err(a, b):
    if abs(b) < 1e-12:
        return float("inf") if abs(a) > 1e-12 else 0.0
    return abs((a - b) / abs(b)) * 100.0


def live_split_world_compare(sigma):
    results = {}
    for asset in ASSET_CLASSES_4 + ["All"]:
        raw = load_asset_contracts(
            asset,
            test_start=TEST_START,
            test_end=TEST_END,
            excluded_contracts=BASE_EXCLUDED,
            source_overrides=SOURCE_OVERRIDES,
        )
        port = compute_portfolio_returns(raw, "Long", sigma, aggregation_mode="variable_n")
        metric_values = compute_metrics(port, n_contracts=len(raw))
        reporting = compute_reporting_mdd_calmar_risk_price_sigma0(raw, sigma_tgt=sigma, strat="Long")
        if reporting is not None:
            metric_values[5], metric_values[6], _ = reporting
        metrics = dict(zip(METRICS_9, metric_values))
        paper = PAPER_TABLE3[asset]["Long"]
        percent_errors = {name: pct_err(metrics[name], paper[name]) for name in METRICS_9}
        absolute_gaps = {name: abs_gap(metrics[name], paper[name]) for name in METRICS_9}
        results[asset] = {
            "contracts": len(raw),
            "metrics": metrics,
            "paper": paper,
            "percent_errors": percent_errors,
            "absolute_gaps": absolute_gaps,
        }
    return results


def additive_trade_compare(sigma):
    results = {}
    for asset in ASSET_CLASSES_4 + ["All"]:
        results[asset] = evaluate_table(
            asset,
            PAPER_TABLE3,
            METRIC_DEF,
            excluded_contracts=BASE_EXCLUDED,
            sigma_tgt=sigma,
            aggregation_mode="variable_n",
            source_overrides=SOURCE_OVERRIDES,
            test_start=TEST_START,
            test_end=TEST_END,
        )
    return results


def summarize_scores(results):
    pass_10 = 0
    pass_15 = 0
    total = 0
    rows = []
    for asset in ASSET_CLASSES_4 + ["All"]:
        r = results[asset]
        errs = r["percent_errors"]
        n10 = sum(1 for m in METRICS_9 if errs[m] < 10)
        n15 = sum(1 for m in METRICS_9 if errs[m] < 15)
        pass_10 += n10
        pass_15 += n15
        total += len(METRICS_9)
        rows.append([
            asset,
            r["contracts"],
            n10,
            n15,
            f"{r['metrics']['E(R)']:+.3f}",
            f"{r['paper']['E(R)']:+.3f}",
            f"{r['absolute_gaps']['E(R)']:.3f}",
            f"{r['metrics']['Sharpe']:+.3f}",
            f"{r['paper']['Sharpe']:+.3f}",
            f"{r['absolute_gaps']['Sharpe']:.3f}",
        ])
    return rows, pass_10, pass_15, total


def active_source_rows(asset):
    rows = []
    for tk in ASSET_CLASSES[asset]:
        if tk in BASE_EXCLUDED:
            continue
        rows.append([tk, SOURCE_OVERRIDES.get(tk, "RAD")])
    return rows


def contribution_decomposition(raw_data, sigma):
    trade_series = {}
    signal_series = {}
    tc_series = {}
    for rd in raw_data:
        comps = contract_additive_components(rd, strat="Long", sigma_tgt=sigma)
        tk = rd["tk"]
        trade_series[tk] = comps["trade"]
        signal_series[tk] = comps["signal"]
        tc_series[tk] = comps["tc"]

    trade_df = pd.DataFrame(trade_series)
    signal_df = pd.DataFrame(signal_series)
    tc_df = pd.DataFrame(tc_series)
    availability = trade_df.notna().sum(axis=1).replace(0, np.nan)

    rows = []
    for tk in trade_df.columns:
        contrib_series = (trade_df[tk] / availability).fillna(0.0)
        signal_contrib = (signal_df[tk] / availability).fillna(0.0)
        tc_contrib = (tc_df[tk] / availability).fillna(0.0)
        rows.append({
            "ticker": tk,
            "source": SOURCE_OVERRIDES.get(tk, "RAD"),
            "trade_contrib": float(contrib_series.mean() * 252),
            "signal_contrib": float(signal_contrib.mean() * 252),
            "tc_contrib": float(tc_contrib.mean() * 252),
            "cumulative_contrib": float(contrib_series.sum()),
            "n_obs": int(trade_df[tk].notna().sum()),
        })
    rows.sort(key=lambda r: r["trade_contrib"])
    return rows


def classify_delta(asset_gap_delta, asset_sharpe_gap_delta):
    if asset_gap_delta < 0 and asset_sharpe_gap_delta < 0:
        return "pressure_source"
    if asset_gap_delta > 0 and asset_sharpe_gap_delta > 0:
        return "supporting_fit"
    return "mixed"


def leave_one_out_rows(asset, sigma):
    baseline_asset = evaluate_table(
        asset,
        PAPER_TABLE3,
        METRIC_DEF,
        excluded_contracts=BASE_EXCLUDED,
        sigma_tgt=sigma,
        aggregation_mode="variable_n",
        source_overrides=SOURCE_OVERRIDES,
        test_start=TEST_START,
        test_end=TEST_END,
    )
    baseline_all = evaluate_table(
        "All",
        PAPER_TABLE3,
        METRIC_DEF,
        excluded_contracts=BASE_EXCLUDED,
        sigma_tgt=sigma,
        aggregation_mode="variable_n",
        source_overrides=SOURCE_OVERRIDES,
        test_start=TEST_START,
        test_end=TEST_END,
    )
    raw = load_asset_contracts(
        asset,
        test_start=TEST_START,
        test_end=TEST_END,
        excluded_contracts=BASE_EXCLUDED,
        source_overrides=SOURCE_OVERRIDES,
    )
    rows = []
    for rd in raw:
        tk = rd["tk"]
        new_excluded = list(BASE_EXCLUDED) + [tk]
        asset_result = evaluate_table(
            asset,
            PAPER_TABLE3,
            METRIC_DEF,
            excluded_contracts=new_excluded,
            sigma_tgt=sigma,
            aggregation_mode="variable_n",
            source_overrides=SOURCE_OVERRIDES,
            test_start=TEST_START,
            test_end=TEST_END,
        )
        all_result = evaluate_table(
            "All",
            PAPER_TABLE3,
            METRIC_DEF,
            excluded_contracts=new_excluded,
            sigma_tgt=sigma,
            aggregation_mode="variable_n",
            source_overrides=SOURCE_OVERRIDES,
            test_start=TEST_START,
            test_end=TEST_END,
        )
        asset_gap_delta = asset_result["absolute_gaps"]["E(R)"] - baseline_asset["absolute_gaps"]["E(R)"]
        asset_sharpe_gap_delta = asset_result["absolute_gaps"]["Sharpe"] - baseline_asset["absolute_gaps"]["Sharpe"]
        all_gap_delta = all_result["absolute_gaps"]["E(R)"] - baseline_all["absolute_gaps"]["E(R)"]
        all_sharpe_gap_delta = all_result["absolute_gaps"]["Sharpe"] - baseline_all["absolute_gaps"]["Sharpe"]
        rows.append({
            "ticker": tk,
            "source": SOURCE_OVERRIDES.get(tk, "RAD"),
            "asset_gap_delta": asset_gap_delta,
            "asset_sharpe_gap_delta": asset_sharpe_gap_delta,
            "all_gap_delta": all_gap_delta,
            "all_sharpe_gap_delta": all_sharpe_gap_delta,
            "classification": classify_delta(asset_gap_delta, asset_sharpe_gap_delta),
            "asset_gap_after_drop": asset_result["absolute_gaps"]["E(R)"],
            "asset_sharpe_gap_after_drop": asset_result["absolute_gaps"]["Sharpe"],
        })
    rows.sort(key=lambda r: (r["asset_gap_delta"], r["asset_sharpe_gap_delta"], r["ticker"]))
    return rows


def add_back_rows(sigma):
    base = additive_trade_compare(sigma)
    rows = []
    for tk in BASE_EXCLUDED:
        asset = next(name for name, tickers in ASSET_CLASSES.items() if tk in tickers)
        new_excluded = [x for x in BASE_EXCLUDED if x != tk]
        asset_result = evaluate_table(
            asset,
            PAPER_TABLE3,
            METRIC_DEF,
            excluded_contracts=new_excluded,
            sigma_tgt=sigma,
            aggregation_mode="variable_n",
            source_overrides=SOURCE_OVERRIDES,
            test_start=TEST_START,
            test_end=TEST_END,
        )
        all_result = evaluate_table(
            "All",
            PAPER_TABLE3,
            METRIC_DEF,
            excluded_contracts=new_excluded,
            sigma_tgt=sigma,
            aggregation_mode="variable_n",
            source_overrides=SOURCE_OVERRIDES,
            test_start=TEST_START,
            test_end=TEST_END,
        )
        base_asset = base[asset]
        base_all = base["All"]
        rows.append([
            tk,
            asset,
            SOURCE_OVERRIDES.get(tk, "RAD"),
            f"{asset_result['metrics']['E(R)']:+.3f}",
            f"{asset_result['absolute_gaps']['E(R)'] - base_asset['absolute_gaps']['E(R)']:+.3f}",
            f"{asset_result['absolute_gaps']['Sharpe'] - base_asset['absolute_gaps']['Sharpe']:+.3f}",
            f"{all_result['absolute_gaps']['E(R)'] - base_all['absolute_gaps']['E(R)']:+.3f}",
            f"{all_result['absolute_gaps']['Sharpe'] - base_all['absolute_gaps']['Sharpe']:+.3f}",
        ])
    return rows


def sigma_summary_rows():
    rows = []
    for sigma in SIGMA_GRID:
        live = live_split_world_compare(sigma)
        live_rows, live_n10, live_n15, live_total = summarize_scores(live)
        additive = additive_trade_compare(sigma)
        _, add_n10, add_n15, add_total = summarize_scores(additive)
        rows.append([
            f"{sigma:.3f}",
            f"{live_n10}/{live_total}",
            f"{live_n15}/{live_total}",
            f"{add_n10}/{add_total}",
            f"{add_n15}/{add_total}",
            f"{live['All']['metrics']['E(R)']:+.3f}",
            f"{live['All']['metrics']['Sharpe']:+.3f}",
        ])
    return rows


def er_report_text():
    sections = [
        "# E(R) Attribution Report",
        "",
        "- Focus: full current-baseline attribution across all 50 live contracts and all four asset classes.",
        "- Scope: Table 3 Long, current source doctrine, no historical memory_5 preset.",
        f"- Sigma band: `{', '.join(f'{x:.3f}' for x in SIGMA_GRID)}`",
        "",
        "## Sigma Sweep Summary",
        "",
        md_table(
            ["Sigma", "Live n10/45", "Live n15/45", "Additive n10/45", "Additive n15/45", "All E(R)", "All Sharpe"],
            sigma_summary_rows(),
        ),
        "",
    ]

    for sigma in SIGMA_GRID:
        live = live_split_world_compare(sigma)
        additive = additive_trade_compare(sigma)
        live_rows, live_n10, live_n15, live_total = summarize_scores(live)
        add_rows, add_n10, add_n15, add_total = summarize_scores(additive)
        sections.extend([
            f"## Sigma `{sigma:.3f}`",
            "",
            f"- Live split-world baseline: `n10={live_n10}/{live_total}`, `n15={live_n15}/{live_total}`",
            f"- Additive trade-lane attribution context: `n10={add_n10}/{add_total}`, `n15={add_n15}/{add_total}`",
            "",
            "### Live Split-World Context",
            "",
            md_table(
                ["Asset", "#", "n10", "n15", "E(R) ours", "E(R) paper", "|E(R) gap|", "Sharpe ours", "Sharpe paper", "|Sharpe gap|"],
                live_rows,
            ),
            "",
            "### Additive Trade-Lane Context",
            "",
            md_table(
                ["Asset", "#", "n10", "n15", "E(R) ours", "E(R) paper", "|E(R) gap|", "Sharpe ours", "Sharpe paper", "|Sharpe gap|"],
                add_rows,
            ),
            "",
            "### Add-Back Candidates From Current Excluded Set",
            "",
            md_table(
                ["Ticker", "Asset", "Source", "Asset E(R) after add-back", "Δ asset |E(R) gap|", "Δ asset |Sharpe gap|", "Δ All |E(R) gap|", "Δ All |Sharpe gap|"],
                add_back_rows(sigma),
            ),
            "",
        ])
        for asset in ASSET_CLASSES_4:
            raw = load_asset_contracts(
                asset,
                test_start=TEST_START,
                test_end=TEST_END,
                excluded_contracts=BASE_EXCLUDED,
                source_overrides=SOURCE_OVERRIDES,
            )
            contrib = contribution_decomposition(raw, sigma)
            loo = leave_one_out_rows(asset, sigma)
            sections.extend([
                f"### {asset}",
                "",
                "#### Active Sources",
                "",
                md_table(["Ticker", "Source"], active_source_rows(asset)),
                "",
                "#### Realized E(R) Contributions",
                "",
                md_table(
                    ["Ticker", "Source", "Trade contrib", "Signal contrib", "TC contrib", "Obs"],
                    [[r["ticker"], r["source"], f"{r['trade_contrib']:+.3f}", f"{r['signal_contrib']:+.3f}", f"{r['tc_contrib']:+.3f}", r["n_obs"]] for r in contrib],
                ),
                "",
                "#### Leave-One-Out Diagnostics",
                "",
                md_table(
                    ["Ticker", "Source", "Δ asset |E(R) gap|", "Δ asset |Sharpe gap|", "Δ All |E(R) gap|", "Δ All |Sharpe gap|", "Class"],
                    [[r["ticker"], r["source"], f"{r['asset_gap_delta']:+.3f}", f"{r['asset_sharpe_gap_delta']:+.3f}", f"{r['all_gap_delta']:+.3f}", f"{r['all_sharpe_gap_delta']:+.3f}", r["classification"]] for r in loo],
                ),
                "",
            ])
    sections.extend([
        "## Interpretation",
        "",
        "- This is the current attribution authority for the repo.",
        "- Any report mentioning `memory_5` or `0.0618 / 0.0627 / 0.0630` should be treated as archive-only.",
        "- Search and data-quality decisions should now cite the deltas in this sigma-band report, not the historical sweep logs.",
    ])
    return "\n".join(sections) + "\n"


def equity_report_text():
    sections = [
        "# Equity Contract Contribution Report",
        "",
        "- Focus: Equity-only full-contract leave-one-out attribution under the current doctrine.",
        f"- Sigma band: `{', '.join(f'{x:.3f}' for x in SIGMA_GRID)}`",
        "",
    ]
    for sigma in SIGMA_GRID:
        loo = leave_one_out_rows("Equity Index", sigma)
        sections.extend([
            f"## Sigma `{sigma:.3f}`",
            "",
            md_table(
                ["Ticker", "Source", "Δ Equity |E(R) gap|", "Δ Equity |Sharpe gap|", "Δ All |E(R) gap|", "Δ All |Sharpe gap|", "Equity |E(R) gap| after drop", "Equity |Sharpe gap| after drop", "Class"],
                [[r["ticker"], r["source"], f"{r['asset_gap_delta']:+.3f}", f"{r['asset_sharpe_gap_delta']:+.3f}", f"{r['all_gap_delta']:+.3f}", f"{r['all_sharpe_gap_delta']:+.3f}", f"{r['asset_gap_after_drop']:.3f}", f"{r['asset_sharpe_gap_after_drop']:.3f}", r["classification"]] for r in loo],
            ),
            "",
            f"- Strongest current pressure sources: `{', '.join(r['ticker'] for r in loo[:5])}`",
            f"- Strongest current supporting-fit contracts: `{', '.join(r['ticker'] for r in loo[-5:])}`",
            "",
        ])
    sections.append("- This file is now current-doctrine and no longer tied to `memory_5`.")
    return "\n".join(sections) + "\n"


def all_report_text():
    sections = [
        "# All-Row Contribution Report",
        "",
        "- Focus: all 50 included contracts and their contribution to the current `All` row.",
        f"- Sigma band: `{', '.join(f'{x:.3f}' for x in SIGMA_GRID)}`",
        "",
    ]
    for sigma in SIGMA_GRID:
        raw = load_asset_contracts(
            "All",
            test_start=TEST_START,
            test_end=TEST_END,
            excluded_contracts=BASE_EXCLUDED,
            source_overrides=SOURCE_OVERRIDES,
        )
        contrib = contribution_decomposition(raw, sigma)
        baseline_all = evaluate_table(
            "All",
            PAPER_TABLE3,
            METRIC_DEF,
            excluded_contracts=BASE_EXCLUDED,
            sigma_tgt=sigma,
            aggregation_mode="variable_n",
            source_overrides=SOURCE_OVERRIDES,
            test_start=TEST_START,
            test_end=TEST_END,
        )
        loo = leave_one_out_rows("All", sigma) if False else None
        rows = []
        for r in contrib:
            new_excluded = list(BASE_EXCLUDED) + [r["ticker"]]
            all_result = evaluate_table(
                "All",
                PAPER_TABLE3,
                METRIC_DEF,
                excluded_contracts=new_excluded,
                sigma_tgt=sigma,
                aggregation_mode="variable_n",
                source_overrides=SOURCE_OVERRIDES,
                test_start=TEST_START,
                test_end=TEST_END,
            )
            rows.append([
                r["ticker"],
                r["source"],
                f"{r['trade_contrib']:+.3f}",
                f"{r['cumulative_contrib']:+.3f}",
                f"{all_result['absolute_gaps']['E(R)'] - baseline_all['absolute_gaps']['E(R)']:+.3f}",
                f"{all_result['absolute_gaps']['Sharpe'] - baseline_all['absolute_gaps']['Sharpe']:+.3f}",
            ])
        rows.sort(key=lambda row: float(row[4]))
        sections.extend([
            f"## Sigma `{sigma:.3f}`",
            "",
            md_table(
                ["Ticker", "Source", "Annualized mean contribution", "Cumulative contribution", "Δ All |E(R) gap| if dropped", "Δ All |Sharpe gap| if dropped"],
                rows,
            ),
            "",
        ])
    sections.append("- This file is now current-doctrine and covers the full included universe, not the old memory_5 scenario.")
    return "\n".join(sections) + "\n"


def main():
    ER_DOC.write_text(er_report_text(), encoding="utf-8")
    EQUITY_DOC.write_text(equity_report_text(), encoding="utf-8")
    ALL_DOC.write_text(all_report_text(), encoding="utf-8")
    print(ER_DOC)
    print(EQUITY_DOC)
    print(ALL_DOC)


if __name__ == "__main__":
    main()
