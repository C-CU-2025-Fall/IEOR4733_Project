#!/usr/bin/env python3
"""
Current baseline attribution report for Table 3 Long.

This script replaces the older archived attribution reports that were generated
under historical sigma grids such as 0.0618 / 0.0627 / 0.0630. The current
authority line is the live baseline comparison at sigma=0.058.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ASSET_CLASSES, EXCLUDED_CONTRACTS, PAPER_TABLE3, SOURCE_OVERRIDES
from baseline_run import (
    compute_portfolio_returns,
    compute_reporting_mdd_calmar_risk_price_sigma0,
)
from metrics import compute_metrics
from repro_analysis import (
    METRIC_DEFINITIONS,
    evaluate_table,
    load_asset_contracts,
    realized_er_contributions,
)


DOC_PATH = ROOT / "archive" / "docs" / "current_baseline_attribution.md"
TEST_START = "2011-01-01"
TEST_END = "2019-12-31"
SIGMA = 0.058
METRIC_DEF = METRIC_DEFINITIONS["additive_subset"]
ASSET_CLASSES_4 = ["Commodity", "Equity Index", "Fixed Income", "Forex"]
BASE_EXCLUDED = list(EXCLUDED_CONTRACTS)


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def scenario_compare(excluded_contracts):
    results = {}
    for asset in ASSET_CLASSES_4 + ["All"]:
        results[asset] = evaluate_table(
            asset,
            PAPER_TABLE3,
            METRIC_DEF,
            excluded_contracts=excluded_contracts,
            sigma_tgt=SIGMA,
            aggregation_mode="variable_n",
            source_overrides=SOURCE_OVERRIDES,
            test_start=TEST_START,
            test_end=TEST_END,
        )
    return results


def live_split_world_compare(excluded_contracts):
    results = {}
    metrics = ["E(R)", "std(R)", "DD", "Sharpe", "Sortino", "MDD", "Calmar", "% +ve", "Ave P/L"]
    for asset in ASSET_CLASSES_4 + ["All"]:
        raw = load_asset_contracts(
            asset,
            test_start=TEST_START,
            test_end=TEST_END,
            excluded_contracts=excluded_contracts,
            source_overrides=SOURCE_OVERRIDES,
        )
        port = compute_portfolio_returns(raw, "Long", SIGMA, aggregation_mode="variable_n")
        m_all = compute_metrics(port, n_contracts=len(raw))
        reporting = compute_reporting_mdd_calmar_risk_price_sigma0(raw, sigma_tgt=SIGMA, strat="Long")
        if reporting is not None:
            m_all[5], m_all[6], _ = reporting
        metric_map = dict(zip(metrics, m_all))
        paper = PAPER_TABLE3[asset]["Long"]
        percent_errors = {
            name: abs((metric_map[name] - paper[name]) / abs(paper[name])) * 100 if paper[name] != 0 else 0.0
            for name in metrics
        }
        absolute_gaps = {
            name: abs(metric_map[name] - paper[name])
            for name in metrics
        }
        results[asset] = {
            "contracts": len(raw),
            "metrics": metric_map,
            "paper": paper,
            "percent_errors": percent_errors,
            "absolute_gaps": absolute_gaps,
        }
    return results


def summarize_current_baseline(results):
    rows = []
    pass_10 = 0
    pass_15 = 0
    total = 0
    metrics = ["E(R)", "std(R)", "DD", "Sharpe", "Sortino", "MDD", "Calmar", "% +ve", "Ave P/L"]
    for asset in ASSET_CLASSES_4 + ["All"]:
        r = results[asset]
        errs = r["percent_errors"]
        n10 = sum(1 for m in metrics if errs[m] < 10)
        n15 = sum(1 for m in metrics if errs[m] < 15)
        pass_10 += n10
        pass_15 += n15
        total += len(metrics)
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


def asset_contribution_rows(asset, excluded_contracts):
    raw = load_asset_contracts(
        asset,
        test_start=TEST_START,
        test_end=TEST_END,
        excluded_contracts=excluded_contracts,
        source_overrides=SOURCE_OVERRIDES,
    )
    rows = realized_er_contributions(raw, sigma_tgt=SIGMA)
    rows.sort(key=lambda r: r["er_contrib"])
    return rows


def classify_delta(asset_gap_delta, asset_sharpe_gap_delta):
    if asset_gap_delta < 0 and asset_sharpe_gap_delta < 0:
        return "pressure_source"
    if asset_gap_delta > 0 and asset_sharpe_gap_delta > 0:
        return "supporting_fit"
    return "mixed"


def leave_one_out_rows(asset, excluded_contracts):
    baseline_asset = evaluate_table(
        asset,
        PAPER_TABLE3,
        METRIC_DEF,
        excluded_contracts=excluded_contracts,
        sigma_tgt=SIGMA,
        aggregation_mode="variable_n",
        source_overrides=SOURCE_OVERRIDES,
        test_start=TEST_START,
        test_end=TEST_END,
    )
    baseline_all = evaluate_table(
        "All",
        PAPER_TABLE3,
        METRIC_DEF,
        excluded_contracts=excluded_contracts,
        sigma_tgt=SIGMA,
        aggregation_mode="variable_n",
        source_overrides=SOURCE_OVERRIDES,
        test_start=TEST_START,
        test_end=TEST_END,
    )
    raw = load_asset_contracts(
        asset,
        test_start=TEST_START,
        test_end=TEST_END,
        excluded_contracts=excluded_contracts,
        source_overrides=SOURCE_OVERRIDES,
    )
    rows = []
    for rd in raw:
        tk = rd["tk"]
        new_excluded = list(excluded_contracts) + [tk]
        asset_result = evaluate_table(
            asset,
            PAPER_TABLE3,
            METRIC_DEF,
            excluded_contracts=new_excluded,
            sigma_tgt=SIGMA,
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
            sigma_tgt=SIGMA,
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
            "asset_gap_delta": asset_gap_delta,
            "asset_sharpe_gap_delta": asset_sharpe_gap_delta,
            "all_gap_delta": all_gap_delta,
            "all_sharpe_gap_delta": all_sharpe_gap_delta,
            "classification": classify_delta(asset_gap_delta, asset_sharpe_gap_delta),
        })
    rows.sort(key=lambda r: (r["asset_gap_delta"], r["asset_sharpe_gap_delta"], r["ticker"]))
    return rows


def current_source_map(asset):
    rows = []
    for tk in ASSET_CLASSES[asset]:
        if tk in BASE_EXCLUDED:
            continue
        rows.append([tk, SOURCE_OVERRIDES.get(tk, "RAD")])
    return rows


def section_for_asset(asset, excluded_contracts):
    contrib_rows_raw = asset_contribution_rows(asset, excluded_contracts)
    contrib_rows = []
    for r in contrib_rows_raw:
        contrib_rows.append([
            r["ticker"],
            SOURCE_OVERRIDES.get(r["ticker"], "RAD"),
            f"{r['er_contrib']:+.3f}",
            f"{r['signal_contrib']:+.3f}",
            f"{r['tc_contrib']:+.3f}",
            r["n_obs"],
        ])

    loo_raw = leave_one_out_rows(asset, excluded_contracts)
    loo_rows = []
    for r in loo_raw:
        loo_rows.append([
            r["ticker"],
            SOURCE_OVERRIDES.get(r["ticker"], "RAD"),
            f"{r['asset_gap_delta']:+.3f}",
            f"{r['asset_sharpe_gap_delta']:+.3f}",
            f"{r['all_gap_delta']:+.3f}",
            f"{r['all_sharpe_gap_delta']:+.3f}",
            r["classification"],
        ])

    best_pressure = [r["ticker"] for r in loo_raw[:5]]
    strongest_support = [r["ticker"] for r in loo_raw[-5:]]

    return "\n".join([
        f"## {asset}",
        "",
        "### Active Sources",
        "",
        md_table(["Ticker", "Source"], current_source_map(asset)),
        "",
        "### Realized E(R) Contributions",
        "",
        md_table(
            ["Ticker", "Source", "Trade contrib", "Signal contrib", "TC contrib", "Obs"],
            contrib_rows,
        ),
        "",
        "### Leave-One-Out Diagnostics",
        "",
        md_table(
            ["Ticker", "Source", "Δ asset |E(R) gap|", "Δ asset |Sharpe gap|", "Δ All |E(R) gap|", "Δ All |Sharpe gap|", "Class"],
            loo_rows,
        ),
        "",
        f"- Strongest current pressure sources: `{', '.join(best_pressure)}`",
        f"- Strongest current supporting-fit contracts: `{', '.join(strongest_support)}`",
    ])


def main():
    current = scenario_compare(BASE_EXCLUDED)
    split_world = live_split_world_compare(BASE_EXCLUDED)
    live_rows, live_pass_10, live_pass_15, live_total = summarize_current_baseline(split_world)
    additive_rows, additive_pass_10, additive_pass_15, additive_total = summarize_current_baseline(current)

    text = "\n".join([
        "# Current Baseline Attribution Report",
        "",
        "- Focus: current live Table 3 Long baseline attribution under the **current** reference sigma.",
        f"- Metric definition: `{METRIC_DEF.name}`",
        f"- Sigma: `{SIGMA}`",
        f"- Active excluded set: `{', '.join(BASE_EXCLUDED) if BASE_EXCLUDED else '(none)'}`",
        "- Source policy: current `config.SOURCE_OVERRIDES` only; no historical memory_5 preset.",
        "- This report supersedes the older attribution docs that were generated under `0.0618 / 0.0627 / 0.0630` historical sweeps.",
        "",
        "## Math Identity",
        "",
        "For variable-N aggregation,",
        "",
        "```",
        "R_port,t = (1 / N_t) * Σ_i R_i,t",
        "E(R_port) = 252 * mean_t[(1 / N_t) * Σ_i R_i,t]",
        "```",
        "",
        "So each contract has realized annualized contribution",
        "",
        "```",
        "contrib_i = 252 * mean_t[I_i,t * R_i,t / N_t]",
        "```",
        "",
        "and because `R_i,t = signal_i,t - tc_i,t`, the same identity holds for the signal and transaction-cost pieces.",
        "",
        "## Current 45-Comparison Context",
        "",
        f"- Current live split-world baseline score: `n10={live_pass_10}/{live_total}`, `n15={live_pass_15}/{live_total}`",
        f"- Trade-lane additive context used by the attribution math: `n10={additive_pass_10}/{additive_total}`, `n15={additive_pass_15}/{additive_total}`",
        "- The first table below is the current live baseline context; the second is the additive trade-lane context used by the attribution identities.",
        "",
        md_table(
            ["Asset", "#", "n10", "n15", "E(R) ours", "E(R) paper", "|E(R) gap|", "Sharpe ours", "Sharpe paper", "|Sharpe gap|"],
            live_rows,
        ),
        "",
        "### Additive Trade-Lane Context Used For Attribution",
        "",
        md_table(
            ["Asset", "#", "n10", "n15", "E(R) ours", "E(R) paper", "|E(R) gap|", "Sharpe ours", "Sharpe paper", "|Sharpe gap|"],
            additive_rows,
        ),
        "",
        section_for_asset("Commodity", BASE_EXCLUDED),
        "",
        section_for_asset("Equity Index", BASE_EXCLUDED),
        "",
        section_for_asset("Fixed Income", BASE_EXCLUDED),
        "",
        section_for_asset("Forex", BASE_EXCLUDED),
        "",
        "## Interpretation",
        "",
        "- Use this report for any current contract-level attribution discussion.",
        "- Older reports that mention `memory_5` or `sigma=0.0618/0.0627/0.0630` are historical archive only.",
        "- If we want a new exclusion or source claim, it should be justified against the deltas in this report, not the archived sigma-grid runs.",
    ])

    DOC_PATH.write_text(text + "\n", encoding="utf-8")
    print(DOC_PATH)


if __name__ == "__main__":
    main()
