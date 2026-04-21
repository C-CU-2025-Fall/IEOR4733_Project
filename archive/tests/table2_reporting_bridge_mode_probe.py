#!/usr/bin/env python3
"""
Compare Table 2 reporting bridge modes explicitly and save the full probe result.

This probe focuses on the currently most informative Table 2 consistency cases:
- sigma = 0.058
- port_vol_target = 0.97
- bridges:
  - constant_posthoc
  - rolling252_lagged

It writes a markdown report with:
1. Forex side-by-side 9-metric tables for both bridge families and both report modes
2. Four-asset aggregate counts for each bridge/mode pair
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_run import compute_portfolio_returns, load_contracts, compute_reporting_mdd_calmar_risk_price_sigma0
from config import METRIC_NAMES, PAPER_TABLE2
from metrics import compute_metrics
from vol_scaling import get_portfolio_bridge


ASSETS4 = ["Commodity", "Equity Index", "Fixed Income", "Forex"]
SIGMA = 0.058
PORT_VOL_TARGET = 0.97
BRIDGES = ["constant_posthoc", "rolling252_lagged"]
REPORT_SOURCE = "RISK_PRICE_SIGMA0"
MODES = ["split_world", "same_as_port_contract"]


def pct_err(ours: float, paper: float) -> float:
    if paper == 0:
        return 0.0
    return abs((ours - paper) / abs(paper)) * 100.0


def run_asset(asset: str, bridge: str, mode: str):
    raw = load_contracts(asset)
    n = len(raw)
    r = compute_portfolio_returns(raw, "Long", SIGMA, aggregation_mode="variable_n")
    r = get_portfolio_bridge(bridge, PORT_VOL_TARGET)(r)
    vals = compute_metrics(r, n_contracts=n, round_output=False)
    metrics = dict(zip(METRIC_NAMES, vals))

    reporting = compute_reporting_mdd_calmar_risk_price_sigma0(
        raw,
        sigma_tgt=SIGMA,
        strat="Long",
        port_bridge=bridge,
        port_vol_target=PORT_VOL_TARGET,
        report_bridge_mode=mode,
        round_output=False,
    )
    if reporting is not None:
        mdd_rep, calmar_rep, _ = reporting
        metrics["MDD"] = mdd_rep
        metrics["Calmar"] = calmar_rep

    paper = PAPER_TABLE2[asset]["Long"]
    errs = {k: pct_err(metrics[k], paper[k]) for k in METRIC_NAMES}
    n10 = sum(v < 10 for v in errs.values())
    n15 = sum(v < 15 for v in errs.values())
    return {
        "asset": asset,
        "contracts": n,
        "metrics": metrics,
        "paper": paper,
        "errs": errs,
        "n10": n10,
        "n15": n15,
    }


def fmt_row(vals):
    return " | ".join(vals)


def metric_table(title: str, result: dict) -> list[str]:
    lines = [
        f"## {title}",
        "",
        f"- contracts: `{result['contracts']}`",
        f"- `<=10`: `{result['n10']}/9`",
        f"- `<=15`: `{result['n15']}/9`",
        "",
        "| Metric | Ours | Paper | %Err |",
        "| --- | ---: | ---: | ---: |",
    ]
    for m in METRIC_NAMES:
        lines.append(
            fmt_row([
                m,
                f"{result['metrics'][m]:+.3f}",
                f"{result['paper'][m]:+.3f}",
                f"{result['errs'][m]:.1f}%",
            ])
            .join(["| ", " |"])
        )
    return lines


def four_asset_summary(all_results: dict[str, dict[str, dict[str, dict]]]) -> list[str]:
    lines = [
        "## Four-Asset Summary",
        "",
        "| Bridge | Mode | Commodity <=15/9 | Equity <=15/9 | Fixed Income <=15/9 | Forex <=15/9 | Total <=10/36 | Total <=15/36 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for bridge in BRIDGES:
        for mode in MODES:
            rows = [all_results[bridge][mode][asset] for asset in ASSETS4]
            total10 = sum(r["n10"] for r in rows)
            total15 = sum(r["n15"] for r in rows)
            lines.append(
                fmt_row([
                    bridge,
                    mode,
                    str(all_results[bridge][mode]["Commodity"]["n15"]),
                    str(all_results[bridge][mode]["Equity Index"]["n15"]),
                    str(all_results[bridge][mode]["Fixed Income"]["n15"]),
                    str(all_results[bridge][mode]["Forex"]["n15"]),
                    str(total10),
                    str(total15),
                ])
                .join(["| ", " |"])
            )
    return lines


def main():
    all_results: dict[str, dict[str, dict[str, dict]]] = {
        bridge: {mode: {} for mode in MODES} for bridge in BRIDGES
    }
    for bridge in BRIDGES:
        for mode in MODES:
            for asset in ASSETS4:
                all_results[bridge][mode][asset] = run_asset(asset, bridge, mode)

    out = [
        "# Table 2 Reporting Bridge Mode Probe",
        "",
        "- fixed setup:",
        f"  - `sigma = {SIGMA}`",
        f"  - `port_vol_target = {PORT_VOL_TARGET}`",
        f"  - `bridges = {', '.join(BRIDGES)}`",
        f"  - `report_source = {REPORT_SOURCE}`",
        "",
        "This report preserves the explicit side-by-side experiment instead of only a prose summary.",
        "",
    ]
    for bridge in BRIDGES:
        for mode in MODES:
            out.extend(metric_table(f"Forex — {bridge} — {mode}", all_results[bridge][mode]["Forex"]))
            out.append("")
    out.extend(four_asset_summary(all_results))
    out.append("")
    out.extend([
        "## Reproduce",
        "",
        "```bash",
        "python archive/tests/table2_reporting_bridge_mode_probe.py",
        "python baseline_run.py --table 2 --asset Forex --all-metrics --sigma 0.058 --port-vol-target 0.97 --port-bridge constant_posthoc --report-bridge-mode split_world",
        "python baseline_run.py --table 2 --asset Forex --all-metrics --sigma 0.058 --port-vol-target 0.97 --port-bridge constant_posthoc --report-bridge-mode same_as_port_contract",
        "python baseline_run.py --table 2 --asset Forex --all-metrics --sigma 0.058 --port-vol-target 0.97 --port-bridge rolling252_lagged --report-bridge-mode split_world",
        "python baseline_run.py --table 2 --asset Forex --all-metrics --sigma 0.058 --port-vol-target 0.97 --port-bridge rolling252_lagged --report-bridge-mode same_as_port_contract",
        "```",
    ])

    path = ROOT / "archive" / "docs" / "table2_reporting_bridge_mode_probe.md"
    path.write_text("\n".join(out) + "\n")
    print(path)


if __name__ == "__main__":
    main()
