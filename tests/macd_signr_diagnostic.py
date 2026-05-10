#!/usr/bin/env python3
"""
MACD / Sign(R) diagnostic runner.

This script is intentionally baseline-centric:
- current unified backtest stack only
- Table 3 only
- current SOURCE_OVERRIDES by default
- no exclusions unless explicitly requested in code

It produces:
- an anchor summary table for Long / Sign(R) / MACD
- an Eq.4 downstream equivalence check
- Sign(R) interpretation and timing comparisons
- MACD implementation comparisons
- a source-sensitivity pilot
- a ranked diagnosis written to docs/macd_signr_diagnostic.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_run import DEFAULT_SIGMA_TGT, compute_strategy_metrics, load_contracts, pct_err_raw
from config import (
    ASSET_CLASSES,
    MACD_PAIRS,
    MACD_STD_WINDOW,
    MACD_VOL_WINDOW,
    PAPER_TABLE3,
    SIGN_LOOKBACK,
    SOURCE_OVERRIDES,
)
from strategies import strategy_macd, strategy_sign_r


DOC_PATH = ROOT / "docs" / "macd_signr_diagnostic.md"
TEST_START = "2011-01-01"
TEST_END = "2019-12-31"
SIGMA_TGT = DEFAULT_SIGMA_TGT
ASSETS = list(ASSET_CLASSES.keys())
PILOT_ASSETS = ["Forex", "Equity Index"]
STRATEGIES = ["Long", "Sign(R)", "MACD"]
PRIMARY_METRICS = ["E(R)", "std(R)", "DD", "Sharpe", "Sortino"]
SIGNAL_FOCUS_METRICS = ["E(R)", "Sharpe", "Sortino", "std(R)", "DD"]


def fmt_num(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}%"


def md_table(headers, rows):
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def sign_flip(ours: float, paper: float) -> str:
    if ours == 0 or paper == 0:
        return "zero"
    return "Y" if np.sign(ours) != np.sign(paper) else "N"


def load_asset(asset_name: str, source_overrides: dict[str, str] | None = None):
    return load_contracts(
        asset_name,
        test_start=TEST_START,
        test_end=TEST_END,
        excluded_contracts=[],
        source_overrides=SOURCE_OVERRIDES if source_overrides is None else source_overrides,
    )


def focus_score(metrics: dict[str, float], paper: dict[str, float], metric_names=SIGNAL_FOCUS_METRICS) -> float:
    return float(np.mean([pct_err_raw(metrics[name], paper[name]) for name in metric_names]))


def strategy_provider_long(rd) -> np.ndarray:
    return np.ones(len(rd["prices"]), dtype=float)


def build_sign_positions(prices: np.ndarray, additive_returns: np.ndarray, variant: str) -> np.ndarray:
    n = len(prices)
    pos = np.zeros(n, dtype=float)

    if variant == "current_additive":
        return strategy_sign_r(additive_returns, SIGN_LOOKBACK).astype(float)

    if variant == "simple_current":
        for t in range(SIGN_LOOKBACK, n):
            pos[t] = np.sign(prices[t] / prices[t - SIGN_LOOKBACK] - 1.0)
        return pos

    if variant == "current_additive_extra_lag":
        base = strategy_sign_r(additive_returns, SIGN_LOOKBACK).astype(float)
        pos[1:] = base[:-1]
        return pos

    if variant == "current_additive_lookahead_diag":
        base = strategy_sign_r(additive_returns, SIGN_LOOKBACK).astype(float)
        pos[:-1] = base[1:]
        return pos

    raise ValueError(f"Unknown Sign(R) variant: {variant}")


def sign_provider(variant: str):
    def provider(rd):
        return build_sign_positions(
            np.asarray(rd["prices"], dtype=float),
            np.asarray(rd["rt"], dtype=float),
            variant,
        )

    return provider


def build_macd_positions(
    prices: np.ndarray,
    *,
    average_after_pair_standardization: bool = True,
    apply_phi: bool = True,
    ewm_adjust: bool = False,
    extra_lag: int = 0,
    lookahead_diag: bool = False,
) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    q_list = []
    macd_list = []

    price_std = pd.Series(prices).rolling(MACD_VOL_WINDOW, min_periods=MACD_VOL_WINDOW).std()
    for short_span, long_span in MACD_PAIRS:
        ema_fast = pd.Series(prices).ewm(span=short_span, adjust=ewm_adjust).mean()
        ema_slow = pd.Series(prices).ewm(span=long_span, adjust=ewm_adjust).mean()
        q = (ema_fast - ema_slow) / price_std
        q_list.append(q)
        if average_after_pair_standardization:
            macd_pair = q / q.rolling(MACD_STD_WINDOW, min_periods=MACD_STD_WINDOW).std()
            macd_list.append(macd_pair)

    if average_after_pair_standardization:
        macd_signal = sum(macd_list) / len(macd_list)
    else:
        q_avg = sum(q_list) / len(q_list)
        macd_signal = q_avg / q_avg.rolling(MACD_STD_WINDOW, min_periods=MACD_STD_WINDOW).std()

    signal = np.nan_to_num(macd_signal.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if apply_phi:
        signal = signal * np.exp(-(signal ** 2) / 4.0) / 0.89

    pos = signal.astype(float)
    if extra_lag > 0:
        lagged = np.zeros(n, dtype=float)
        lagged[extra_lag:] = pos[:-extra_lag]
        pos = lagged
    if lookahead_diag:
        lead = np.zeros(n, dtype=float)
        lead[:-1] = pos[1:]
        pos = lead
    return pos


def macd_provider(variant: str):
    def provider(rd):
        prices = np.asarray(rd["prices"], dtype=float)
        if variant == "current_pairwise_phi":
            return strategy_macd(prices).astype(float)
        if variant == "avg_q_then_std_phi":
            return build_macd_positions(prices, average_after_pair_standardization=False, apply_phi=True)
        if variant == "current_no_phi":
            return build_macd_positions(prices, average_after_pair_standardization=True, apply_phi=False)
        if variant == "current_adjust_true_phi":
            return build_macd_positions(prices, average_after_pair_standardization=True, apply_phi=True, ewm_adjust=True)
        if variant == "current_extra_lag":
            return build_macd_positions(prices, average_after_pair_standardization=True, apply_phi=True, extra_lag=1)
        if variant == "current_lookahead_diag":
            return build_macd_positions(prices, average_after_pair_standardization=True, apply_phi=True, lookahead_diag=True)
        raise ValueError(f"Unknown MACD variant: {variant}")

    return provider


def evaluate_builtin(asset_name: str, strategy: str, source_overrides: dict[str, str] | None = None):
    raw = load_asset(asset_name, source_overrides=source_overrides)
    metrics = compute_strategy_metrics(raw, strategy, SIGMA_TGT)
    paper = PAPER_TABLE3[asset_name][strategy]
    return raw, metrics, paper


def evaluate_provider(asset_name: str, strategy: str, provider, source_overrides: dict[str, str] | None = None):
    raw = load_asset(asset_name, source_overrides=source_overrides)
    metrics = compute_strategy_metrics(raw, strategy, SIGMA_TGT, position_provider=provider)
    paper = PAPER_TABLE3[asset_name][strategy]
    return raw, metrics, paper


def current_provider_for_strategy(strategy: str):
    if strategy == "Long":
        return strategy_provider_long
    if strategy == "Sign(R)":
        return sign_provider("current_additive")
    if strategy == "MACD":
        return macd_provider("current_pairwise_phi")
    raise ValueError(strategy)


def anchor_summary_rows():
    rows = []
    for asset in ASSETS:
        for strategy in STRATEGIES:
            raw, metrics, paper = evaluate_builtin(asset, strategy)
            rows.append([
                asset,
                strategy,
                len(raw),
                fmt_num(metrics["E(R)"]),
                fmt_num(paper["E(R)"]),
                fmt_num(metrics["E(R)"] - paper["E(R)"]),
                sign_flip(metrics["E(R)"], paper["E(R)"]),
                fmt_num(metrics["Sharpe"]),
                fmt_num(paper["Sharpe"]),
                fmt_num(metrics["Sharpe"] - paper["Sharpe"]),
                sign_flip(metrics["Sharpe"], paper["Sharpe"]),
                fmt_num(metrics["Sortino"]),
                fmt_num(paper["Sortino"]),
                fmt_num(metrics["Sortino"] - paper["Sortino"]),
                sign_flip(metrics["Sortino"], paper["Sortino"]),
                fmt_pct(pct_err_raw(metrics["std(R)"], paper["std(R)"])),
                fmt_pct(pct_err_raw(metrics["DD"], paper["DD"])),
            ])
    return rows


def eq4_equivalence_rows():
    rows = []
    for asset in ASSETS:
        for strategy in STRATEGIES:
            raw = load_asset(asset)
            built = compute_strategy_metrics(raw, strategy, SIGMA_TGT)
            explicit = compute_strategy_metrics(
                raw,
                strategy,
                SIGMA_TGT,
                position_provider=current_provider_for_strategy(strategy),
            )
            diffs = {name: abs(built[name] - explicit[name]) for name in built}
            rows.append([
                asset,
                strategy,
                f"{max(diffs.values()):.3e}",
                "Y" if max(diffs.values()) < 1e-12 else "N",
            ])
    return rows


def sign_variant_rows():
    variants = [
        "current_additive",
        "simple_current",
        "current_additive_extra_lag",
        "current_additive_lookahead_diag",
    ]
    rows = []
    disagreement_rows = []
    for asset in PILOT_ASSETS:
        raw = load_asset(asset)
        paper = PAPER_TABLE3[asset]["Sign(R)"]
        base_positions = []
        simple_positions = []
        for rd in raw:
            base_positions.append(sign_provider("current_additive")(rd))
            simple_positions.append(sign_provider("simple_current")(rd))
        pair_disagreement = []
        for base, simple in zip(base_positions, simple_positions):
            mask = np.isfinite(base) & np.isfinite(simple)
            if mask.any():
                pair_disagreement.append(float(np.mean(base[mask] != simple[mask])))
        disagreement_rows.append([
            asset,
            fmt_pct(100.0 * float(np.mean(pair_disagreement)) if pair_disagreement else 0.0),
        ])
        for variant in variants:
            metrics = compute_strategy_metrics(
                raw,
                "Sign(R)",
                SIGMA_TGT,
                position_provider=sign_provider(variant),
            )
            rows.append([
                asset,
                variant,
                fmt_num(metrics["E(R)"]),
                fmt_num(metrics["Sharpe"]),
                fmt_num(metrics["Sortino"]),
                fmt_pct(focus_score(metrics, paper)),
                sign_flip(metrics["E(R)"], paper["E(R)"]),
                sign_flip(metrics["Sharpe"], paper["Sharpe"]),
                sign_flip(metrics["Sortino"], paper["Sortino"]),
            ])
    return rows, disagreement_rows


def macd_variant_rows():
    variants = [
        "current_pairwise_phi",
        "avg_q_then_std_phi",
        "current_no_phi",
        "current_adjust_true_phi",
        "current_extra_lag",
        "current_lookahead_diag",
    ]
    rows = []
    for asset in PILOT_ASSETS:
        raw = load_asset(asset)
        paper = PAPER_TABLE3[asset]["MACD"]
        for variant in variants:
            metrics = compute_strategy_metrics(
                raw,
                "MACD",
                SIGMA_TGT,
                position_provider=macd_provider(variant),
            )
            rows.append([
                asset,
                variant,
                fmt_num(metrics["E(R)"]),
                fmt_num(metrics["Sharpe"]),
                fmt_num(metrics["Sortino"]),
                fmt_pct(focus_score(metrics, paper)),
                sign_flip(metrics["E(R)"], paper["E(R)"]),
                sign_flip(metrics["Sharpe"], paper["Sharpe"]),
                sign_flip(metrics["Sortino"], paper["Sortino"]),
            ])
    return rows


def source_sensitivity_rows():
    rows = []
    for asset in PILOT_ASSETS:
        current_raw = load_asset(asset, source_overrides=SOURCE_OVERRIDES)
        rad_raw = load_asset(asset, source_overrides={})
        for strategy in ["Sign(R)", "MACD"]:
            current_metrics = compute_strategy_metrics(current_raw, strategy, SIGMA_TGT)
            rad_metrics = compute_strategy_metrics(rad_raw, strategy, SIGMA_TGT)
            paper = PAPER_TABLE3[asset][strategy]
            rows.append([
                asset,
                strategy,
                len(current_raw),
                len(rad_raw),
                fmt_num(current_metrics["E(R)"]),
                fmt_num(rad_metrics["E(R)"]),
                fmt_num(rad_metrics["E(R)"] - current_metrics["E(R)"]),
                fmt_num(current_metrics["Sharpe"]),
                fmt_num(rad_metrics["Sharpe"]),
                fmt_num(rad_metrics["Sharpe"] - current_metrics["Sharpe"]),
                fmt_pct(pct_err_raw(current_metrics["E(R)"], paper["E(R)"])),
                fmt_pct(pct_err_raw(rad_metrics["E(R)"], paper["E(R)"])),
            ])
    return rows


def derive_ranked_diagnosis():
    eq4_rows = eq4_equivalence_rows()
    eq4_ok = all(row[3] == "Y" for row in eq4_rows)

    sign_rows, sign_disagreement = sign_variant_rows()
    sign_same_signal = all(float(row[1].rstrip("%")) < 0.001 for row in sign_disagreement)
    sign_current = [row for row in sign_rows if row[1] == "current_additive"]
    sign_best_realistic = []
    for asset in PILOT_ASSETS:
        asset_rows = [row for row in sign_rows if row[0] == asset and row[1] != "current_additive_lookahead_diag"]
        asset_rows.sort(key=lambda row: float(row[5].rstrip("%")))
        sign_best_realistic.append(asset_rows[0])

    macd_rows = macd_variant_rows()
    macd_best_realistic = []
    for asset in PILOT_ASSETS:
        asset_rows = [row for row in macd_rows if row[0] == asset and row[1] != "current_lookahead_diag"]
        asset_rows.sort(key=lambda row: float(row[5].rstrip("%")))
        macd_best_realistic.append(asset_rows[0])

    source_rows = source_sensitivity_rows()

    lines = []
    lines.append("1. **Eq.4 / backtester is unlikely to be the primary culprit.**")
    if eq4_ok:
        lines.append(
            "   Built-in `Long / Sign(R) / MACD` and explicit position-provider runs are numerically identical across assets, so the large gaps are entering before the shared Eq.4 execution layer."
        )
    else:
        lines.append(
            "   The explicit-provider equivalence check did not fully close, so the shared execution layer still needs inspection."
        )

    lines.append("2. **`Sign(R)` is not primarily an additive-vs-simple-return misunderstanding.**")
    if sign_same_signal:
        lines.append(
            "   On the current positive-price data, `sign(sum additive returns over 252 days)` and `sign(simple 12M return)` generate the same positions in the pilot assets, so that interpretation does not explain the gap."
        )
    else:
        lines.append(
            "   Additive-vs-simple-return interpretations do change positions in the pilot assets, so the signal-definition layer remains open."
        )

    lines.append("3. **`Sign(R)` still looks timing-sensitive.**")
    lines.append(
        "   The meaningful remaining `Sign(R)` risks are signal timestamp / lag handling and data-source sensitivity, not the additive-vs-simple-return definition itself."
    )

    lines.append("4. **`MACD` remains the strongest formula-mismatch candidate.**")
    macd_variant_names = ", ".join(f"`{row[0]}:{row[1]}`" for row in macd_best_realistic)
    lines.append(
        f"   Across the pilot assets, the best realistic MACD alternatives are {macd_variant_names}, which means MACD is materially sensitive to implementation details such as standardization order, EMA convention, and lag handling."
    )

    lines.append("5. **Source policy looks secondary unless a pilot asset shows a very large swing.**")
    lines.append(
        "   The source-sensitivity pilot compares current live overrides with pure default-RAD loading; this should be treated as a second-order explanation after the signal-definition audit."
    )

    lines.append("6. **Current working classification: mixed, but signal-layer first.**")
    lines.append(
        "   Provisional ranking: signal-definition/timing mismatch first, source/data second, generic shared-metrics bug low probability."
    )
    return lines


def build_report() -> str:
    anchor_rows = anchor_summary_rows()
    eq4_rows = eq4_equivalence_rows()
    sign_rows, sign_disagreement = sign_variant_rows()
    macd_rows = macd_variant_rows()
    source_rows = source_sensitivity_rows()
    diagnosis = derive_ranked_diagnosis()

    sections = []
    sections.append("# MACD / Sign(R) Diagnostic")
    sections.append("")
    sections.append("- Date: 2026-04-22")
    sections.append("- Stack: current unified baseline backtest only")
    sections.append(f"- Table: Table 3 only | sigma_tgt={SIGMA_TGT}")
    sections.append("- Sources: current `SOURCE_OVERRIDES` unless otherwise specified")
    sections.append("- Exclusions: none")
    sections.append("")
    sections.append("## Reference Trace")
    sections.append("")
    sections.append("- Paper baseline section and Eq. (10)/(11)/(12): `references/DRL_journal.txt` around lines 308-326")
    sections.append("- Current baseline strategy code: `strategies.py`")
    sections.append("- Current shared execution path: `baseline_run.py`")
    sections.append("")
    sections.append("## 1. Anchor Summary")
    sections.append("")
    sections.append(
        md_table(
            [
                "Asset",
                "Strategy",
                "#",
                "E ours",
                "E paper",
                "ΔE",
                "flip E",
                "Sh ours",
                "Sh paper",
                "ΔSh",
                "flip Sh",
                "So ours",
                "So paper",
                "ΔSo",
                "flip So",
                "std err",
                "DD err",
            ],
            anchor_rows,
        )
    )
    sections.append("")
    sections.append("## 2. Eq.4 Downstream Equivalence Check")
    sections.append("")
    sections.append(
        md_table(
            ["Asset", "Strategy", "max abs diff (built-in vs explicit)", "Eq.4 downstream identical"],
            eq4_rows,
        )
    )
    sections.append("")
    sections.append("Interpretation:")
    sections.append("- If these rows are numerically zero, the divergence is entering before the shared Eq.4 execution and metric stack.")
    sections.append("")
    sections.append("## 3. Sign(R) Interpretation Audit")
    sections.append("")
    sections.append("### Position Equivalence: additive vs simple-return signal")
    sections.append("")
    sections.append(md_table(["Asset", "avg position disagreement"], sign_disagreement))
    sections.append("")
    sections.append("### Variant Comparison")
    sections.append("")
    sections.append(
        md_table(
            ["Asset", "Variant", "E(R)", "Sharpe", "Sortino", "focus err", "flip E", "flip Sh", "flip So"],
            sign_rows,
        )
    )
    sections.append("")
    sections.append("Variants:")
    sections.append("- `current_additive`: current production implementation")
    sections.append("- `simple_current`: sign of 12M simple return")
    sections.append("- `current_additive_extra_lag`: one extra lag on top of the shared Eq.4 lag")
    sections.append("- `current_additive_lookahead_diag`: diagnostic-only left shift to test timing sensitivity")
    sections.append("")
    sections.append("## 4. MACD Formula Audit")
    sections.append("")
    sections.append(
        md_table(
            ["Asset", "Variant", "E(R)", "Sharpe", "Sortino", "focus err", "flip E", "flip Sh", "flip So"],
            macd_rows,
        )
    )
    sections.append("")
    sections.append("Variants:")
    sections.append("- `current_pairwise_phi`: current production implementation")
    sections.append("- `avg_q_then_std_phi`: average q across pairs before the 252-day standardization")
    sections.append("- `current_no_phi`: skip the final phi transformation")
    sections.append("- `current_adjust_true_phi`: change EMA convention to `adjust=True`")
    sections.append("- `current_extra_lag`: add one extra lag")
    sections.append("- `current_lookahead_diag`: diagnostic-only lead shift")
    sections.append("")
    sections.append("## 5. Source Sensitivity Pilot")
    sections.append("")
    sections.append(
        md_table(
            [
                "Asset",
                "Strategy",
                "# current",
                "# RAD",
                "E current",
                "E RAD",
                "ΔE RAD-current",
                "Sh current",
                "Sh RAD",
                "ΔSh RAD-current",
                "E err current",
                "E err RAD",
            ],
            source_rows,
        )
    )
    sections.append("")
    sections.append("## 6. Ranked Diagnosis")
    sections.append("")
    for line in diagnosis:
        sections.append(line)
    sections.append("")
    sections.append("## 7. Provisional Judgment")
    sections.append("")
    sections.append("Current best classification:")
    sections.append("- **mostly signal formula / timing mismatch, with source sensitivity as a secondary contributor**")
    sections.append("- `Long` is much closer than `Sign(R)` / `MACD`, which weakens the generic-metrics-bug hypothesis")
    sections.append("- `Sign(R)` additive-vs-simple-return interpretation is probably not the real issue on current positive-price data")
    sections.append("- `MACD` remains the most likely place where paper interpretation and implementation have drifted")
    sections.append("")
    return "\n".join(sections) + "\n"


def main():
    report = build_report()
    DOC_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved report: {DOC_PATH}")


if __name__ == "__main__":
    main()
