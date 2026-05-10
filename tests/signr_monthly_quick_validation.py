#!/usr/bin/env python3
"""
Quick validation for Sign(R):

Compare the current repo's daily 252-day sign signal against a
Moskowitz-style monthly 12M signal that is held for the next month.
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
from config import ASSET_CLASSES, PAPER_TABLE3, SOURCE_OVERRIDES
from data_loader import load_clc_full
from strategies import strategy_sign_r


DOC_PATH = ROOT / "docs" / "signr_monthly_quick_validation.md"
SIGMA_TGT = DEFAULT_SIGMA_TGT
TEST_START = "2011-01-01"
TEST_END = "2019-12-31"
ASSETS = ["Forex", "Equity Index", "Fixed Income", "Commodity"]


def md_table(headers, rows):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def fmt_num(x: float) -> str:
    return f"{x:+.3f}"


def fmt_pct(x: float) -> str:
    return f"{x:.1f}%"


def focus_err(metrics: dict[str, float], paper: dict[str, float]) -> float:
    names = ["E(R)", "Sharpe", "Sortino", "std(R)", "DD"]
    return float(np.mean([pct_err_raw(metrics[name], paper[name]) for name in names]))


def _full_contract_frame(ticker: str, source: str) -> pd.DataFrame:
    df = load_clc_full(ticker, source=source, anchor_date=TEST_START)
    if df is None:
        raise ValueError(f"Missing full frame for {ticker} ({source})")
    return df.reset_index(drop=True)


def build_monthly_next_positions(prices: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    series = pd.Series(np.asarray(prices, dtype=float), index=pd.DatetimeIndex(dates))
    month_close = series.resample("ME").last()
    month_signal = np.sign(month_close / month_close.shift(12) - 1.0).fillna(0.0)
    signal_by_period = pd.Series(month_signal.values, index=month_signal.index.to_period("M"))

    out = np.zeros(len(series), dtype=float)
    daily_periods = series.index.to_period("M")
    for i, period in enumerate(daily_periods):
        prev_period = period - 1
        out[i] = float(signal_by_period.get(prev_period, 0.0))
    return out


def build_monthly_same_month_diag_positions(prices: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    series = pd.Series(np.asarray(prices, dtype=float), index=pd.DatetimeIndex(dates))
    month_close = series.resample("ME").last()
    month_signal = np.sign(month_close / month_close.shift(12) - 1.0).fillna(0.0)
    signal_by_period = pd.Series(month_signal.values, index=month_signal.index.to_period("M"))

    out = np.zeros(len(series), dtype=float)
    daily_periods = series.index.to_period("M")
    for i, period in enumerate(daily_periods):
        out[i] = float(signal_by_period.get(period, 0.0))
    return out


def current_daily_provider(rd):
    return strategy_sign_r(np.asarray(rd["rt"], dtype=float), lookback=252).astype(float)


def monthly_next_provider(rd):
    df = _full_contract_frame(rd["tk"], rd["source"])
    return build_monthly_next_positions(df["Close"].to_numpy(dtype=float), pd.to_datetime(df["Date"]))


def monthly_same_month_diag_provider(rd):
    df = _full_contract_frame(rd["tk"], rd["source"])
    return build_monthly_same_month_diag_positions(df["Close"].to_numpy(dtype=float), pd.to_datetime(df["Date"]))


def average_position_disagreement(raw_data, provider_a, provider_b) -> float:
    vals = []
    for rd in raw_data:
        a = np.asarray(provider_a(rd), dtype=float)
        b = np.asarray(provider_b(rd), dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.any():
            vals.append(float(np.mean(a[mask] != b[mask])))
    return float(np.mean(vals)) if vals else 0.0


def evaluate_variant(asset_name: str, label: str, provider):
    raw = load_contracts(
        asset_name,
        test_start=TEST_START,
        test_end=TEST_END,
        excluded_contracts=[],
        source_overrides=SOURCE_OVERRIDES,
    )
    metrics = compute_strategy_metrics(raw, "Sign(R)", SIGMA_TGT, position_provider=provider)
    paper = PAPER_TABLE3[asset_name]["Sign(R)"]
    return raw, metrics, paper, [
        asset_name,
        label,
        len(raw),
        fmt_num(metrics["E(R)"]),
        fmt_num(metrics["Sharpe"]),
        fmt_num(metrics["Sortino"]),
        fmt_pct(pct_err_raw(metrics["std(R)"], paper["std(R)"])),
        fmt_pct(pct_err_raw(metrics["DD"], paper["DD"])),
        fmt_pct(focus_err(metrics, paper)),
    ]


def build_report() -> str:
    rows = []
    disagreement_rows = []
    for asset in ASSETS:
        raw, _metrics, _paper, row = evaluate_variant(asset, "current_daily_252d", current_daily_provider)
        rows.append(row)
        _, _, _, row = evaluate_variant(asset, "monthly_12m_next_month_hold", monthly_next_provider)
        rows.append(row)
        _, _, _, row = evaluate_variant(asset, "monthly_12m_same_month_diag", monthly_same_month_diag_provider)
        rows.append(row)
        disagreement_rows.append([
            asset,
            fmt_pct(100.0 * average_position_disagreement(raw, current_daily_provider, monthly_next_provider)),
            fmt_pct(100.0 * average_position_disagreement(raw, current_daily_provider, monthly_same_month_diag_provider)),
        ])

    out = []
    out.append("# Sign(R) Monthly Quick Validation")
    out.append("")
    out.append("- Date: 2026-04-22")
    out.append("- Goal: test whether current daily `252d sign` is materially different from a monthly `12m signal + next-month hold` interpretation")
    out.append("- Evaluation stack: current unified baseline backtest only")
    out.append(f"- sigma_tgt = {SIGMA_TGT}")
    out.append("")
    out.append("Reference used:")
    out.append("- `DRL_37.pdf` (Moskowitz, Ooi, Pedersen 2012) describes a canonical `12-month lookback + 1-month holding` TSMOM strategy with inverse-volatility sizing.")
    out.append("- This is not the same object as a daily re-evaluated `sign(p_t - p_{t-252})` signal unless the two constructions happen to align in practice.")
    out.append("")
    out.append("## Variant Comparison")
    out.append("")
    out.append(md_table(
        ["Asset", "Variant", "#", "E(R)", "Sharpe", "Sortino", "std err", "DD err", "focus err"],
        rows,
    ))
    out.append("")
    out.append("## Position Disagreement")
    out.append("")
    out.append(md_table(
        ["Asset", "current vs monthly-next", "current vs same-month-diag"],
        disagreement_rows,
    ))
    out.append("")
    out.append("## Quick Reading")
    out.append("")
    out.append("- `monthly_12m_next_month_hold` is the most faithful monthly TSMOM analogue in this repo.")
    out.append("- `monthly_12m_same_month_diag` is intentionally look-ahead contaminated and is only a timing-sensitivity probe.")
    out.append("- If `current_daily_252d` and `monthly_12m_next_month_hold` differ meaningfully, then the daily Sign(R) implementation is not just a harmless restatement of the original monthly TSMOM logic.")
    out.append("")
    return "\n".join(out) + "\n"


def main():
    report = build_report()
    DOC_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved report: {DOC_PATH}")


if __name__ == "__main__":
    main()
