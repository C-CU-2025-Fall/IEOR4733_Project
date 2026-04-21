#!/usr/bin/env python3
"""Utilities for single-strategy baseline report output (4 assets + All)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from baseline_run import EWMA_SPAN, compute_portfolio_returns
from config import ASSET_CLASSES, METRIC_NAMES, PAPER_TABLE2, PAPER_TABLE3, SIGN_LOOKBACK
from data_loader import load_clc_full
from metrics import compute_metrics
from strategies import strategy_macd

ASSETS4 = ["Commodity", "Equity Index", "Fixed Income", "Forex"]


def _fmt(vals):
    return "  ".join(f"{v:>+7.3f}" for v in vals)


def load_contracts_with_overrides(
    ac_name: str,
    test_start: str = "2011-01-01",
    test_end: str = "2019-12-31",
    default_dataset: str = "RAD",
    source_overrides: dict[str, str] | None = None,
    excluded: set[str] | None = None,
) -> list[dict]:
    """Load contracts for one asset class with optional per-ticker dataset overrides."""
    tickers = ASSET_CLASSES.get(ac_name, [])
    source_overrides = source_overrides or {}
    excluded = excluded or set()

    raw = []
    for tk in tickers:
        if tk in excluded:
            continue

        dataset = source_overrides.get(tk, default_dataset).upper()
        df = load_clc_full(tk, data_dir="data/CLCDATA", dataset=dataset)
        if df is None:
            df = load_clc_full(tk, data_dir="data/CLC", dataset=dataset)
        if df is None:
            continue

        prices = df["Close"].values.astype(float)
        if len(prices) < 500:
            continue
        p0 = prices[0]
        norm_p = prices / p0

        rt = np.zeros(len(norm_p))
        rt[1:] = norm_p[1:] - norm_p[:-1]
        sigma = pd.Series(rt).ewm(span=EWMA_SPAN, adjust=False).std().values

        mask_s = df["Date"] >= test_start
        mask_e = df["Date"] <= test_end
        if not mask_s.any() or not mask_e.any():
            continue

        t0 = mask_s.idxmax()
        t1 = len(df) - 1 - mask_e[::-1].values.argmax()
        start = max(t0, SIGN_LOOKBACK)
        dates = df["Date"].iloc[start:t1].values

        raw.append(
            {
                "tk": tk,
                "rt": rt,
                "sigma": sigma,
                "norm_p": norm_p,
                "prices": prices,
                "start": start,
                "t1": t1,
                "dates": dates,
                "macd_pos": strategy_macd(norm_p),
            }
        )
    return raw


def _paper_values(asset: str, strategy: str) -> dict | None:
    """Return paper target metrics for asset/strategy.

    Table 3 has 4 assets, no explicit All. For All we fallback to Table 2 All target.
    """
    if asset in PAPER_TABLE3 and strategy in PAPER_TABLE3[asset]:
        return PAPER_TABLE3[asset][strategy]
    if asset == "All" and "All" in PAPER_TABLE2 and strategy in PAPER_TABLE2["All"]:
        return PAPER_TABLE2["All"][strategy]
    return None


def run_single_strategy_report(
    strategy: str,
    sigma_tgt: float = 0.064,
    test_start: str = "2011-01-01",
    test_end: str = "2019-12-31",
    default_dataset: str = "RAD",
    source_overrides: dict[str, str] | None = None,
    excluded: set[str] | None = None,
    include_all: bool = True,
    table_label: str = "Table 3",
) -> tuple[int, int, int]:
    """Print per-asset + All report for one strategy only."""
    source_overrides = source_overrides or {}
    excluded = excluded or set()

    assets = list(ASSETS4)
    if include_all:
        assets.append("All")

    grand_n10, grand_n15, grand_total = 0, 0, 0

    for asset in assets:
        if asset == "All":
            raw = []
            for ac in ASSETS4:
                raw.extend(
                    load_contracts_with_overrides(
                        ac_name=ac,
                        test_start=test_start,
                        test_end=test_end,
                        default_dataset=default_dataset,
                        source_overrides=source_overrides,
                        excluded=excluded,
                    )
                )
        else:
            raw = load_contracts_with_overrides(
                ac_name=asset,
                test_start=test_start,
                test_end=test_end,
                default_dataset=default_dataset,
                source_overrides=source_overrides,
                excluded=excluded,
            )

        n_contracts = len(raw)
        if n_contracts == 0:
            continue

        print(f"\n{'=' * 110}")
        print(f"  {table_label} — {asset} ({n_contracts} contracts)")
        print(f"  σ_tgt={sigma_tgt} | EWMA({EWMA_SPAN}) | bp=0.002")
        if asset == "All":
            print("  note: All paper target uses PAPER_TABLE2['All'] as fallback")
        print(f"{'=' * 110}")

        R = compute_portfolio_returns(raw, strategy, sigma_tgt)
        m = compute_metrics(R, n_contracts)

        paper = _paper_values(asset, strategy)
        print(f"\n  {strategy:8s}")
        print(f"  Ours  : {_fmt(m)}")

        if paper is None:
            print("  Paper : N/A")
            print("  %Err  : N/A")
            continue

        pv = [paper[k] for k in METRIC_NAMES]
        errs = [abs((m[i] - pv[i]) / abs(pv[i])) * 100 if pv[i] != 0 else 0 for i in range(9)]
        n10 = sum(1 for e in errs if e < 10)
        n15 = sum(1 for e in errs if e < 15)

        print(f"  Paper : {_fmt(pv)}")
        print(f"  %Err  : {'  '.join(f'{e:>6.1f}%' for e in errs)}")
        print(f"  Score : (≤10%:{n10}/9  ≤15%:{n15}/9)")

        grand_n10 += n10
        grand_n15 += n15
        grand_total += 9

    if grand_total > 0:
        print(f"\n{'=' * 60}")
        print(f"  GRAND TOTAL: ≤10%: {grand_n10}/{grand_total} | ≤15%: {grand_n15}/{grand_total}")
        print(f"{'=' * 60}")

    return grand_n10, grand_n15, grand_total
