#!/usr/bin/env python3
"""One-command baseline reproduction for the retained structural-38 configuration."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_run import load_contracts, run_table  # noqa: E402
from config import PAPER_TABLE2, PAPER_TABLE3  # noqa: E402
from data_loader import load_clc_full  # noqa: E402
from frontier_presets import STRUCTURAL_38_EXCLUDED, STRUCTURAL_38_OVERRIDES  # noqa: E402
from strategies import strategy_sign_r  # noqa: E402


OVERRIDES = STRUCTURAL_38_OVERRIDES
EXCLUDED = sorted(STRUCTURAL_38_EXCLUDED)
SIGMA = 0.06
TRADE_METRICS = ['E(R)', 'std(R)', 'DD', 'Sharpe', 'Sortino', '% +ve', 'Ave P/L']
PATH_METRICS = ['MDD', 'Calmar']
VALID_STRATEGIES = ("Long", "Sign(R)", "MACD")


def _full_contract_frame(ticker: str, source: str, test_start: str = "2011-01-01"):
    df = load_clc_full(ticker, source=source, anchor_date=test_start)
    if df is None:
        raise ValueError(f"Missing full frame for {ticker} ({source})")
    return df.reset_index(drop=True)


def build_252d_21d_hold_positions(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    out = np.zeros(len(prices), dtype=float)
    for t in range(252, len(prices), 21):
        sig = np.sign(prices[t] - prices[t - 252])
        out[t:min(t + 21, len(prices))] = sig
    return out


def build_monthly_next_positions(prices: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    series = pd.Series(np.asarray(prices, dtype=float), index=pd.DatetimeIndex(dates))
    month_close = series.resample("ME").last()
    month_signal = np.sign(month_close / month_close.shift(12) - 1.0).fillna(0.0)
    signal_by_period = pd.Series(month_signal.values, index=month_signal.index.to_period("M"))

    out = np.zeros(len(series), dtype=float)
    daily_periods = series.index.to_period("M")
    for i, period in enumerate(daily_periods):
        out[i] = float(signal_by_period.get(period - 1, 0.0))
    return out


def signr_position_provider(mode: str, test_start: str = "2011-01-01"):
    if mode == "current":
        return lambda rd: strategy_sign_r(np.asarray(rd["rt"], dtype=float), lookback=252).astype(float)

    if mode == "hold21":
        def _provider(rd):
            df = _full_contract_frame(rd["tk"], rd["source"], test_start=test_start)
            return build_252d_21d_hold_positions(df["Close"].to_numpy(dtype=float))

        return _provider

    if mode == "month":
        def _provider(rd):
            df = _full_contract_frame(rd["tk"], rd["source"], test_start=test_start)
            return build_monthly_next_positions(
                df["Close"].to_numpy(dtype=float),
                pd.to_datetime(df["Date"]),
            )

        return _provider

    raise ValueError(f"Unsupported Sign(R) mode: {mode}")


def parse_strategies(spec: str) -> list[str]:
    tokens = [s.strip() for s in spec.split(",") if s.strip()]
    if not tokens:
        raise ValueError("At least one strategy must be provided.")
    invalid = [s for s in tokens if s not in VALID_STRATEGIES]
    if invalid:
        raise ValueError(f"Unsupported strategies: {', '.join(invalid)}")
    return tokens


def run_baseline_tables(
    table: str,
    with_path_metrics: bool = False,
    signr_mode: str = "none",
    strategies: list[str] | None = None,
):
    asset_classes = ["Commodity", "Equity Index", "Fixed Income", "Forex"]
    tables = []
    if table in {"3", "both"}:
        tables.append(("Table 3", PAPER_TABLE3, None))
    if table in {"2", "both"}:
        tables.append(("Table 2", PAPER_TABLE2, 0.97))
    metric_names = list(TRADE_METRICS)
    if with_path_metrics:
        metric_names = metric_names[:5] + PATH_METRICS + metric_names[5:]
    chosen = VALID_STRATEGIES if strategies is None else tuple(strategies)
    strategy_entries = []
    for strat in chosen:
        if strat == "Sign(R)" and signr_mode != "none":
            strategy_entries.append((strat, signr_position_provider(signr_mode)))
        else:
            strategy_entries.append((strat, None))

    for table_label, paper_table, port_vol in tables:
        total10 = total15 = totaln = 0
        for ac in asset_classes:
            raw = load_contracts(ac, excluded_contracts=EXCLUDED, source_overrides=OVERRIDES)
            n10, n15, n = run_table(
                raw,
                ac,
                SIGMA,
                paper_table,
                table_label,
                port_vol_target=port_vol,
                metric_names=metric_names,
                port_bridge="rolling252_lagged",
                strategy_entries=strategy_entries,
            )
            total10 += n10
            total15 += n15
            totaln += n
        print("\n" + "=" * 60)
        print(f"{table_label} TOTAL: <=10%: {total10}/{totaln} | <=15%: {total15}/{totaln}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", choices=["2", "3", "both"], default="both")
    parser.add_argument(
        "--with-path-metrics",
        action="store_true",
        help="Also show portfolio-path MDD and Calmar from the current unified backtest stack.",
    )
    parser.add_argument(
        "--signr-mode",
        choices=["none", "current", "hold21", "month"],
        default="none",
        help="Optional Sign(R) variant to show alongside Long. "
             "`hold21` = 252d signal + 21d hold, `month` = natural-month next-month hold.",
    )
    parser.add_argument(
        "--strategies",
        default="Long",
        help="Comma-separated strategies to display, e.g. `Long`, `Sign(R)`, `MACD`, "
             "or `Long,Sign(R),MACD`.",
    )
    args = parser.parse_args()
    strategies = parse_strategies(args.strategies)
    print("Structural-38 baseline")
    print("=" * 60)
    print(f"excluded: {', '.join(EXCLUDED)}")
    print("source overrides:")
    for tk, src in sorted(OVERRIDES.items()):
        print(f"  {tk}: {src}")
    print(f"strategies: {', '.join(strategies)}")
    if args.signr_mode != "none":
        print(f"Sign(R) mode: {args.signr_mode}")
    print()
    run_baseline_tables(
        args.table,
        with_path_metrics=args.with_path_metrics,
        signr_mode=args.signr_mode,
        strategies=strategies,
    )


if __name__ == "__main__":
    main()
