#!/usr/bin/env python3
"""
Compute and save feature correlation matrices for all 48 active structural-38
contracts.

Outputs:
  results/feature_correlation_48/
    {ticker}_corr.csv       — 12x12 correlation matrix
    {ticker}_stats.csv      — per-feature mean/std/min/max/NaN count
    aggregate_corr.csv      — mean correlation across all contracts
    summary.json            — version, timestamp, aggregate stats
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.chdir(str(REPO))

from config import ASSET_CLASSES
from data_loader import load_clc_full
from frontier_presets import STRUCTURAL_38_EXCLUDED, STRUCTURAL_38_OVERRIDES

# =============================================================================
# Proposed 12D feature labels
# =============================================================================

FEATURE_LABELS = [
    "ret_1d",       # 0: r_t / sigma_t
    "ret_5d",       # 1: (p_t-p_{t-5}) / (sigma * sqrt(5))
    "ret_21d",      # 2: (p_t-p_{t-21}) / (sigma * sqrt(21))
    "ret_126d",     # 3: (p_t-p_{t-126}) / (sigma * sqrt(126))
    "macd_8_24",    # 4: MACD(8,24) normalized
    "rsi_5",        # 5: (RSI_5 - 50) / 50
    "rsi_30",       # 6: (RSI_30 - 50) / 50
    "atr_norm",     # 7: ATR(20) / ATR_MA(20) — OHLC range regime
    "vol_norm",     # 8: Volume / Vol_MA(20)
    "oi_chg",       # 9: ΔOI / |OI_{t-1}|, clipped [-5,5]
    "drawdown",     # 10: (p_t - max_126d) / max_126d
    "days_to_roll", # 11: (days_to_next_roll / 90) - 0.5
]


# =============================================================================
# Feature computation (mirrors proposed build_feature_matrix_v3)
# =============================================================================

def compute_ewma_sigma(returns):
    return pd.Series(returns).ewm(span=60, adjust=False).std().to_numpy(dtype=float)


def compute_rsi(prices, window):
    delta = np.diff(prices, prepend=prices[0])
    alpha = 1.0 / window
    gain = pd.Series(np.where(delta > 0, delta, 0.0)).ewm(alpha=alpha, adjust=False).mean().values
    loss = pd.Series(np.where(delta < 0, -delta, 0.0)).ewm(alpha=alpha, adjust=False).mean().values + 1e-10
    rsi = 100.0 - 100.0 / (1.0 + gain / loss)
    return (rsi - 50.0) / 50.0


def ret_norm(prices, sigma, horizon):
    f = np.zeros(len(prices))
    for i in range(horizon, len(prices)):
        f[i] = (prices[i] - prices[i - horizon]) / (sigma[i] * np.sqrt(horizon) + 1e-10)
    return f


def parse_asc_roll_dates(ticker):
    """Parse roll dates from {TICKER}_CLC.ASC file."""
    asc_path = REPO / f"config/TEMP/{ticker}_CLC.ASC"
    if not asc_path.exists():
        return None
    roll_dates = []
    prev_date = None
    with open(asc_path) as f:
        for line in f:
            cols = line.strip().split()
            if cols[0] == "00000000":
                if prev_date is not None:
                    roll_dates.append(prev_date)
            else:
                prev_date = cols[0]
    return pd.to_datetime(roll_dates, format="%Y%m%d") if roll_dates else None


def compute_features(prices, high, low, volume, oi, roll_dates_df, dates):
    """Compute the 12D feature matrix. Returns (N, 12) float32 array."""
    n = len(prices)
    returns = np.zeros(n, dtype=float)
    returns[1:] = prices[1:] - prices[:-1]
    sigma = compute_ewma_sigma(returns)

    feats = np.zeros((n, 12), dtype=np.float32)

    # 0: ret_1d_vol_norm
    feats[:, 0] = returns / (sigma + 1e-10)

    # 1: ret_5d
    feats[:, 1] = ret_norm(prices, sigma, 5)

    # 2: ret_21d
    feats[:, 2] = ret_norm(prices, sigma, 21)

    # 3: ret_126d
    feats[:, 3] = ret_norm(prices, sigma, 126)

    # 4: MACD(8,24)
    macd_vol = pd.Series(prices).rolling(63, min_periods=5).std().to_numpy(dtype=float)
    ema8 = pd.Series(prices).ewm(span=8, adjust=False).mean().to_numpy(dtype=float)
    ema24 = pd.Series(prices).ewm(span=24, adjust=False).mean().to_numpy(dtype=float)
    q_t = (ema8 - ema24) / (macd_vol + 1e-10)
    q_std = pd.Series(q_t).rolling(252, min_periods=21).std().to_numpy(dtype=float)
    feats[:, 4] = q_t / (q_std + 1e-10)

    # 5: RSI_5
    feats[:, 5] = compute_rsi(prices, 5)

    # 6: RSI_30
    feats[:, 6] = compute_rsi(prices, 30)

    # 7: ATR norm (True Range / ATR_MA(20))
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - prices[i - 1]), abs(low[i] - prices[i - 1]))
    atr_ma = pd.Series(tr).rolling(20).mean().to_numpy(dtype=float)
    feats[:, 7] = tr / (atr_ma + 1e-10)

    # 8: Volume norm (Volume / Vol_MA(20))
    vol_ma = pd.Series(volume).rolling(20).mean().to_numpy(dtype=float)
    feats[:, 8] = volume / (vol_ma + 1e-10)

    # 9: OI change (ΔOI / |OI_{t-1}|)
    oi_chg = np.zeros(n)
    oi_chg[1:] = (oi[1:] - oi[:-1]) / (np.abs(oi[:-1]) + 1e-10)
    feats[:, 9] = np.clip(oi_chg, -5, 5)

    # 10: Drawdown from 126d high
    rolling_max = pd.Series(prices).rolling(126, min_periods=20).max().to_numpy(dtype=float)
    feats[:, 10] = (prices - rolling_max) / (rolling_max + 1e-10)

    # 11: Days to next roll
    if roll_dates_df is not None and len(roll_dates_df) > 0:
        days_to_roll = np.zeros(n)
        for i in range(n):
            dt = dates[i]
            future_rolls = roll_dates_df[roll_dates_df >= dt]
            if len(future_rolls) > 0:
                days_to_roll[i] = (future_rolls[0] - dt) / np.timedelta64(1, "D")
            else:
                days_to_roll[i] = 90
        feats[:, 11] = days_to_roll / 90.0 - 0.5  # normalize to ~[-0.5, 0.5]
    else:
        feats[:, 11] = 0.0

    # Clean NaNs and Infs
    feats = np.nan_to_num(feats, nan=0.0, posinf=3.0, neginf=-3.0)
    return feats


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Feature correlation analysis for all 48 active contracts")
    parser.add_argument("--burnin", type=int, default=252, help="Burn-in days before correlation computation")
    parser.add_argument("--output", type=str, default="results/feature_correlation_48")
    args = parser.parse_args()

    burnin = args.burnin
    out_dir = REPO / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get all active tickers
    excluded = set(STRUCTURAL_38_EXCLUDED)
    all_tickers = []
    for ac, tickers in ASSET_CLASSES.items():
        for t in tickers:
            if t not in excluded:
                all_tickers.append(t)

    print(f"Computing 12D feature correlations for {len(all_tickers)} contracts...")
    print(f"Burn-in: {burnin} days")
    print(f"Output: {out_dir}")
    print()

    per_ticker_corr = {}   # ticker -> 12x12 ndarray
    per_ticker_stats = {}  # ticker -> dict of stats
    problems = []

    for ticker in sorted(all_tickers):
        source = STRUCTURAL_38_OVERRIDES.get(ticker, "RAD")
        try:
            df = load_clc_full(ticker, start_date="2003-01-01", source=source)
            if df is None or len(df) <= burnin + 100:
                problems.append(f"{ticker}: insufficient data ({len(df) if df is not None else 0} rows)")
                continue

            prices = df["Close"].values
            high = df["High"].values
            low = df["Low"].values
            volume = df["Volume"].values.astype(float)
            oi = df["OI"].values.astype(float)
            dates = df["Date"].values

            roll_dates_df = parse_asc_roll_dates(ticker)

            feats = compute_features(prices, high, low, volume, oi, roll_dates_df, dates)
            feats_burned = feats[burnin:]

            # Guard against zero-variance columns
            stds = np.std(feats_burned, axis=0)
            zero_var_cols = np.where(stds < 1e-10)[0]
            if len(zero_var_cols) > 0:
                print(f"  {ticker:6s}: zero-variance columns: {[FEATURE_LABELS[i] for i in zero_var_cols]}")
                for ci in zero_var_cols:
                    feats_burned[:, ci] += np.random.randn(len(feats_burned)) * 1e-8

            # Build correlation matrix
            corr = np.corrcoef(feats_burned.T)
            per_ticker_corr[ticker] = corr

            # Per-feature statistics
            stats = {}
            for i, label in enumerate(FEATURE_LABELS):
                col = feats_burned[:, i]
                stats[label] = {
                    "mean": float(np.mean(col)),
                    "std": float(np.std(col)),
                    "min": float(np.min(col)),
                    "max": float(np.max(col)),
                    "nan_count": int(np.isnan(col).sum()),
                    "inf_count": int(np.isinf(col).sum()),
                    "zero_frac": float(np.mean(np.abs(col) < 1e-8)),
                }
            per_ticker_stats[ticker] = stats

            # Save individual CSV
            corr_df = pd.DataFrame(corr, index=FEATURE_LABELS, columns=FEATURE_LABELS)
            corr_df.to_csv(out_dir / f"{ticker}_corr.csv", float_format="%.4f")

            stats_df = pd.DataFrame(stats).T
            stats_df.to_csv(out_dir / f"{ticker}_stats.csv", float_format="%.4f")

            # Log max off-diagonal
            np.fill_diagonal(corr, 0)
            max_corr = np.max(np.abs(corr))
            print(f"  {ticker:6s} (source={source:>10s}, n={len(df):5d}): max_off_diag={max_corr:.3f}")

        except Exception as e:
            problems.append(f"{ticker}: {e}")
            print(f"  {ticker:6s}: ERROR — {e}")

    # Aggregate correlation matrix (mean across all contracts)
    if per_ticker_corr:
        corr_stack = np.stack(list(per_ticker_corr.values()), axis=0)  # (N_tickers, 12, 12)
        mean_corr = np.mean(corr_stack, axis=0)
        std_corr = np.std(corr_stack, axis=0)

        mean_df = pd.DataFrame(mean_corr, index=FEATURE_LABELS, columns=FEATURE_LABELS)
        std_df = pd.DataFrame(std_corr, index=FEATURE_LABELS, columns=FEATURE_LABELS)
        mean_df.to_csv(out_dir / "aggregate_corr_mean.csv", float_format="%.4f")
        std_df.to_csv(out_dir / "aggregate_corr_std.csv", float_format="%.4f")

        # Top off-diagonal correlations in aggregate
        print(f"\n{'=' * 70}")
        print(f"Aggregate (mean) correlation matrix — top off-diagonals:")
        print(f"{'=' * 70}")
        np.fill_diagonal(mean_corr, 0)
        flat_idx = np.argsort(np.abs(mean_corr).flatten())[::-1]
        for idx in flat_idx[:15]:
            i, j = divmod(idx, 12)
            if i < j:
                sign = "(+)" if mean_corr[i, j] > 0 else "(-)"
                print(f"  {sign} {FEATURE_LABELS[i]:14s} ↔ {FEATURE_LABELS[j]:14s}: {mean_corr[i, j]:.3f}")

        # Summary JSON
        summary = {
            "version": "structural_38_enhanced_12d",
            "timestamp": datetime.now().isoformat(),
            "burnin_days": burnin,
            "contracts_processed": len(per_ticker_corr),
            "contracts_failed": len(problems),
            "feature_labels": FEATURE_LABELS,
            "per_ticker_max_off_diagonal": {},
            "aggregate_max_off_diagonal_nonself": {k: {} for k in FEATURE_LABELS},
        }
        for i in range(12):
            for j in range(12):
                if i != j:
                    summary["aggregate_max_off_diagonal_nonself"][FEATURE_LABELS[i]][FEATURE_LABELS[j]] = round(float(mean_corr[i, j]), 4)

        for ticker, corr in per_ticker_corr.items():
            c = corr.copy()
            np.fill_diagonal(c, 0)
            summary["per_ticker_max_off_diagonal"][ticker] = round(float(np.max(np.abs(c))), 4)

        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    # Problem report
    if problems:
        print(f"\n=== PROBLEMS ({len(problems)}) ===")
        for p in problems:
            print(f"  {p}")

    print(f"\nResults saved to {out_dir}/")
    print(f"  {len(per_ticker_corr)} contracts processed")
    print(f"  {len(problems)} problems")


if __name__ == "__main__":
    main()
