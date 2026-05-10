#!/usr/bin/env python3
"""DQN vs Long rolling Sharpe analysis.

Outputs per-asset CSV files with daily returns, 252-day rolling Sharpe,
and cumulative wealth curves for both DQN and Long-only strategies.

Usage:
    python scripts/dqn_rolling_sharpe.py --round 1
    python scripts/dqn_rolling_sharpe.py --round 1 --asset Forex
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_run import load_contracts
from drl.dqn.spec import current_source_policy
from drl.dqn.backtest.engine import dqn_position_provider

SIGMA_TGT = 0.058
ASSET_CLASSES_4 = ["Forex", "Equity Index", "Commodity", "Fixed Income"]


def _policy_overrides():
    p = current_source_policy()
    return p["source_overrides"], p["excluded_contracts"]


def compute_rolling_sharpe(returns: np.ndarray, window: int = 252) -> np.ndarray:
    """252-day rolling annualized Sharpe ratio."""
    s = pd.Series(returns)
    roll_mean = s.rolling(window).mean() * 252
    roll_std = s.rolling(window).std() * np.sqrt(252)
    return (roll_mean / roll_std).values


def portfolio_daily_returns(
    raw_data: list,
    position_provider_fn,
    sigma_tgt: float = SIGMA_TGT,
) -> pd.Series:
    """Equal-weight portfolio daily returns from a position provider."""
    all_rets: dict[str, pd.Series] = {}
    for rd in raw_data:
        tk = rd["tk"]
        returns = np.asarray(rd["rt"], dtype=float)
        sigma = np.asarray(rd["sigma"], dtype=float)
        start, t1 = int(rd["start"]), int(rd["t1"])
        dates = pd.to_datetime(rd["dates"])
        eval_len = min(len(dates), max(0, t1 - start + 1))

        pos = position_provider_fn(rd)
        eval_pos = pos[start:start + eval_len]
        scaled = eval_pos * (sigma_tgt / (np.abs(sigma[start:start + eval_len]) + 1e-10)) * returns[start:start + eval_len]
        all_rets[tk] = pd.Series(scaled, index=dates[:eval_len])

    return pd.DataFrame(all_rets).mean(axis=1)


def make_long_fn(sigma_tgt: float = SIGMA_TGT):
    """Create a Long-only position provider."""
    def long_fn(rd):
        prices = np.asarray(rd["prices"], dtype=float)
        sigma = np.asarray(rd["sigma"], dtype=float)
        start, t1 = int(rd["start"]), int(rd["t1"])
        dates = pd.to_datetime(rd["dates"])
        eval_len = min(len(dates), max(0, t1 - start + 1))
        pos = np.zeros(len(prices))
        pos[start:start + eval_len] = sigma_tgt / (np.abs(sigma[start:start + eval_len]) + 1e-10)
        return pos
    return long_fn


def run_analysis(
    asset_name: str,
    round_num: int,
    device: str = "cuda",
    sigma_tgt: float = SIGMA_TGT,
    test_start: str = "2011-01-01",
    test_end: str = "2019-12-31",
    output_dir: str | None = None,
) -> pd.DataFrame:
    """Run rolling Sharpe analysis for one asset class and return merged DataFrame."""
    overrides, excluded = _policy_overrides()
    raw = load_contracts(asset_name, test_start=test_start, test_end=test_end,
                         excluded_contracts=excluded, source_overrides=overrides)
    provider = dqn_position_provider(
        asset_name=asset_name, round_num=round_num,
        device=device, expected_sigma_tgt=sigma_tgt,
    )

    port_dqn = portfolio_daily_returns(raw, provider, sigma_tgt)
    port_long = portfolio_daily_returns(raw, make_long_fn(sigma_tgt), sigma_tgt)

    rs_dqn = compute_rolling_sharpe(port_dqn.values)
    rs_long = compute_rolling_sharpe(port_long.values)

    df = pd.DataFrame({
        "date": port_dqn.index,
        "ret_dqn": port_dqn.values,
        "ret_long": port_long.reindex(port_dqn.index).values,
        "rolling_sharpe_dqn_252": rs_dqn,
        "rolling_sharpe_long_252": rs_long,
        "wealth_dqn": np.cumprod(1 + port_dqn.values),
        "wealth_long": np.cumprod(1 + port_long.reindex(port_dqn.index).values),
        "alpha_daily": port_dqn.values - port_long.reindex(port_dqn.index).values,
    })

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        slug = asset_name.replace(" ", "_")
        path = out / f"dqn_r{round_num}_rolling_{slug}.csv"
        df.to_csv(path, index=False)
        print(f"  [saved] {path} ({len(df)} days)")

    return df


def print_summary(results: dict[str, pd.DataFrame]):
    """Print summary table."""
    print(f"\n{'='*70}")
    print("  DQN vs Long: 252-day Rolling Sharpe Summary")
    print(f"{'='*70}")
    print(f"  {'Asset':>15s} {'DQN μ':>8s} {'Long μ':>8s} {'DQN>Long':>10s}")
    print(f"  {'-'*45}")
    for ac, df in results.items():
        rs_d = df["rolling_sharpe_dqn_252"].values
        rs_l = df["rolling_sharpe_long_252"].values
        valid = (~np.isnan(rs_d)) & (~np.isnan(rs_l))
        if valid.sum() == 0:
            continue
        n_better = np.sum(rs_d[valid] > rs_l[valid])
        total = valid.sum()
        print(f"  {ac:>15s} {np.nanmean(rs_d[valid]):>+8.3f} {np.nanmean(rs_l[valid]):>+8.3f} {n_better}/{total}d ({n_better/total*100:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="DQN rolling Sharpe analysis")
    parser.add_argument("--asset", default=None, help="Single asset class (default: all 4)")
    parser.add_argument("--round", type=int, required=True, help="Round number (1 or 2)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sigma", type=float, default=SIGMA_TGT)
    parser.add_argument("--output-dir", default="archive", help="Output CSV directory")
    parser.add_argument("--test-start", default="2011-01-01")
    parser.add_argument("--test-end", default="2019-12-31")
    args = parser.parse_args()

    assets = [args.asset] if args.asset else ASSET_CLASSES_4
    results = {}

    for ac in assets:
        print(f"Computing {ac}...")
        df = run_analysis(
            asset_name=ac,
            round_num=args.round,
            device=args.device,
            sigma_tgt=args.sigma,
            test_start=args.test_start,
            test_end=args.test_end,
            output_dir=args.output_dir,
        )
        results[ac] = df

    print_summary(results)


if __name__ == "__main__":
    main()
