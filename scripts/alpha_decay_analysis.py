#!/usr/bin/env python3
"""
Alpha Decay Analysis for DQN Strategy.

Uses baseline_run's reward computation (the SAME path as run_strategy_backtest.py).
No reward discrepancy — one single source of truth.

Usage:
  python scripts/alpha_decay_analysis.py --asset Forex --round 1
  python scripts/alpha_decay_analysis.py --asset Forex --round 2
  python scripts/alpha_decay_analysis.py --asset Forex  # all rounds stitched
"""
import sys, os, argparse, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

from baseline_run import (
    load_contracts,
    compute_contract_returns_from_positions,
    DEFAULT_SIGMA_TGT,
)
from drl.dqn.backtest.engine import dqn_position_provider
from drl_shared.spec import RETRAIN_ROUNDS


# ─── helpers ──────────────────────────────────────────────────────

def sharpe(R):
    """Annualized Sharpe from daily returns."""
    if len(R) < 20:
        return np.nan
    mu = np.mean(R) * 252
    sig = np.std(R) * np.sqrt(252)
    return mu / sig if sig > 1e-10 else 0.0


def portfolio_daily_returns(asset, round_num, checkpoint_bundle, sigma_tgt):
    """Equal-weight portfolio daily returns for the test period.

    Uses compute_contract_returns_from_positions (baseline_run vectorized Eq.4)
    — identical to run_strategy_backtest.py.

    Returns pd.Series indexed by sorted dates.
    """
    raw = load_contracts(asset)
    provider = dqn_position_provider(
        asset_name=asset,
        round_num=round_num,
        checkpoint_bundle=checkpoint_bundle,
        device="cpu",
    )
    series = []
    for rd in raw:
        pos = np.asarray(provider(rd), dtype=float)
        Rt = compute_contract_returns_from_positions(rd, pos, sigma_tgt)
        start, t1, dates = rd["start"], rd["t1"], rd["dates"]
        slc = Rt[start : t1 + 1]
        min_len = min(len(slc), len(dates))
        series.append(pd.Series(slc[:min_len], index=dates[:min_len]))

    df = pd.DataFrame(series)
    return df.T.mean(axis=1).sort_index()


def resolve_checkpoint(asset, round_num, checkpoint_override=None):
    import glob
    if checkpoint_override:
        return checkpoint_override
    pattern = f"drl/dqn/models/{asset}/r{round_num}/*/checkpoint.pt"
    ckpts = sorted(glob.glob(pattern))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint: {pattern}")
    return os.path.dirname(ckpts[-1])


# ─── analysis ─────────────────────────────────────────────────────

def run_analysis(port, label=""):
    dates = port.index
    R = port.values
    n = len(R)

    print(f"\n{'=' * 78}")
    print(f"  Alpha Decay Analysis — {label}")
    print(f"  Period: {dates[0].date()} → {dates[-1].date()}  ({n} trading days)")
    print(f"  Reward: baseline_run vectorized Eq.4 (same as run_strategy_backtest)")
    print(f"{'=' * 78}")

    # ── [1] Non-overlapping Sharpe by window ──
    print(f"\n  [1] Non-overlapping Sharpe by Window")
    print(f"  {'Window':>6} {'#':>4} {'Mean':>8} {'Median':>8} {'Std':>7} {'Min':>8} {'Max':>8} {'%>0':>5}")
    print(f"  {'-' * 62}")
    for lbl, wd in [("1W", 5), ("1M", 21), ("6M", 126), ("1Y", 252)]:
        if wd > n:
            continue
        ss = np.array([sharpe(R[s : s + wd]) for s in range(0, n - wd + 1, wd)])
        ss = ss[~np.isnan(ss)]
        if len(ss) == 0:
            continue
        print(
            f"  {lbl:>6} {len(ss):>4} {np.mean(ss):>+8.3f} {np.median(ss):>+8.3f} "
            f"{np.std(ss):>7.3f} {np.min(ss):>+8.3f} {np.max(ss):>+8.3f} {(ss > 0).mean() * 100:>4.0f}%"
        )
    # full period
    full_s = sharpe(R)
    print(
        f"  {'Full':>6} {1:>4} {full_s:>+8.3f} {full_s:>+8.3f} "
        f"{'—':>7} {full_s:>+8.3f} {full_s:>+8.3f} {int(full_s > 0) * 100:>4}%"
    )

    # ── [2] Rolling 1Y Sharpe ──
    rw = 252
    rs_list, rd_list = [], []
    for s in range(0, n - rw + 1, 21):
        rs_list.append(sharpe(R[s : s + rw]))
        rd_list.append(dates[s + rw // 2])
    rs = np.array(rs_list)

    print(f"\n  [2] Rolling 1Y Sharpe (21d step)")
    print(f"  Mean={np.nanmean(rs):+.3f}  Range=[{np.nanmin(rs):+.3f}, {np.nanmax(rs):+.3f}]  %>0={np.nanmean(rs > 0) * 100:.0f}%")

    # ── [3] Calendar Year ──
    print(f"\n  [3] Calendar Year")
    print(f"  {'Year':>6} {'Sharpe':>8} {'CumR':>8} {'Days':>6} {'%>0':>6}")
    for y in sorted(dates.year.unique()):
        m = dates.year == y
        ry = R[m]
        if len(ry) < 20:
            continue
        cr = np.cumsum(ry)[-1]
        print(f"  {y:>6} {sharpe(ry):>+8.3f} {cr:>+8.4f} {len(ry):>6} {(ry > 0).mean() * 100:>5.1f}%")

    # ── [4] Decay Tests ──
    print(f"\n  [4] Decay Tests")
    # linear trend
    x = np.arange(len(rs))
    y = rs
    x_m, y_m = x.mean(), np.nanmean(y)
    slope = np.nanmean((x - x_m) * (y - y_m)) / (np.mean((x - x_m) ** 2) + 1e-10)
    ss_res = np.nansum((y - (slope * x + (y_m - slope * x_m))) ** 2)
    ss_tot = np.nansum((y - y_m) ** 2)
    r_sq = 1 - ss_res / (ss_tot + 1e-10)
    print(f"  Rolling Sharpe trend: slope={slope:+.5f}/step  R²={r_sq:.3f}")

    # first half vs second half
    mid = n // 2
    s1, s2 = sharpe(R[:mid]), sharpe(R[mid:])
    delta = s2 - s1
    tag = "DECAY" if delta < -0.1 else "IMPROVEMENT" if delta > 0.1 else "FLAT"
    print(f"  1st half ({dates[0].date()}→{dates[mid-1].date()}): Sharpe={s1:+.3f}")
    print(f"  2nd half ({dates[mid].date()}→{dates[-1].date()}): Sharpe={s2:+.3f}")
    print(f"  Δ = {delta:+.3f}  ({tag})")

    # year-over-year trend
    yr_sharpes = []
    for y in sorted(dates.year.unique()):
        m = dates.year == y
        ry = R[m]
        if len(ry) > 20:
            yr_sharpes.append(sharpe(ry))
    if len(yr_sharpes) >= 3:
        yr_x = np.arange(len(yr_sharpes))
        yr_y = np.array(yr_sharpes)
        yr_xm, yr_ym = yr_x.mean(), yr_y.mean()
        slope_yr = np.mean((yr_x - yr_xm) * (yr_y - yr_ym)) / (np.mean((yr_x - yr_xm) ** 2) + 1e-10)
        print(f"  Year-over-year trend: slope={slope_yr:+.3f}/yr")

    # ── [5] Cumulative Return Milestones ──
    print(f"\n  [5] Cumulative Return Milestones")
    cum = np.cumsum(R)
    for frac in [0.25, 0.5, 0.75, 1.0]:
        idx = min(int(n * frac), n - 1)
        print(f"    {frac:.0%} ({dates[idx].date()}): cumR={cum[idx]:+.4f}")


# ─── main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Alpha decay analysis for DQN strategy")
    parser.add_argument("--asset", required=True)
    parser.add_argument("--round", type=int, default=None, help="Round number. Omit to analyze all rounds.")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    sigma_tgt = DEFAULT_SIGMA_TGT

    if args.round is not None:
        cfg = RETRAIN_ROUNDS[args.round]
        ckpt = resolve_checkpoint(args.asset, args.round, args.checkpoint)
        print(f"Model: {ckpt}")
        print(f"Round {args.round}: test={cfg['test_start']}→{cfg['test_end']}")
        port = portfolio_daily_returns(args.asset, args.round, ckpt, sigma_tgt)
        # Filter to this round's test period
        port = port[(port.index >= cfg["test_start"]) & (port.index <= cfg["test_end"])]
        run_analysis(port, label=f"DQN {args.asset} R{args.round}")
    else:
        # Analyze each round separately
        for rnd in sorted(RETRAIN_ROUNDS):
            cfg = RETRAIN_ROUNDS[rnd]
            try:
                ckpt = resolve_checkpoint(args.asset, rnd, args.checkpoint)
            except FileNotFoundError:
                print(f"R{rnd}: no checkpoint, skipping")
                continue
            print(f"\nModel: {ckpt}")
            port = portfolio_daily_returns(args.asset, rnd, ckpt, sigma_tgt)
            port = port[(port.index >= cfg["test_start"]) & (port.index <= cfg["test_end"])]
            run_analysis(port, label=f"DQN {args.asset} R{rnd}")


if __name__ == "__main__":
    main()
