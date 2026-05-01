#!/usr/bin/env python3
"""Integration test: DQN vs Long-only comparison.

Supports:
- Single-contract evaluation (--ticker)
- Multi-seed evaluation with median +/- IQR (--seeds)
- Stitched r1+r2 evaluation (--stitch)
- Stitched from saved files, no retraining (--stitch --from-files)
- Reward persistence (.npz) for offline analysis (--save-rewards)
- Asset-class portfolio evaluation (--asset)
- Equal-weight portfolio aggregation across contracts (--asset --portfolio)

Usage examples:
    # Single-contract, single-seed
    python tests/test_integration_dqn_vs_long.py --ticker AN --round 1

    # Both rounds
    python tests/test_integration_dqn_vs_long.py --both

    # Multi-seed (5 seeds from LOCKED_SEEDS)
    python tests/test_integration_dqn_vs_long.py --ticker AN --seeds 5

    # Stitched r1+r2 (trains both rounds, concatenates rewards)
    python tests/test_integration_dqn_vs_long.py --ticker AN --stitch

    # Stitched from saved files (no retraining)
    python tests/test_integration_dqn_vs_long.py --ticker AN --stitch --from-files
    python tests/test_integration_dqn_vs_long.py --ticker AN --stitch --from-files --ver v2

    # Save rewards to .npz for later stitching
    python tests/test_integration_dqn_vs_long.py --ticker AN --round 1 --save-rewards
    python tests/test_integration_dqn_vs_long.py --ticker AN --round 2 --save-rewards --ver v2

    # Full asset-class evaluation (trains on all Forex, per-contract + portfolio)
    python tests/test_integration_dqn_vs_long.py --asset Forex --round 1

    # Asset-class portfolio with multi-seed
    python tests/test_integration_dqn_vs_long.py --asset Forex --seeds 3
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from config import ASSET_CLASSES
from drl.dqn.model import DQNAgent
from drl.dqn.spec import LOCKED_SEEDS
from drl.dqn.train.train_dqn_walkforward import (
    SIGMA_TGT_DEFAULT,
    load_contract_round,
    train_asset_round,
)
from drl_shared.spec import RETRAIN_ROUNDS, universe_tickers
from drl_shared.state_space import WARMUP, ContractEnv
from metrics import compute_metrics

# --- Paper's 7 trade metrics (excludes MDD / Calmar) ---
TRADE_METRIC_NAMES = ["E(R)", "std(R)", "DD", "Sharpe", "Sortino", "% +ve", "Ave P/L"]
_TRADE_IDX = [0, 1, 2, 3, 4, 7, 8]

RESULTS_ROOT = REPO / "results"
RESULTS_VERSION = "v1"


def _results_dir(version: str | None = None) -> Path:
    v = version or RESULTS_VERSION
    d = RESULTS_ROOT / v
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── helpers ──────────────────────────────────────────────────────────────────

def _run_dqn_test_period(
    agent: DQNAgent,
    contract,
    test_start_idx: int,
    sigma_tgt: float = SIGMA_TGT_DEFAULT,
) -> tuple[np.ndarray, list[float]]:
    n = len(contract.prices)
    env = ContractEnv(contract, sigma_tgt=sigma_tgt, start_idx=test_start_idx, max_idx=n)
    state = env.reset()
    rewards: list[float] = []
    positions: list[float] = []
    done = False
    while not done:
        action_id = agent.act(state, eps=0.0)
        positions.append({0: -1.0, 1: 0.0, 2: 1.0}[action_id])
        next_state, reward, done = env.step(action_id)
        rewards.append(reward)
        state = next_state
    return np.array(rewards, dtype=float), positions


def _run_long_test_period(
    contract,
    test_start_idx: int,
    sigma_tgt: float = SIGMA_TGT_DEFAULT,
) -> np.ndarray:
    n = len(contract.prices)
    env = ContractEnv(contract, sigma_tgt=sigma_tgt, start_idx=test_start_idx, max_idx=n)
    env.reset()
    rewards: list[float] = []
    done = False
    while not done:
        _, reward, done = env.step(2)
        rewards.append(reward)
    return np.array(rewards, dtype=float)


def _extract_trade_metrics(metrics_list: list[float]) -> dict[str, float]:
    return {TRADE_METRIC_NAMES[i]: metrics_list[idx] for i, idx in enumerate(_TRADE_IDX)}


def _position_dist(positions: list[float]) -> str:
    if not positions:
        return "N/A"
    n = len(positions)
    long_pct = sum(1 for p in positions if p > 0.5) / n * 100
    flat_pct = sum(1 for p in positions if abs(p) < 0.5) / n * 100
    short_pct = sum(1 for p in positions if p < -0.5) / n * 100
    unique = len(set(round(p, 1) for p in positions))
    return f"L={long_pct:.0f}% F={flat_pct:.0f}% S={short_pct:.0f}% | unique={unique}"


# ── persistence ────────────────────────────────────────────────────────────────

def _results_filename(ticker: str, round_num: int, seed: int) -> str:
    return f"{ticker}_r{round_num}_s{seed}.npz"


def _portfolio_results_filename(asset_name: str, round_num: int, seed: int) -> str:
    slug = asset_name.replace(" ", "_")
    return f"{slug}_r{round_num}_s{seed}_portfolio.npz"


def save_results(
    result: dict,
    results_dir: Path | str | None | bool = None,
) -> Path:
    if results_dir is None or results_dir is True:
        results_dir = _results_dir()
    elif isinstance(results_dir, bool):
        results_dir = _results_dir()
    else:
        results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    ticker = result["ticker"]
    round_num = result["round"]
    seed = result["seed"]

    filepath = results_dir / _results_filename(ticker, round_num, seed)
    np.savez_compressed(
        filepath,
        dqn_rewards=result["dqn_rewards"],
        long_rewards=result["long_rewards"],
        dqn_positions=np.array(result.get("dqn_positions", []), dtype=float),
        ticker=ticker,
        round_num=round_num,
        seed=seed,
        dqn_metrics=json.dumps(result["dqn_metrics"]),
        long_metrics=json.dumps(result["long_metrics"]),
        dqn_steps=result["dqn_steps"],
        long_steps=result["long_steps"],
        wins=result["wins"],
    )
    print(f"  Saved: {filepath}")
    return filepath


def load_results(
    ticker: str,
    round_num: int,
    seed: int,
    results_dir: Path | str | None | bool = None,
) -> dict:
    if results_dir is None or results_dir is True:
        results_dir = _results_dir()
    elif isinstance(results_dir, bool):
        results_dir = _results_dir()
    else:
        results_dir = Path(results_dir)
    filepath = results_dir / _results_filename(ticker, round_num, seed)
    if not filepath.exists():
        raise FileNotFoundError(f"No saved results at {filepath}")
    data = np.load(filepath, allow_pickle=True)
    return {
        "ticker": str(data["ticker"]),
        "round": int(data["round_num"]),
        "seed": int(data["seed"]),
        "dqn_rewards": data["dqn_rewards"],
        "long_rewards": data["long_rewards"],
        "dqn_positions": data["dqn_positions"].tolist(),
        "dqn_metrics": json.loads(str(data["dqn_metrics"])),
        "long_metrics": json.loads(str(data["long_metrics"])),
        "dqn_steps": int(data["dqn_steps"]),
        "long_steps": int(data["long_steps"]),
        "wins": int(data["wins"]),
    }


def save_portfolio_results(
    result: dict,
    results_dir: Path | str | None | bool = None,
) -> Path:
    if results_dir is None or results_dir is True:
        results_dir = _results_dir()
    elif isinstance(results_dir, bool):
        results_dir = _results_dir()
    else:
        results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    asset_name = result["asset_name"]
    round_num = result["round"]
    seed = result["seed"]

    filepath = results_dir / _portfolio_results_filename(asset_name, round_num, seed)
    contract_rewards_dqn = np.stack(
        [r["dqn_rewards"] for r in result["contract_results"].values()], axis=0
    )
    contract_rewards_long = np.stack(
        [r["long_rewards"] for r in result["contract_results"].values()], axis=0
    )
    tickers = list(result["contract_results"].keys())

    np.savez_compressed(
        filepath,
        dqn_portfolio_rewards=result["portfolio"]["dqn_rewards"],
        long_portfolio_rewards=result["portfolio"]["long_rewards"],
        contract_rewards_dqn=contract_rewards_dqn,
        contract_rewards_long=contract_rewards_long,
        tickers=tickers,
        asset_name=asset_name,
        round_num=round_num,
        seed=seed,
        dqn_portfolio_metrics=json.dumps(result["portfolio"]["dqn_metrics"]),
        long_portfolio_metrics=json.dumps(result["portfolio"]["long_metrics"]),
    )
    print(f"  Saved portfolio: {filepath}")
    return filepath


# ── printing ──────────────────────────────────────────────────────────────────

def _print_comparison_table(
    dqn_vals: dict[str, float],
    long_vals: dict[str, float],
    ticker: str,
    round_num: int,
    episodes: int,
    device: str,
    seed: int,
    dqn_positions: list[float] | None = None,
) -> int:
    header = f"{'Metric':>10s}  {'DQN':>9s}  {'Long':>9s}  {'Win':>6s}  {'Delta':>9s}"
    sep = "-" * 55
    print(f"\n{'=' * 70}")
    print(f"  {ticker}  |  Round: r{round_num}  |  Episodes: {episodes}  |  Device: {device}  |  Seed: {seed}")
    print(f"{'=' * 70}")
    print(header)
    print(sep)

    wins = 0
    for name in TRADE_METRIC_NAMES:
        d = dqn_vals[name]
        l = long_vals[name]
        diff = d - l
        if name in ("std(R)", "DD"):
            winner = "DQN" if d < l else ("Long" if d > l else "-")
        else:
            winner = "DQN" if d > l else ("Long" if d < l else "-")
        if winner == "DQN":
            wins += 1
        print(f"  {name:>10s}  {d:>+9.4f}  {l:>+9.4f}  {winner:>6s}  {diff:>+9.4f}")

    print(sep)
    print(f"  DQN wins {wins}/{len(TRADE_METRIC_NAMES)} metrics vs Long-only on this {ticker} test period")
    if dqn_positions:
        print(f"  Position dist: {_position_dist(dqn_positions)}")
    print(f"{'=' * 70}\n")
    return wins


def _print_multi_seed_summary(
    seed_results: list[dict],
    ticker: str,
    round_num: int,
    long_metrics: dict[str, float],
) -> int:
    print(f"\n{'=' * 70}")
    print(f"  {ticker} r{round_num} -- Multi-seed summary ({len(seed_results)} seeds: {LOCKED_SEEDS[:len(seed_results)]})")
    print(f"{'=' * 70}")
    print(f"  {'Metric':>10s}  {'DQN med':>9s}  {'DQN IQR':>9s}  {'Long':>9s}  {'Win':>6s}")
    print("  " + "-" * 51)

    total_wins = 0
    for name in TRADE_METRIC_NAMES:
        vals = [r["dqn_metrics"][name] for r in seed_results]
        median = float(np.median(vals))
        iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
        l = long_metrics[name]
        if name in ("std(R)", "DD"):
            winner = "DQN" if median < l else ("Long" if median > l else "-")
        else:
            winner = "DQN" if median > l else ("Long" if median < l else "-")
        if winner == "DQN":
            total_wins += 1
        print(f"  {name:>10s}  {median:>+9.4f}  +/-{iqr:<8.4f}  {l:>+9.4f}  {winner:>6s}")

    print("  " + "-" * 51)
    print(f"  DQN wins {total_wins}/{len(TRADE_METRIC_NAMES)} metrics (median across {len(seed_results)} seeds)")
    print(f"{'=' * 70}\n")
    return total_wins


def _print_per_contract_table(
    contract_results: dict[str, dict],
    asset_name: str,
    round_num: int,
    seed: int,
) -> None:
    tickers = list(contract_results.keys())
    print(f"\n{'=' * 70}")
    print(f"  {asset_name} r{round_num} -- Per-contract DQN vs Long  |  Seed: {seed}")
    print(f"{'=' * 70}")
    header = f"  {'Ticker':>6s}"
    for name in TRADE_METRIC_NAMES:
        header += f"  {'D_' + name[:4]:>7s}"
    header += f"  {'Wins':>4s}"
    print(header)
    print("  " + "-" * (8 + 7 * len(TRADE_METRIC_NAMES) + 5))

    for ticker in tickers:
        r = contract_results[ticker]
        dm = r["dqn_metrics"]
        line = f"  {ticker:>6s}"
        for name in TRADE_METRIC_NAMES:
            line += f"  {dm[name]:>+7.3f}"
        line += f"  {r['wins']:>4d}"
        print(line)

    print("  " + "-" * (8 + 7 * len(TRADE_METRIC_NAMES) + 5))
    print(f"  Avg wins: {np.mean([contract_results[t]['wins'] for t in tickers]):.1f}/{len(TRADE_METRIC_NAMES)}")
    print(f"{'=' * 70}\n")


def _print_portfolio_table(
    dqn_metrics: dict[str, float],
    long_metrics: dict[str, float],
    asset_name: str,
    round_num: int,
    n_contracts: int,
    n_steps: int,
    seed: int,
) -> int:
    print(f"\n{'=' * 70}")
    print(f"  {asset_name} r{round_num} -- EQUAL-WEIGHT PORTFOLIO ({n_contracts} contracts, {n_steps} steps)  |  Seed: {seed}")
    print(f"{'=' * 70}")
    print(f"  {'Metric':>10s}  {'DQN':>9s}  {'Long':>9s}  {'Win':>6s}  {'Delta':>9s}")
    print("  " + "-" * 55)

    wins = 0
    for name in TRADE_METRIC_NAMES:
        d = dqn_metrics[name]
        l = long_metrics[name]
        diff = d - l
        if name in ("std(R)", "DD"):
            winner = "DQN" if d < l else ("Long" if d > l else "-")
        else:
            winner = "DQN" if d > l else ("Long" if d < l else "-")
        if winner == "DQN":
            wins += 1
        print(f"  {name:>10s}  {d:>+9.4f}  {l:>+9.4f}  {winner:>6s}  {diff:>+9.4f}")

    print("  " + "-" * 55)
    print(f"  DQN wins {wins}/{len(TRADE_METRIC_NAMES)} metrics (portfolio)")
    print(f"{'=' * 70}\n")
    return wins


# ── main comparison routines ──────────────────────────────────────────────────

def compare_single_contract(
    ticker: str = "AN",
    round_num: int = 1,
    episodes: int = 100,
    device: str = "auto",
    sigma_tgt: float = SIGMA_TGT_DEFAULT,
    seed: int = 42,
    save_dir: Path | str | None | bool = None,
) -> dict:
    """Train DQN on a single contract, compare against Long-only.

    Args:
        save_dir: If True/None, use results/<RESULTS_VERSION>/. If a path, use that.
                 If False, don't save. Saves .npz reward arrays.
    """
    round_info = RETRAIN_ROUNDS[round_num]
    t_total_start = time.time()

    full_contract, feature_meta = load_contract_round(ticker, round_num)
    test_start_idx = int(feature_meta["test_start_idx"])
    test_end_idx = int(feature_meta["test_end_idx"])
    train_end_idx = int(feature_meta["train_end_idx"])

    print(f"\n  {ticker} r{round_num}: train [{feature_meta['train_start']} ~ {feature_meta['train_end']}] "
          f"({train_end_idx - int(feature_meta['train_start_idx']) + 1} days)")
    print(f"  {ticker} r{round_num}: test  [{feature_meta['test_start']} ~ {feature_meta['test_end']}] "
          f"({test_end_idx - test_start_idx + 1} days)")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_model_root = Path(tmpdir) / "models"
        with mock.patch("drl.dqn.spec.MODEL_ROOT", tmp_model_root):
            t_train_start = time.time()
            checkpoint_path, log_dir = train_asset_round(
                asset_name="Forex",
                round_num=round_num,
                episodes=episodes,
                device=device,
                seed=seed,
                sigma_tgt=sigma_tgt,
                tickers_override=[ticker],
            )
            t_train_done = time.time()

            agent = DQNAgent(device="cpu")
            agent.load(checkpoint_path)

    t_infer_start = time.time()
    dqn_rewards, dqn_positions = _run_dqn_test_period(agent, full_contract, test_start_idx, sigma_tgt)
    t_infer_done = time.time()

    long_rewards = _run_long_test_period(full_contract, test_start_idx, sigma_tgt)

    dqn_full = compute_metrics(dqn_rewards, n_contracts=1, round_output=False)
    long_full = compute_metrics(long_rewards, n_contracts=1, round_output=False)

    dqn_metrics = _extract_trade_metrics(dqn_full)
    long_metrics = _extract_trade_metrics(long_full)

    wins = _print_comparison_table(
        dqn_metrics, long_metrics, ticker, round_num, episodes, device, seed, dqn_positions
    )

    t_total = time.time() - t_total_start
    print(f"  Timing: train={t_train_done - t_train_start:.1f}s  "
          f"infer={t_infer_done - t_infer_start:.1f}s  total={t_total:.1f}s")

    result = {
        "ticker": ticker,
        "round": round_num,
        "episodes": episodes,
        "device": device,
        "seed": seed,
        "dqn_metrics": dqn_metrics,
        "long_metrics": long_metrics,
        "dqn_rewards": dqn_rewards,
        "long_rewards": long_rewards,
        "dqn_positions": dqn_positions,
        "dqn_steps": len(dqn_rewards),
        "long_steps": len(long_rewards),
        "unique_positions": len(set(round(p, 1) for p in dqn_positions)),
        "train_time_s": t_train_done - t_train_start,
        "infer_time_s": t_infer_done - t_infer_start,
        "total_time_s": t_total,
        "wins": wins,
    }

    if save_dir is not None and save_dir is not False:
        save_results(result, save_dir)

    return result


def compare_stitched(
    ticker: str = "AN",
    episodes: int = 100,
    device: str = "auto",
    sigma_tgt: float = SIGMA_TGT_DEFAULT,
    seed: int = 42,
    save_dir: Path | str | None | bool = None,
) -> dict:
    """Train DQN on r1+r2 separately, stitch test-period rewards, compute full-period metrics."""
    r1 = compare_single_contract(
        ticker=ticker, round_num=1, episodes=episodes,
        device=device, sigma_tgt=sigma_tgt, seed=seed,
    )
    r2 = compare_single_contract(
        ticker=ticker, round_num=2, episodes=episodes,
        device=device, sigma_tgt=sigma_tgt, seed=seed,
    )

    dqn_stitched = np.concatenate([r1["dqn_rewards"], r2["dqn_rewards"]])
    long_stitched = np.concatenate([r1["long_rewards"], r2["long_rewards"]])

    dqn_full = compute_metrics(dqn_stitched, n_contracts=1, round_output=False)
    long_full = compute_metrics(long_stitched, n_contracts=1, round_output=False)
    dqn_metrics = _extract_trade_metrics(dqn_full)
    long_metrics = _extract_trade_metrics(long_full)

    print(f"\n{'=' * 70}")
    print(f"  {ticker} -- STITCHED r1+r2 (2011-2019) | Seed: {seed}")
    print(f"  DQN: {len(dqn_stitched)} steps | Long: {len(long_stitched)} steps")
    print(f"{'=' * 70}")
    print(f"  {'Metric':>10s}  {'DQN':>9s}  {'Long':>9s}  {'Win':>6s}  {'Delta':>9s}")
    print("  " + "-" * 55)

    wins = 0
    for name in TRADE_METRIC_NAMES:
        d = dqn_metrics[name]
        l = long_metrics[name]
        diff = d - l
        if name in ("std(R)", "DD"):
            winner = "DQN" if d < l else ("Long" if d > l else "-")
        else:
            winner = "DQN" if d > l else ("Long" if d < l else "-")
        if winner == "DQN":
            wins += 1
        print(f"  {name:>10s}  {d:>+9.4f}  {l:>+9.4f}  {winner:>6s}  {diff:>+9.4f}")
    print("  " + "-" * 55)
    print(f"  DQN wins {wins}/{len(TRADE_METRIC_NAMES)} metrics (stitched r1+r2)")
    print(f"{'=' * 70}\n")

    result = {
        "ticker": ticker,
        "dqn_metrics": dqn_metrics,
        "long_metrics": long_metrics,
        "dqn_rewards": dqn_stitched,
        "long_rewards": long_stitched,
        "dqn_steps": len(dqn_stitched),
        "long_steps": len(long_stitched),
        "wins": wins,
        "r1": r1,
        "r2": r2,
}

    if save_dir is not None and save_dir is not False:
        save_portfolio_results(result, save_dir)

    return result


def compare_stitched_from_files(
    ticker: str = "AN",
    seed: int = 42,
    results_dir: Path | str | None = None,
) -> dict:
    """Load saved r1 and r2 reward arrays from .npz, compute stitched metrics (no training)."""
    r1 = load_results(ticker, 1, seed, results_dir)
    r2 = load_results(ticker, 2, seed, results_dir)

    dqn_stitched = np.concatenate([r1["dqn_rewards"], r2["dqn_rewards"]])
    long_stitched = np.concatenate([r1["long_rewards"], r2["long_rewards"]])

    dqn_full = compute_metrics(dqn_stitched, n_contracts=1, round_output=False)
    long_full = compute_metrics(long_stitched, n_contracts=1, round_output=False)
    dqn_metrics = _extract_trade_metrics(dqn_full)
    long_metrics = _extract_trade_metrics(long_full)

    print(f"\n{'=' * 70}")
    print(f"  {ticker} -- STITCHED r1+r2 (from files) | Seed: {seed}")
    print(f"  DQN: {len(dqn_stitched)} steps | Long: {len(long_stitched)} steps")
    print(f"{'=' * 70}")
    print(f"  {'Metric':>10s}  {'DQN':>9s}  {'Long':>9s}  {'Win':>6s}  {'Delta':>9s}")
    print("  " + "-" * 55)

    wins = 0
    for name in TRADE_METRIC_NAMES:
        d = dqn_metrics[name]
        l = long_metrics[name]
        diff = d - l
        if name in ("std(R)", "DD"):
            winner = "DQN" if d < l else ("Long" if d > l else "-")
        else:
            winner = "DQN" if d > l else ("Long" if d < l else "-")
        if winner == "DQN":
            wins += 1
        print(f"  {name:>10s}  {d:>+9.4f}  {l:>+9.4f}  {winner:>6s}  {diff:>+9.4f}")
    print("  " + "-" * 55)
    print(f"  DQN wins {wins}/{len(TRADE_METRIC_NAMES)} metrics (stitched from files)")
    print(f"{'=' * 70}\n")

    return {
        "ticker": ticker,
        "dqn_metrics": dqn_metrics,
        "long_metrics": long_metrics,
        "dqn_rewards": dqn_stitched,
        "long_rewards": long_stitched,
        "dqn_steps": len(dqn_stitched),
        "long_steps": len(long_stitched),
        "wins": wins,
        "r1": r1,
        "r2": r2,
    }


def compare_asset_class(
    asset_name: str = "Forex",
    round_num: int = 1,
    episodes: int = 100,
    device: str = "auto",
    sigma_tgt: float = SIGMA_TGT_DEFAULT,
    seed: int = 42,
    save_dir: Path | str | None | bool = None,
) -> dict:
    """Train one DQN on the full asset class, evaluate each contract + equal-weight portfolio.

    Paper methodology: one shared model per asset class per round. Each contract is evaluated
    independently, then per-contract Eq.4 rewards are averaged (equal-weight) to form the
    portfolio return series.
    """
    tickers = universe_tickers(asset_name)
    if not tickers:
        raise ValueError(f"No tickers found for asset class '{asset_name}'")

    print(f"\n{'=' * 70}")
    print(f"  ASSET-CLASS EVALUATION: {asset_name} ({len(tickers)} contracts) r{round_num}  |  Seed: {seed}")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"{'=' * 70}")

    t_total_start = time.time()

    # 1. Train one shared DQN on the full asset class
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_model_root = Path(tmpdir) / "models"
        with mock.patch("drl.dqn.spec.MODEL_ROOT", tmp_model_root):
            t_train_start = time.time()
            checkpoint_path, log_dir = train_asset_round(
                asset_name=asset_name,
                round_num=round_num,
                episodes=episodes,
                device=device,
                seed=seed,
                sigma_tgt=sigma_tgt,
            )
            t_train_done = time.time()
            print(f"  Training complete in {t_train_done - t_train_start:.1f}s")

            agent = DQNAgent(device="cpu")
            agent.load(checkpoint_path)

    # 2. Evaluate each contract
    contract_results: dict[str, dict] = {}
    skipped: list[str] = []
    all_dqn_rewards: list[np.ndarray] = []
    all_long_rewards: list[np.ndarray] = []

    for ticker in tickers:
        try:
            full_contract, feature_meta = load_contract_round(ticker, round_num)
        except (FileNotFoundError, ValueError) as e:
            print(f"  SKIP {ticker}: {e}")
            skipped.append(ticker)
            continue

        test_start_idx = int(feature_meta["test_start_idx"])

        dqn_rewards, dqn_positions = _run_dqn_test_period(
            agent, full_contract, test_start_idx, sigma_tgt
        )
        long_rewards = _run_long_test_period(full_contract, test_start_idx, sigma_tgt)

        dqn_full = compute_metrics(dqn_rewards, n_contracts=1, round_output=False)
        long_full = compute_metrics(long_rewards, n_contracts=1, round_output=False)
        dqn_metrics = _extract_trade_metrics(dqn_full)
        long_metrics = _extract_trade_metrics(long_full)

        wins = 0
        for name in TRADE_METRIC_NAMES:
            d = dqn_metrics[name]
            l = long_metrics[name]
            if name in ("std(R)", "DD"):
                if d < l:
                    wins += 1
            else:
                if d > l:
                    wins += 1

        print(f"  {ticker}: DQN wins {wins}/{len(TRADE_METRIC_NAMES)} | "
              f"Sharpe DQN={dqn_metrics['Sharpe']:+.4f} Long={long_metrics['Sharpe']:+.4f} | "
              f"{_position_dist(dqn_positions)}")

        contract_results[ticker] = {
            "dqn_metrics": dqn_metrics,
            "long_metrics": long_metrics,
            "dqn_rewards": dqn_rewards,
            "long_rewards": long_rewards,
            "dqn_positions": dqn_positions,
            "dqn_steps": len(dqn_rewards),
            "long_steps": len(long_rewards),
            "wins": wins,
        }
        all_dqn_rewards.append(dqn_rewards)
        all_long_rewards.append(long_rewards)

    if skipped:
        print(f"  Skipped tickers: {', '.join(skipped)}")

    n_eval = len(contract_results)
    if n_eval == 0:
        raise RuntimeError(f"No contracts evaluated for {asset_name} r{round_num}")

    # 3. Equal-weight portfolio (Paper Eq.13: R_port = 1/N * sum_i R_i,t)
    min_len = min(len(r) for r in all_dqn_rewards)
    dqn_portfolio = np.mean([r[:min_len] for r in all_dqn_rewards], axis=0)
    long_portfolio = np.mean([r[:min_len] for r in all_long_rewards], axis=0)

    dqn_port_full = compute_metrics(dqn_portfolio, n_contracts=n_eval, round_output=False)
    long_port_full = compute_metrics(long_portfolio, n_contracts=n_eval, round_output=False)
    dqn_port_metrics = _extract_trade_metrics(dqn_port_full)
    long_port_metrics = _extract_trade_metrics(long_port_full)

    # 4. Print results
    _print_per_contract_table(contract_results, asset_name, round_num, seed)
    port_wins = _print_portfolio_table(
        dqn_port_metrics, long_port_metrics, asset_name, round_num,
        n_eval, min_len, seed,
    )

    t_total = time.time() - t_total_start
    print(f"  Total time: {t_total:.1f}s (train={t_train_done - t_train_start:.1f}s)")

    result = {
        "asset_name": asset_name,
        "round": round_num,
        "episodes": episodes,
        "device": device,
        "seed": seed,
        "tickers": list(contract_results.keys()),
        "skipped": skipped,
        "contract_results": contract_results,
        "portfolio": {
            "dqn_metrics": dqn_port_metrics,
            "long_metrics": long_port_metrics,
            "dqn_rewards": dqn_portfolio,
            "long_rewards": long_portfolio,
            "n_contracts": n_eval,
            "n_steps": min_len,
            "wins": port_wins,
        },
        "train_time_s": t_train_done - t_train_start,
        "total_time_s": t_total,
    }

    if save_dir is not None and save_dir is not False:
        save_portfolio_results(result, save_dir)

    return result


# ── pytest entry ───────────────────────────────────────────────────────────────

def test_integration_dqn_vs_long_r1():
    """Integration test: train DQN (r1) on AN, compare 7 trade metrics vs Long-only."""
    result = compare_single_contract(ticker="AN", round_num=1, episodes=100, device="auto", seed=LOCKED_SEEDS[0])
    assert result["dqn_steps"] == result["long_steps"], \
        f"Step mismatch: DQN={result['dqn_steps']} vs Long={result['long_steps']}"
    assert result["dqn_steps"] > 0, "DQN produced 0 steps"
    for name in TRADE_METRIC_NAMES:
        assert np.isfinite(result["dqn_metrics"][name]), f"DQN {name} is NaN/Inf"
        assert np.isfinite(result["long_metrics"][name]), f"Long {name} is NaN/Inf"
    assert result["unique_positions"] >= 2, \
        f"DQN degenerate: only {result['unique_positions']} unique position(s)"


def test_integration_dqn_vs_long_r2():
    """Integration test: train DQN (r2) on AN, compare 7 trade metrics vs Long-only."""
    result = compare_single_contract(ticker="AN", round_num=2, episodes=100, device="auto", seed=LOCKED_SEEDS[0])
    assert result["dqn_steps"] == result["long_steps"]
    assert result["dqn_steps"] > 0
    for name in TRADE_METRIC_NAMES:
        assert np.isfinite(result["dqn_metrics"][name])
        assert np.isfinite(result["long_metrics"][name])
    assert result["unique_positions"] >= 2


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DQN vs Long-only comparison (single-contract, multi-seed, portfolio)"
    )
    parser.add_argument("--ticker", default=None, help="Single contract ticker (e.g. AN)")
    parser.add_argument("--round", type=int, default=None, choices=[1, 2])
    parser.add_argument("--both", action="store_true", help="Run both r1 and r2")
    parser.add_argument("--stitch", action="store_true", help="Stitch r1+r2 test rewards into one full-period comparison")
    parser.add_argument("--from-files", action="store_true", help="Load saved .npz rewards instead of retraining (requires --stitch)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--sigma-tgt", type=float, default=SIGMA_TGT_DEFAULT)
    parser.add_argument("--seed", type=int, default=None, help="Single seed (overrides --seeds)")
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds from LOCKED_SEEDS to run")
    parser.add_argument("--save-rewards", action="store_true", help="Save reward arrays to results/<ver>/ as .npz")
    parser.add_argument("--asset", default=None, help="Asset class for portfolio evaluation (e.g. Forex)")
    parser.add_argument("--ver", default=None, help=f"Results version directory (default: {RESULTS_VERSION}). Changes only matter when saving/loading.")
    args = parser.parse_args()

    save_dir = _results_dir(args.ver) if args.save_rewards else None

    seeds = [args.seed] if args.seed is not None else LOCKED_SEEDS[:min(args.seeds, len(LOCKED_SEEDS))]

    # ── Stitched from files (no training) ──
    load_dir = _results_dir(args.ver)
    if args.stitch and args.from_files:
        for seed in seeds:
            compare_stitched_from_files(
                ticker=args.ticker or "AN",
                seed=seed,
                results_dir=load_dir,
            )
        return

    # ── Stitched with training ──
    if args.stitch:
        for seed in seeds:
            compare_stitched(
                ticker=args.ticker or "AN",
                episodes=args.episodes,
                device=args.device,
                sigma_tgt=args.sigma_tgt,
                seed=seed,
                save_dir=save_dir,
            )
        return

    # ── Asset-class portfolio evaluation ──
    if args.asset:
        for seed in seeds:
            for rn in ([1, 2] if args.both or args.round is None else [args.round]):
                compare_asset_class(
                    asset_name=args.asset,
                    round_num=rn,
                    episodes=args.episodes,
                    device=args.device,
                    sigma_tgt=args.sigma_tgt,
                    seed=seed,
                    save_dir=save_dir,
                )
        return

    # ── Single-contract evaluation ──
    ticker = (args.ticker or "AN").upper()
    rounds = [1, 2] if args.both or args.round is None else [args.round]

    all_results: list[dict] = []
    for rn in rounds:
        seed_results: list[dict] = []
        for seed in seeds:
            res = compare_single_contract(
                ticker=ticker,
                round_num=rn,
                episodes=args.episodes,
                device=args.device,
                sigma_tgt=args.sigma_tgt,
                seed=seed,
                save_dir=save_dir,
            )
            seed_results.append(res)
            all_results.append(res)

        if len(seed_results) > 1:
            _print_multi_seed_summary(
                seed_results, ticker, rn, seed_results[0]["long_metrics"]
            )

    if len(rounds) == 2 and len(seeds) == 1:
        print("-" * 70)
        print("  Summary across rounds")
        print("-" * 70)
        total_wins = 0
        for res in all_results:
            wins = sum(
                1 for name in TRADE_METRIC_NAMES
                if (
                    name in ("std(R)", "DD")
                    and res["dqn_metrics"][name] < res["long_metrics"][name]
                ) or (
                    name not in ("std(R)", "DD")
                    and res["dqn_metrics"][name] > res["long_metrics"][name]
                )
            )
            total_wins += wins
            print(f"  r{res['round']}: DQN wins {wins}/{len(TRADE_METRIC_NAMES)} metrics")
        print(f"  Total: DQN wins {total_wins}/{len(TRADE_METRIC_NAMES) * 2} metrics across r1+r2")


if __name__ == "__main__":
    main()