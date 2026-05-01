#!/usr/bin/env python3
"""Integration test: single-contract DQN vs Long-only comparison.

Trains DQN on a single contract, then compares against Long-only on the
test period using the paper's 7 trade metrics (excluding MDD/Calmar).

Usage as script:
    python tests/test_integration_dqn_vs_long.py --ticker AN --episodes 100
    python tests/test_integration_dqn_vs_long.py --ticker AN --round 1 --episodes 100 --device mps
    python tests/test_integration_dqn_vs_long.py --both --episodes 100 --device mps

Usage as pytest:
    pytest tests/test_integration_dqn_vs_long.py -v -s
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from drl.dqn.model import DQNAgent
from drl.dqn.train.train_dqn_walkforward import (
    SIGMA_TGT_DEFAULT,
    load_contract_round,
    train_asset_round,
)
from drl_shared.spec import RETRAIN_ROUNDS, SEQ_LEN
from drl_shared.state_space import WARMUP, ContractEnv
from metrics import compute_metrics

# --- Paper's 7 trade metrics (excludes MDD / Calmar) ---
TRADE_METRIC_NAMES = ["E(R)", "std(R)", "DD", "Sharpe", "Sortino", "% +ve", "Ave P/L"]
# Indices in compute_metrics() output: [0:E(R), 1:std, 2:DD, 3:Sharpe, 4:Sortino, 5:MDD, 6:Calmar, 7:%+ve, 8:AveP/L]
_TRADE_IDX = [0, 1, 2, 3, 4, 7, 8]


# ── helpers ──────────────────────────────────────────────────────────────────

def _run_dqn_test_period(
    agent: DQNAgent,
    contract,
    test_start_idx: int,
    sigma_tgt: float = SIGMA_TGT_DEFAULT,
) -> np.ndarray:
    """Run greedy DQN inference on the test period, return reward array."""
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
    """Run Long-only strategy on the test period, return reward array."""
    n = len(contract.prices)
    env = ContractEnv(contract, sigma_tgt=sigma_tgt, start_idx=test_start_idx, max_idx=n)
    state = env.reset()  # noqa: F841
    rewards: list[float] = []
    done = False
    while not done:
        _, reward, done = env.step(2)  # action_id=2 → position=+1 (long)
        rewards.append(reward)
    return np.array(rewards, dtype=float)


def _extract_trade_metrics(metrics_list: list[float]) -> dict[str, float]:
    """Extract the 7 trade metrics from the 9-metric list."""
    return {TRADE_METRIC_NAMES[i]: metrics_list[idx] for i, idx in enumerate(_TRADE_IDX)}


def _print_comparison_table(
    dqn_vals: dict[str, float],
    long_vals: dict[str, float],
    ticker: str,
    round_num: int,
    episodes: int,
    device: str,
    dqn_positions: list[float] | None = None,
):
    """Print side-by-side comparison table."""
    header = f"{'Metric':>10s}  {'DQN':>9s}  {'Long':>9s}  {'Win':>6s}  {'Δ':>9s}"
    sep = "-" * 55
    print(f"\n{'=' * 70}")
    print(f"  {ticker}  |  Round: r{round_num}  |  Episodes: {episodes}  |  Device: {device}")
    print(f"{'=' * 70}")
    print(header)
    print(sep)

    wins = 0
    for name in TRADE_METRIC_NAMES:
        d = dqn_vals[name]
        l = long_vals[name]
        diff = d - l
        # Higher is better for all 7 metrics (E(R), Sharpe, Sortino, %+ve, AveP/L)
        # For std(R) and DD, LOWER is better
        if name in ("std(R)", "DD"):
            winner = "DQN" if d < l else ("Long" if d > l else "—")
        else:
            winner = "DQN" if d > l else ("Long" if d < l else "—")
        if winner == "DQN":
            wins += 1
        print(f"  {name:>10s}  {d:>+9.4f}  {l:>+9.4f}  {winner:>6s}  {diff:>+9.4f}")

    print(sep)
    print(f"  DQN wins {wins}/{len(TRADE_METRIC_NAMES)} metrics vs Long-only on this {ticker} test period")
    if dqn_positions:
        unique = len(set(round(p, 1) for p in dqn_positions))
        long_pct = sum(1 for p in dqn_positions if p > 0.5) / max(1, len(dqn_positions)) * 100
        short_pct = sum(1 for p in dqn_positions if p < -0.5) / max(1, len(dqn_positions)) * 100
        flat_pct = sum(1 for p in dqn_positions if abs(p) < 0.5) / max(1, len(dqn_positions)) * 100
        print(f"  Position dist: L={long_pct:.0f}%  F={flat_pct:.0f}%  S={short_pct:.0f}%  |  unique={unique}")
    print(f"{'=' * 70}\n")


# ── main comparison routine ──────────────────────────────────────────────────

def compare_single_contract(
    ticker: str = "AN",
    round_num: int = 1,
    episodes: int = 100,
    device: str = "auto",
    sigma_tgt: float = SIGMA_TGT_DEFAULT,
    seed: int = 42,
) -> dict:
    """Train DQN on a single contract, then compare against Long-only.

    Returns a dict with DQN/Long 7-metric values, win counts, and timing info.
    """
    round_info = RETRAIN_ROUNDS[round_num]
    t_total_start = time.time()

    # ── 1. Load contract & extract test-period boundaries ──
    full_contract, feature_meta = load_contract_round(ticker, round_num)
    test_start_idx = int(feature_meta["test_start_idx"])
    test_end_idx = int(feature_meta["test_end_idx"])
    train_end_idx = int(feature_meta["train_end_idx"])

    print(f"\n  {ticker} r{round_num}: train [{feature_meta['train_start']} ~ {feature_meta['train_end']}] "
          f"({train_end_idx - int(feature_meta['train_start_idx']) + 1} days)")
    print(f"  {ticker} r{round_num}: test  [{feature_meta['test_start']} ~ {feature_meta['test_end']}] "
          f"({test_end_idx - test_start_idx + 1} days)")

    # ── 2. Train DQN in a temp directory ──
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

            # ── 3. Load trained agent ──
            agent = DQNAgent(device="cpu")
            agent.load(checkpoint_path)

    # ── 4. DQN greedy evaluation on test period ──
    t_infer_start = time.time()
    dqn_rewards, dqn_positions = _run_dqn_test_period(agent, full_contract, test_start_idx, sigma_tgt)
    t_infer_done = time.time()

    # ── 5. Long-only evaluation on same test period ──
    long_rewards = _run_long_test_period(full_contract, test_start_idx, sigma_tgt)

    # ── 6. Compute 7 trade metrics ──
    dqn_full = compute_metrics(dqn_rewards, n_contracts=1, round_output=False)
    long_full = compute_metrics(long_rewards, n_contracts=1, round_output=False)

    dqn_metrics = _extract_trade_metrics(dqn_full)
    long_metrics = _extract_trade_metrics(long_full)

    # ── 7. Print comparison ──
    _print_comparison_table(dqn_metrics, long_metrics, ticker, round_num, episodes, device, dqn_positions)

    # ── 8. Timing ──
    t_total = time.time() - t_total_start
    print(f"  Timing: train={t_train_done - t_train_start:.1f}s  "
          f"infer={t_infer_done - t_infer_start:.1f}s  total={t_total:.1f}s")

    return {
        "ticker": ticker,
        "round": round_num,
        "episodes": episodes,
        "device": device,
        "dqn_metrics": dqn_metrics,
        "long_metrics": long_metrics,
        "dqn_steps": len(dqn_rewards),
        "long_steps": len(long_rewards),
        "unique_positions": len(set(round(p, 1) for p in dqn_positions)),
        "train_time_s": t_train_done - t_train_start,
        "infer_time_s": t_infer_done - t_infer_start,
        "total_time_s": t_total,
    }


# ── pytest entry ─────────────────────────────────────────────────────────────

def test_integration_dqn_vs_long_r1():
    """Integration test: train DQN (r1) on AN, compare 7 trade metrics vs Long-only."""
    result = compare_single_contract(ticker="AN", round_num=1, episodes=100, device="auto", seed=42)
    # Sanity assertions
    assert result["dqn_steps"] == result["long_steps"], \
        f"Step mismatch: DQN={result['dqn_steps']} vs Long={result['long_steps']}"
    assert result["dqn_steps"] > 0, "DQN produced 0 steps"
    for name in TRADE_METRIC_NAMES:
        assert np.isfinite(result["dqn_metrics"][name]), f"DQN {name} is NaN/Inf"
        assert np.isfinite(result["long_metrics"][name]), f"Long {name} is NaN/Inf"
    # DQN should use more than 1 position (not degenerate all-flat or all-long)
    assert result["unique_positions"] >= 2, \
        f"DQN degenerate: only {result['unique_positions']} unique position(s)"


def test_integration_dqn_vs_long_r2():
    """Integration test: train DQN (r2) on AN, compare 7 trade metrics vs Long-only."""
    result = compare_single_contract(ticker="AN", round_num=2, episodes=100, device="auto", seed=42)
    assert result["dqn_steps"] == result["long_steps"]
    assert result["dqn_steps"] > 0
    for name in TRADE_METRIC_NAMES:
        assert np.isfinite(result["dqn_metrics"][name])
        assert np.isfinite(result["long_metrics"][name])
    assert result["unique_positions"] >= 2


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Single-contract DQN vs Long-only 7-metric comparison"
    )
    parser.add_argument("--ticker", default="AN")
    parser.add_argument("--round", type=int, default=None, choices=[1, 2])
    parser.add_argument("--both", action="store_true", help="Run both r1 and r2")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--sigma-tgt", type=float, default=SIGMA_TGT_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rounds = [1, 2] if args.both or args.round is None else [args.round]
    all_results = []
    for rn in rounds:
        res = compare_single_contract(
            ticker=args.ticker.upper(),
            round_num=rn,
            episodes=args.episodes,
            device=args.device,
            sigma_tgt=args.sigma_tgt,
            seed=args.seed,
        )
        all_results.append(res)

    if len(all_results) == 2:
        print("─" * 70)
        print("  Summary across rounds")
        print("─" * 70)
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
