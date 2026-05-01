#!/usr/bin/env python3
"""Feature occlusion analysis for trained DQN models.

For each feature dimension, occludes (zeros out) that feature during greedy
inference and measures the impact on 7 trade metrics, Q-value distributions,
and action positions vs baseline.

Usage:
    # Single-contract occlusion (trains, then occludes)
    python tests/test_feature_occlusion.py --ticker AN --round 1 --episodes 100

    # Both rounds
    python tests/test_feature_occlusion.py --ticker AN --both

    # Stitched r1+r2 (occlusion on stitched baseline)
    python tests/test_feature_occlusion.py --ticker AN --stitch

    # Multi-seed with median/IQR
    python tests/test_feature_occlusion.py --ticker AN --seeds 5

    # Asset-class portfolio occlusion
    python tests/test_feature_occlusion.py --asset Forex --round 1
"""
from __future__ import annotations

import argparse
import copy
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
from drl_shared.spec import FEATURE_DIM, RETRAIN_ROUNDS, universe_tickers
from drl_shared.state_space import WARMUP, ContractArrays, ContractEnv, action_id_to_position
from metrics import compute_metrics

TRADE_METRIC_NAMES = ["E(R)", "std(R)", "DD", "Sharpe", "Sortino", "% +ve", "Ave P/L"]
_TRADE_IDX = [0, 1, 2, 3, 4, 7, 8]

FEATURE_LABELS = {
    0: "ret_1d_vol_norm",
    1: "ret_21d_vol_norm",
    2: "macd_8_24",
    3: "macd_16_48",
    4: "rsi_30",
}


def _extract_trade_metrics(metrics_list: list[float]) -> dict[str, float]:
    return {TRADE_METRIC_NAMES[i]: metrics_list[idx] for i, idx in enumerate(_TRADE_IDX)}


def _position_dist_pct(positions: list[float]) -> dict[str, float]:
    n = len(positions)
    if n == 0:
        return {"long": 0.0, "flat": 0.0, "short": 0.0}
    return {
        "long": sum(1 for p in positions if p > 0.5) / n * 100,
        "flat": sum(1 for p in positions if abs(p) < 0.5) / n * 100,
        "short": sum(1 for p in positions if p < -0.5) / n * 100,
    }


def _occlude_contract_features(
    contract: ContractArrays,
    feature_idx: int,
    method: str = "zero",
) -> ContractArrays:
    """Return a copy of the contract with one feature column occluded.

    Args:
        contract: Original ContractArrays.
        feature_idx: Column index in features to occlude (0..4).
        method: "zero" replaces with zeros; "mean" replaces with training-set mean.
    """
    occluded = copy.copy(contract)
    feats = contract.features.copy()
    if method == "zero":
        feats[:, feature_idx] = 0.0
    elif method == "mean":
        feats[:, feature_idx] = np.mean(feats[:, feature_idx])
    else:
        raise ValueError(f"Unknown occlusion method: {method}")
    occluded.features = feats
    return occluded


def _run_dqn_test_period(
    agent: DQNAgent,
    contract: ContractArrays,
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
        positions.append(action_id_to_position(action_id))
        next_state, reward, done = env.step(action_id)
        rewards.append(reward)
        state = next_state
    return np.array(rewards, dtype=float), positions


def _run_long_test_period(
    contract: ContractArrays,
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


def feature_occlusion_single(
    ticker: str = "AN",
    round_num: int = 1,
    episodes: int = 100,
    device: str = "auto",
    sigma_tgt: float = SIGMA_TGT_DEFAULT,
    seed: int = 42,
    occlusion_method: str = "zero",
) -> dict:
    """Train DQN on a single contract, then run feature occlusion on test period.

    Returns a dict with baseline and per-feature-occluded results.
    """
    full_contract, feature_meta = load_contract_round(ticker, round_num)
    test_start_idx = int(feature_meta["test_start_idx"])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_model_root = Path(tmpdir) / "models"
        with mock.patch("drl.dqn.spec.MODEL_ROOT", tmp_model_root):
            t0 = time.time()
            checkpoint_path, log_dir = train_asset_round(
                asset_name="Forex",
                round_num=round_num,
                episodes=episodes,
                device=device,
                seed=seed,
                sigma_tgt=sigma_tgt,
                tickers_override=[ticker],
            )
            t_train = time.time() - t0

            agent = DQNAgent(device="cpu")
            agent.load(checkpoint_path)

    # Baseline
    baseline_rewards, baseline_positions = _run_dqn_test_period(
        agent, full_contract, test_start_idx, sigma_tgt
    )
    baseline_metrics = _extract_trade_metrics(
        compute_metrics(baseline_rewards, n_contracts=1, round_output=False)
    )

    # Feature occlusion
    occlusion_results = {}
    for fi in range(FEATURE_DIM):
        fname = FEATURE_LABELS.get(fi, f"F{fi}")
        occluded_contract = _occlude_contract_features(
            full_contract, fi, method=occlusion_method
        )
        occ_rewards, occ_positions = _run_dqn_test_period(
            agent, occluded_contract, test_start_idx, sigma_tgt
        )
        occ_metrics = _extract_trade_metrics(
            compute_metrics(occ_rewards, n_contracts=1, round_output=False)
        )
        occ_dist = _position_dist_pct(occ_positions)
        occlusion_results[fname] = {
            "feature_idx": fi,
            "method": occlusion_method,
            "metrics": occ_metrics,
            "position_dist": occ_dist,
            "steps": len(occ_rewards),
            "dqn_rewards": occ_rewards,
            "dqn_positions": occ_positions,
        }

    # Long-only baseline
    long_rewards = _run_long_test_period(full_contract, test_start_idx, sigma_tgt)
    long_metrics = _extract_trade_metrics(
        compute_metrics(long_rewards, n_contracts=1, round_output=False)
    )

    baseline_dist = _position_dist_pct(baseline_positions)

    return {
        "ticker": ticker,
        "round": round_num,
        "seed": seed,
        "occlusion_method": occlusion_method,
        "baseline": {
            "metrics": baseline_metrics,
            "position_dist": baseline_dist,
            "steps": len(baseline_rewards),
            "dqn_rewards": baseline_rewards,
            "dqn_positions": baseline_positions,
        },
        "long": long_metrics,
        "occlusion": occlusion_results,
        "train_time_s": t_train,
    }


def print_occlusion_table(result: dict) -> None:
    """Print formatted feature occlusion results."""
    ticker = result["ticker"]
    round_num = result["round"]
    seed = result["seed"]
    method = result["occlusion_method"]
    baseline = result["baseline"]["metrics"]
    baseline_dist = result["baseline"]["position_dist"]

    sep = "=" * 90
    print(f"\n{sep}")
    print(f"  Feature Occlusion: {ticker} r{round_num} | Seed: {seed} | Method: {method}")
    print(f"{sep}")

    print(f"\n  {'Feature':<22s}  ", end="")
    for name in TRADE_METRIC_NAMES:
        print(f"{name:>9s}  ", end="")
    print(f"{'L%':>5s}  {'F%':>5s}  {'S%':>5s}")
    print("  " + "-" * 110)

    # Baseline row
    b = baseline
    bd = baseline_dist
    print(f"  {'BASELINE (all feat)':<22s}  ", end="")
    for name in TRADE_METRIC_NAMES:
        print(f"{b[name]:>+9.4f}  ", end="")
    print(f"{bd['long']:>5.1f}  {bd['flat']:>5.1f}  {bd['short']:>5.1f}")

    # Long-only row
    lg = result["long"]
    print(f"  {'LONG-ONLY':<22s}  ", end="")
    for name in TRADE_METRIC_NAMES:
        print(f"{lg[name]:>+9.4f}  ", end="")
    print()

    # Occluded rows with delta
    for fname in [FEATURE_LABELS.get(i, f"F{i}") for i in range(FEATURE_DIM)]:
        occ = result["occlusion"][fname]
        om = occ["metrics"]
        od = occ["position_dist"]
        print(f"  {fname:<22s}  ", end="")
        for name in TRADE_METRIC_NAMES:
            print(f"{om[name]:>+9.4f}  ", end="")
        print(f"{od['long']:>5.1f}  {od['flat']:>5.1f}  {od['short']:>5.1f}")

    # Delta row (baseline - occluded)
    print(f"\n  {'DELTA (base - occluded)':<22s}  ", end="")
    print("  Impact: positive = removing feature hurts DQN (feature is useful)")
    print("  " + "-" * 110)
    print(f"  {'Feature':<22s}  ", end="")
    for name in TRADE_METRIC_NAMES:
        print(f"{name:>9s}  ", end="")
    print()

    for fname in [FEATURE_LABELS.get(i, f"F{i}") for i in range(FEATURE_DIM)]:
        occ = result["occlusion"][fname]
        om = occ["metrics"]
        print(f"  {fname:<22s}  ", end="")
        for name in TRADE_METRIC_NAMES:
            delta = baseline[name] - om[name]
            # For std(R) and DD, negative delta = occlusion increased risk = feature useful
            if name in ("std(R)", "DD"):
                useful = "v" if delta < 0 else ""
            else:
                useful = "^" if delta > 0 else ""
            print(f"{delta:>+9.4f}{useful:>1s} ", end="")
        print()

    # Summary: absolute impact ranking
    print(f"\n  Feature Impact Ranking (by absolute Sharpe change):")
    impacts = []
    for fname in [FEATURE_LABELS.get(i, f"F{i}") for i in range(FEATURE_DIM)]:
        delta_sharpe = baseline["Sharpe"] - result["occlusion"][fname]["metrics"]["Sharpe"]
        impacts.append((fname, delta_sharpe))
    impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    for rank, (fname, ds) in enumerate(impacts, 1):
        arrow = "CRITICAL" if abs(ds) > 0.05 else ("important" if abs(ds) > 0.01 else "marginal")
        print(f"    {rank}. {fname:<20s}  Delta-Sharpe={ds:+.4f}  [{arrow}]")

    print(f"{sep}\n")

    return impacts


def feature_occlusion_stitched(
    ticker: str = "AN",
    episodes: int = 100,
    device: str = "auto",
    sigma_tgt: float = SIGMA_TGT_DEFAULT,
    seed: int = 42,
    occlusion_method: str = "zero",
) -> dict:
    """Feature occlusion on stitched r1+r2."""
    r1_result = feature_occlusion_single(
        ticker=ticker, round_num=1, episodes=episodes,
        device=device, sigma_tgt=sigma_tgt, seed=seed,
        occlusion_method=occlusion_method,
    )
    r2_result = feature_occlusion_single(
        ticker=ticker, round_num=2, episodes=episodes,
        device=device, sigma_tgt=sigma_tgt, seed=seed,
        occlusion_method=occlusion_method,
    )

    stitched = {
        "ticker": ticker,
        "seed": seed,
        "occlusion_method": occlusion_method,
        "round": "stitched",
    }

    for label, res in [("r1", r1_result), ("r2", r2_result)]:
        baseline_r = np.concatenate([r1_result["baseline"]["dqn_rewards"], r2_result["baseline"]["dqn_rewards"]])
        stitched["baseline_metrics"] = _extract_trade_metrics(
            compute_metrics(baseline_r, n_contracts=1, round_output=False)
        )

    for fi in range(FEATURE_DIM):
        fname = FEATURE_LABELS.get(fi, f"F{fi}")
        occ_r1 = r1_result["occlusion"][fname]["dqn_rewards"]
        occ_r2 = r2_result["occlusion"][fname]["dqn_rewards"]
        occ_stitched = np.concatenate([occ_r1, occ_r2])
        stitched.setdefault("occlusion", {})[fname] = {
            "feature_idx": fi,
            "method": occlusion_method,
            "metrics": _extract_trade_metrics(
                compute_metrics(occ_stitched, n_contracts=1, round_output=False)
            ),
        }

    print(f"\n{'=' * 90}")
    print(f"  Feature Occlusion: {ticker} STITCHED r1+r2 | Seed: {seed}")
    print(f"{'=' * 90}")

    baseline = stitched["baseline_metrics"]
    print(f"\n  Baseline Sharpe: {baseline['Sharpe']:+.4f}")
    for fname in [FEATURE_LABELS.get(i, f"F{i}") for i in range(FEATURE_DIM)]:
        delta = baseline["Sharpe"] - stitched["occlusion"][fname]["metrics"]["Sharpe"]
        print(f"    {fname:<20s}  Delta-Sharpe={delta:+.4f}")

    return stitched


def feature_occlusion_multi_seed(
    ticker: str = "AN",
    round_num: int = 1,
    episodes: int = 100,
    device: str = "auto",
    sigma_tgt: float = SIGMA_TGT_DEFAULT,
    n_seeds: int = 5,
    occlusion_method: str = "zero",
) -> dict:
    """Feature occlusion across multiple seeds, reporting median +/- IQR."""
    seeds = LOCKED_SEEDS[:n_seeds]
    all_results = []
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        result = feature_occlusion_single(
            ticker=ticker, round_num=round_num, episodes=episodes,
            device=device, sigma_tgt=sigma_tgt, seed=seed,
            occlusion_method=occlusion_method,
        )
        all_results.append(result)

    print(f"\n{'=' * 90}")
    print(f"  {ticker} r{round_num} — Multi-seed occlusion summary ({n_seeds} seeds: {seeds})")
    print(f"{'=' * 90}")

    baseline_sharpes = [r["baseline"]["metrics"]["Sharpe"] for r in all_results]
    baseline_median = float(np.median(baseline_sharpes))
    baseline_iqr = float(np.percentile(baseline_sharpes, 75) - np.percentile(baseline_sharpes, 25))

    print(f"\n  Baseline DQN Sharpe: {baseline_median:+.4f} +/- {baseline_iqr:.4f}")
    print(f"\n  {'Feature':<22s}  {'Delta-Sharpe med':>16s}  {'Delta-Sharpe IQR':>16s}  {'Rank':>4s}")
    print("  " + "-" * 65)

    impacts = {}
    for fi in range(FEATURE_DIM):
        fname = FEATURE_LABELS.get(fi, f"F{fi}")
        deltas = []
        for r in all_results:
            delta = r["baseline"]["metrics"]["Sharpe"] - r["occlusion"][fname]["metrics"]["Sharpe"]
            deltas.append(delta)
        impacts[fname] = {
            "median": float(np.median(deltas)),
            "iqr": float(np.percentile(deltas, 75) - np.percentile(deltas, 25)),
            "abs_median": float(np.median(np.abs(deltas))),
        }

    ranked = sorted(impacts.items(), key=lambda x: x[1]["abs_median"], reverse=True)
    for rank, (fname, imp) in enumerate(ranked, 1):
        print(f"  {fname:<22s}  {imp['median']:>+16.4f}  {imp['iqr']:>16.4f}  {rank:>4d}")

    return {"seeds": seeds, "results": all_results, "impacts": impacts}


def feature_occlusion_asset_class(
    asset_name: str = "Forex",
    round_num: int = 1,
    episodes: int = 100,
    device: str = "auto",
    sigma_tgt: float = SIGMA_TGT_DEFAULT,
    seed: int = 42,
    occlusion_method: str = "zero",
) -> dict:
    """Train one shared DQN on the full asset class, then run feature occlusion
    on each contract and aggregate to portfolio level.

    Paper methodology: one shared model per asset class per round. For each
    feature, occlude it on every contract simultaneously, compute per-contract
    rewards, then equal-weight average to form the portfolio return series.
    """
    tickers = universe_tickers(asset_name)
    if not tickers:
        raise ValueError(f"No tickers found for asset class '{asset_name}'")

    print(f"\n{'=' * 90}")
    print(f"  ASSET-CLASS FEATURE OCCLUSION: {asset_name} ({len(tickers)} contracts) r{round_num}  |  Seed: {seed}")
    print(f"  Method: {occlusion_method}")
    print(f"{'=' * 90}")

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

    # 2. Collect per-contract data
    contract_data: dict[str, dict] = {}
    skipped: list[str] = []

    for ticker in tickers:
        try:
            full_contract, feature_meta = load_contract_round(ticker, round_num)
        except (FileNotFoundError, ValueError) as e:
            print(f"  SKIP {ticker}: {e}")
            skipped.append(ticker)
            continue

        test_start_idx = int(feature_meta["test_start_idx"])

        # Baseline
        baseline_rewards, baseline_positions = _run_dqn_test_period(
            agent, full_contract, test_start_idx, sigma_tgt
        )
        long_rewards = _run_long_test_period(full_contract, test_start_idx, sigma_tgt)

        contract_data[ticker] = {
            "contract": full_contract,
            "test_start_idx": test_start_idx,
            "baseline_rewards": baseline_rewards,
            "baseline_positions": baseline_positions,
            "long_rewards": long_rewards,
            "baseline_metrics": _extract_trade_metrics(
                compute_metrics(baseline_rewards, n_contracts=1, round_output=False)
            ),
            "long_metrics": _extract_trade_metrics(
                compute_metrics(long_rewards, n_contracts=1, round_output=False)
            ),
        }

    n_eval = len(contract_data)
    if n_eval == 0:
        raise RuntimeError(f"No contracts evaluated for {asset_name} r{round_num}")

    print(f"\n  Evaluated {n_eval}/{len(tickers)} contracts "
          f"({', '.join(contract_data.keys())})")
    if skipped:
        print(f"  Skipped: {', '.join(skipped)}")

    # 3. Per-feature occlusion (all contracts simultaneously)
    occlusion_results = {}

    for fi in range(FEATURE_DIM):
        fname = FEATURE_LABELS.get(fi, f"F{fi}")
        occ_portfolio_rewards_list: list[np.ndarray] = []
        baseline_portfolio_rewards_list: list[np.ndarray] = []

        for ticker, cd in contract_data.items():
            occluded_contract = _occlude_contract_features(
                cd["contract"], fi, method=occlusion_method
            )
            occ_rewards, _ = _run_dqn_test_period(
                agent, occluded_contract, cd["test_start_idx"], sigma_tgt
            )
            occ_portfolio_rewards_list.append(occ_rewards)
            baseline_portfolio_rewards_list.append(cd["baseline_rewards"])

        # Portfolio-level: equal-weight average (Eq.13)
        min_len = min(len(r) for r in baseline_portfolio_rewards_list)
        baseline_portfolio = np.mean([r[:min_len] for r in baseline_portfolio_rewards_list], axis=0)

        min_len_occ = min(len(r) for r in occ_portfolio_rewards_list)
        occ_portfolio = np.mean([r[:min_len_occ] for r in occ_portfolio_rewards_list], axis=0)

        baseline_port_metrics = _extract_trade_metrics(
            compute_metrics(baseline_portfolio, n_contracts=n_eval, round_output=False)
        )
        occ_port_metrics = _extract_trade_metrics(
            compute_metrics(occ_portfolio, n_contracts=n_eval, round_output=False)
        )

        # Per-contract delta
        per_contract_delta = {}
        for ticker, cd in contract_data.items():
            occluded_contract = _occlude_contract_features(
                cd["contract"], fi, method=occlusion_method
            )
            occ_r, _ = _run_dqn_test_period(
                agent, occluded_contract, cd["test_start_idx"], sigma_tgt
            )
            occ_m = _extract_trade_metrics(
                compute_metrics(occ_r, n_contracts=1, round_output=False)
            )
            per_contract_delta[ticker] = {
                name: cd["baseline_metrics"][name] - occ_m[name]
                for name in TRADE_METRIC_NAMES
            }

        occlusion_results[fname] = {
            "feature_idx": fi,
            "method": occlusion_method,
            "portfolio_metrics": occ_port_metrics,
            "per_contract_delta": per_contract_delta,
        }

    # Also compute baseline portfolio metrics
    baseline_portfolio_rewards_list = [cd["baseline_rewards"] for cd in contract_data.values()]
    min_len_b = min(len(r) for r in baseline_portfolio_rewards_list)
    baseline_portfolio = np.mean([r[:min_len_b] for r in baseline_portfolio_rewards_list], axis=0)
    baseline_port_metrics = _extract_trade_metrics(
        compute_metrics(baseline_portfolio, n_contracts=n_eval, round_output=False)
    )

    # 4. Print results
    print(f"\n  {'Feature':<22s}  ", end="")
    for name in TRADE_METRIC_NAMES:
        print(f"{'D_' + name[:6]:>8s}  ", end="")
    print(f"{'L%':>5s}  {'F%':>5s}  {'S%':>5s}")
    print("  " + "-" * 90)

    # Baseline portfolio row
    baseline_dist = _position_dist_pct(
        [p for cd in contract_data.values() for p in cd["baseline_positions"]]
    )
    print(f"  {'BASELINE PORTFOLIO':<22s}  ", end="")
    for name in TRADE_METRIC_NAMES:
        print(f"{baseline_port_metrics[name]:>+8.4f}  ", end="")
    print(f"{baseline_dist['long']:>5.1f}  {baseline_dist['flat']:>5.1f}  {baseline_dist['short']:>5.1f}")

    # Per-feature portfolio rows
    for fi in range(FEATURE_DIM):
        fname = FEATURE_LABELS.get(fi, f"F{fi}")
        om = occlusion_results[fname]["portfolio_metrics"]
        print(f"  {fname:<22s}  ", end="")
        for name in TRADE_METRIC_NAMES:
            print(f"{om[name]:>+8.4f}  ", end="")
        print()

    # Delta row
    print(f"\n  DELTA (baseline - occluded) at portfolio level:")
    print(f"  {'Feature':<22s}  ", end="")
    for name in TRADE_METRIC_NAMES:
        print(f"{name:>8s}  ", end="")
    print()
    print("  " + "-" * 90)

    for fi in range(FEATURE_DIM):
        fname = FEATURE_LABELS.get(fi, f"F{fi}")
        om = occlusion_results[fname]["portfolio_metrics"]
        print(f"  {fname:<22s}  ", end="")
        for name in TRADE_METRIC_NAMES:
            delta = baseline_port_metrics[name] - om[name]
            if name in ("std(R)", "DD"):
                tag = "v" if delta < 0 else " "
            else:
                tag = "^" if delta > 0 else " "
            print(f"{delta:>+8.4f}{tag} ", end="")
        print()

    # Impact ranking
    print(f"\n  Portfolio-level Feature Impact Ranking:")
    impacts = []
    for fi in range(FEATURE_DIM):
        fname = FEATURE_LABELS.get(fi, f"F{fi}")
        delta_sharpe = baseline_port_metrics["Sharpe"] - occlusion_results[fname]["portfolio_metrics"]["Sharpe"]
        impacts.append((fname, delta_sharpe))
    impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    for rank, (fname, ds) in enumerate(impacts, 1):
        label = "CRITICAL" if abs(ds) > 0.05 else ("important" if abs(ds) > 0.01 else "marginal")
        print(f"    {rank}. {fname:<20s}  Delta-Sharpe={ds:+.4f}  [{label}]")

    print(f"\n{'=' * 90}\n")

    return {
        "asset_name": asset_name,
        "round": round_num,
        "seed": seed,
        "occlusion_method": occlusion_method,
        "n_contracts": n_eval,
        "tickers": list(contract_data.keys()),
        "skipped": skipped,
        "baseline_portfolio_metrics": baseline_port_metrics,
        "occlusion": occlusion_results,
        "train_time_s": t_train_done - t_train_start,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DQN Feature Occlusion Analysis"
    )
    parser.add_argument("--ticker", default="AN", help="Contract ticker")
    parser.add_argument("--round", type=int, default=None, choices=[1, 2])
    parser.add_argument("--both", action="store_true")
    parser.add_argument("--stitch", action="store_true")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--sigma-tgt", type=float, default=SIGMA_TGT_DEFAULT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds (multi-seed mode)")
    parser.add_argument("--method", default="zero", choices=["zero", "mean"],
                        help="Occlusion method: 'zero' sets feature to 0, 'mean' replaces with dataset mean")
    parser.add_argument("--asset", default=None, help="Asset class for portfolio-level occlusion")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else LOCKED_SEEDS[0]
    rounds = [1, 2] if args.both or args.round is None else [args.round]

    # ── Asset-class portfolio occlusion ──
    if args.asset:
        for rn in rounds:
            feature_occlusion_asset_class(
                asset_name=args.asset,
                round_num=rn,
                episodes=args.episodes,
                device=args.device,
                sigma_tgt=args.sigma_tgt,
                seed=seed,
                occlusion_method=args.method,
            )
        return

    if args.stitch:
        feature_occlusion_stitched(
            ticker=args.ticker.upper(),
            episodes=args.episodes,
            device=args.device,
            sigma_tgt=args.sigma_tgt,
            seed=seed,
            occlusion_method=args.method,
        )
        return

    if args.seeds > 1:
        feature_occlusion_multi_seed(
            ticker=args.ticker.upper(),
            round_num=rounds[0],
            episodes=args.episodes,
            device=args.device,
            sigma_tgt=args.sigma_tgt,
            n_seeds=args.seeds,
            occlusion_method=args.method,
        )
        return

    for rn in rounds:
        result = feature_occlusion_single(
            ticker=args.ticker.upper(),
            round_num=rn,
            episodes=args.episodes,
            device=args.device,
            sigma_tgt=args.sigma_tgt,
            seed=seed,
            occlusion_method=args.method,
        )
        print_occlusion_table(result)


if __name__ == "__main__":
    main()