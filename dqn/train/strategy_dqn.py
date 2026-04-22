#!/usr/bin/env python3
"""Shared DQN runtime helpers and compatibility entrypoints."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import SOURCE_OVERRIDES
from data_loader import load_clc_full
from dqn.model import SharedDQNAgent
from dqn.pipeline import build_feature_matrix, compute_additive_returns, compute_ewma_sigma, get_feature_window
from dqn.spec import SHARED_ROUNDS, WARMUP, round_model_path, universe_slug, universe_tickers
from dqn.train.train_dqn_walkforward import train_round_shared


def load_shared_checkpoint(round_num: int, asset_name: str = "All") -> SharedDQNAgent:
    checkpoint = round_model_path(round_num, asset_name)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing shared DQN checkpoint: {checkpoint}")
    agent = SharedDQNAgent()
    agent.load(checkpoint)
    agent.q_net.eval()
    return agent


def infer_positions_from_history(round_num: int, asset_name: str, ticker: str, source: str | None = None) -> np.ndarray:
    source = source or SOURCE_OVERRIDES.get(ticker, "RAD")
    round_info = SHARED_ROUNDS[round_num]
    df = load_clc_full(
        ticker,
        source=source,
        start_date=round_info["train_start"],
        anchor_date=round_info["test_start"],
    )
    if df is None:
        return np.zeros(0, dtype=float)

    full_prices = df["Close"].to_numpy(dtype=float)
    returns = compute_additive_returns(full_prices)
    sigma = compute_ewma_sigma(returns)
    features = build_feature_matrix(full_prices, returns, sigma)
    agent = load_shared_checkpoint(round_num, asset_name)

    positions = np.zeros(len(full_prices), dtype=float)
    for idx in range(WARMUP, len(full_prices)):
        state = get_feature_window(features, idx)
        positions[idx] = float(agent.predict_action_id(state) - 1)
    return positions


def strategy_dqn_positions(ticker: str, round_num: int = 1, asset_name: str = "Forex", source: str | None = None) -> np.ndarray:
    """Compatibility inference entrypoint for a shared DQN checkpoint."""
    return infer_positions_from_history(round_num=round_num, asset_name=asset_name, ticker=ticker, source=source)


def status(asset_name: str = "All"):
    tickers = universe_tickers(asset_name)
    print(f"Shared DQN status — {asset_name} ({len(tickers)} contracts)")
    for round_num in sorted(SHARED_ROUNDS):
        checkpoint = round_model_path(round_num, asset_name)
        print(f"  {universe_slug(asset_name)} / r{round_num}: {'✅' if checkpoint.exists() else '❌'} {checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["train", "status"])
    parser.add_argument("--asset", default="Forex", help='Asset universe or "All"')
    parser.add_argument("--round", type=int, choices=sorted(SHARED_ROUNDS), default=1)
    parser.add_argument("--episodes", type=int, default=200)
    args = parser.parse_args()

    if args.cmd == "status":
        status(args.asset)
    elif args.cmd == "train":
        train_round_shared(args.asset, args.round, args.episodes)
