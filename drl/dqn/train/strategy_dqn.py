#!/usr/bin/env python3
"""Deprecated single-contract DQN runtime helpers and compatibility entrypoints."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import SOURCE_OVERRIDES
from data_loader import load_clc_full
from drl.dqn.model import DQNAgent
from drl.dqn.spec import ACTIVE_MODEL_VERSION, RETRAIN_ROUNDS, WARMUP, resolve_checkpoint_path, ticker_asset_class, universe_tickers
from drl.dqn.train.train_dqn_walkforward import train_contract_round
from drl_shared.state_space import build_contract_arrays, get_feature_window


def load_contract_checkpoint(round_num: int, ticker: str, model_version: str = ACTIVE_MODEL_VERSION) -> DQNAgent:
    ticker = ticker.upper()
    checkpoint = resolve_checkpoint_path(round_num, ticker, model_version=model_version)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing DQN checkpoint: {checkpoint}")
    agent = DQNAgent()
    agent.load(checkpoint)
    agent.q_net.eval()
    return agent


def infer_positions_from_history(round_num: int, ticker: str, source: str | None = None, model_version: str = ACTIVE_MODEL_VERSION) -> np.ndarray:
    ticker = ticker.upper()
    source = source or SOURCE_OVERRIDES.get(ticker, "RAD")
    round_info = RETRAIN_ROUNDS[round_num]
    df = load_clc_full(
        ticker,
        source=source,
        start_date=round_info["train_start"],
        anchor_date=round_info["test_start"],
    )
    if df is None:
        return np.zeros(0, dtype=float)

    contract = build_contract_arrays(
        ticker=ticker,
        prices=df["Close"].to_numpy(dtype=float),
        dates=df["Date"].to_numpy(),
        source=source,
        model_version=model_version,
    )
    agent = load_contract_checkpoint(round_num, ticker, model_version=model_version)

    positions = np.zeros(len(contract.prices), dtype=float)
    for idx in range(WARMUP, len(contract.prices)):
        state = get_feature_window(contract.features, idx)
        positions[idx] = float(agent.predict_action_id(state) - 1)
    return positions


def strategy_dqn_positions(ticker: str, round_num: int = 1, source: str | None = None, model_version: str = ACTIVE_MODEL_VERSION) -> np.ndarray:
    """Compatibility inference entrypoint for a single-contract DQN checkpoint."""
    return infer_positions_from_history(round_num=round_num, ticker=ticker, source=source, model_version=model_version)


def status(asset_name: str = "All", ticker: str | None = None, model_version: str = ACTIVE_MODEL_VERSION):
    tickers = [ticker.upper()] if ticker else universe_tickers(asset_name)
    print(f"DQN status — {asset_name if not ticker else ticker} ({len(tickers)} contracts)")
    for tk in tickers:
        print(f"  {tk} [{ticker_asset_class(tk)}]")
        for round_num in sorted(RETRAIN_ROUNDS):
            checkpoint = resolve_checkpoint_path(round_num, tk, model_version=model_version)
            print(f"    r{round_num}: {'Y' if checkpoint.exists() else 'N'} {checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["train", "status"])
    parser.add_argument("--asset", default="Forex", help='Asset universe or "All"')
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--round", type=int, choices=sorted(RETRAIN_ROUNDS), default=1)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--model-version", default=ACTIVE_MODEL_VERSION)
    args = parser.parse_args()

    if args.cmd == "status":
        status(args.asset, ticker=args.ticker, model_version=args.model_version)
    elif args.cmd == "train":
        if not args.ticker:
            raise ValueError("--ticker is required for single-contract training")
        train_contract_round(args.ticker, args.round, args.episodes, model_version=args.model_version)
