#!/usr/bin/env python3
"""Verification script for the shared-model DQN pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline_run import load_contracts
from config import SOURCE_OVERRIDES
from data_loader import load_clc_full
from dqn.pipeline import (
    build_feature_matrix,
    compute_additive_returns,
    compute_ewma_sigma,
    get_feature_window,
)
from dqn.spec import FEATURE_DIM, SEQ_LEN, SHARED_ROUNDS, WARMUP, round_data_dir, round_model_path


def _load_train_frame(ticker: str, round_num: int):
    round_info = SHARED_ROUNDS[round_num]
    source = SOURCE_OVERRIDES.get(ticker, "RAD")
    df = load_clc_full(
        ticker,
        source=source,
        start_date=round_info["train_start"],
        anchor_date=round_info["test_start"],
    )
    if df is None:
        raise RuntimeError(f"No source data for {ticker} (source={source})")
    train = df[(df["Date"] >= round_info["train_start"]) & (df["Date"] <= round_info["train_end"])].reset_index(drop=True)
    if len(train) < 500:
        raise RuntimeError(f"Training slice too short for {ticker}: {len(train)}")
    return train, source


def verify_state_schema(ticker: str, round_num: int) -> dict:
    train, source = _load_train_frame(ticker, round_num)
    prices = train["Close"].to_numpy(dtype=float)
    returns = compute_additive_returns(prices)
    sigma = compute_ewma_sigma(returns)
    features = build_feature_matrix(prices, returns, sigma)
    window = get_feature_window(features, WARMUP + 8)

    if features.shape[1] != FEATURE_DIM:
        raise AssertionError(f"Feature dim mismatch: {features.shape[1]} vs {FEATURE_DIM}")
    if window.shape != (SEQ_LEN, FEATURE_DIM):
        raise AssertionError(f"Window shape mismatch: {window.shape} vs {(SEQ_LEN, FEATURE_DIM)}")
    if not np.isfinite(features).all():
        raise AssertionError("Feature matrix contains non-finite values")

    return {
        "ticker": ticker,
        "source": source,
        "train_days": len(train),
        "feature_shape": tuple(features.shape),
        "window_shape": tuple(window.shape),
        "window_checksum": float(window.sum()),
    }


def verify_prepared_round_data(ticker: str, asset_name: str, round_num: int) -> dict:
    prepared_path = round_data_dir(round_num, asset_name) / f"{ticker}.npz"
    if not prepared_path.exists():
        raise FileNotFoundError(f"Prepared state file missing: {prepared_path}")

    train, source = _load_train_frame(ticker, round_num)
    prices = train["Close"].to_numpy(dtype=float)
    returns = compute_additive_returns(prices)
    sigma = compute_ewma_sigma(returns)
    features = build_feature_matrix(prices, returns, sigma)

    data = np.load(prepared_path, allow_pickle=True)
    checks = {
        "prices": np.allclose(data["prices"], prices),
        "returns": np.allclose(data["returns"], returns),
        "sigma": np.allclose(data["sigma"], sigma, equal_nan=True),
        "features": np.allclose(data["features"], features),
        "source": str(data["source"]) == source,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"Prepared round data mismatch for {ticker}: {failed}")

    return {
        "prepared_path": str(prepared_path),
        "checks": checks,
        "rows": len(prices),
    }


def verify_checkpoint_presence(asset_name: str, round_num: int) -> dict:
    checkpoint = round_model_path(round_num, asset_name)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Shared checkpoint missing: {checkpoint}. "
            "This is expected before training, but backtest cannot run yet."
        )
    return {"checkpoint": str(checkpoint)}


def verify_backtest_inputs(asset_name: str, round_num: int) -> dict:
    round_info = SHARED_ROUNDS[round_num]
    raw = load_contracts(asset_name, test_start=round_info["test_start"], test_end=round_info["test_end"])
    if not raw:
        raise AssertionError(f"No backtest contracts loaded for {asset_name}")
    missing = []
    for rd in raw:
        if not {"tk", "prices", "rt", "sigma", "start", "t1", "dates"}.issubset(rd.keys()):
            missing.append(rd.get("tk", "<unknown>"))
    if missing:
        raise AssertionError(f"Backtest raw contract payload missing keys: {missing}")
    return {"contracts_loaded": len(raw), "tickers": [rd["tk"] for rd in raw]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="Forex")
    parser.add_argument("--round", type=int, default=1, choices=sorted(SHARED_ROUNDS))
    parser.add_argument("--ticker", default="AN")
    parser.add_argument("--require-prepared", action="store_true")
    parser.add_argument("--require-checkpoint", action="store_true")
    args = parser.parse_args()

    print(f"Verify shared DQN — asset={args.asset} round=r{args.round} ticker={args.ticker}")

    state = verify_state_schema(args.ticker, args.round)
    print(f"[OK] state schema: train_days={state['train_days']} feature_shape={state['feature_shape']} "
          f"window_shape={state['window_shape']} checksum={state['window_checksum']:.6f}")

    backtest = verify_backtest_inputs(args.asset, args.round)
    print(f"[OK] backtest inputs: contracts_loaded={backtest['contracts_loaded']}")

    if args.require_prepared:
        prepared = verify_prepared_round_data(args.ticker, args.asset, args.round)
        print(f"[OK] prepared round data: {prepared['prepared_path']} rows={prepared['rows']}")
    else:
        prepared_path = round_data_dir(args.round, args.asset) / f"{args.ticker}.npz"
        print(f"[INFO] prepared round data not required. expected path: {prepared_path}")

    if args.require_checkpoint:
        checkpoint = verify_checkpoint_presence(args.asset, args.round)
        print(f"[OK] shared checkpoint: {checkpoint['checkpoint']}")
    else:
        checkpoint = round_model_path(args.round, args.asset)
        print(f"[INFO] checkpoint not required. expected path: {checkpoint}")


if __name__ == "__main__":
    main()
