#!/usr/bin/env python3
"""Verification script for the single-contract DQN pipeline."""
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
from drl.dqn.backtest.engine import portfolio_metrics
from drl.dqn.spec import ACTIVE_MODEL_VERSION, RETRAIN_ROUNDS, resolve_checkpoint_path
from drl_shared.spec import FEATURE_DIM, SEQ_LEN, WARMUP, feature_data_path
from drl_shared.state_space import build_contract_arrays, get_feature_window


def _load_train_frame(ticker: str, round_num: int):
    round_info = RETRAIN_ROUNDS[round_num]
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


def verify_state_schema(ticker: str, round_num: int, model_version: str = ACTIVE_MODEL_VERSION) -> dict:
    train, source = _load_train_frame(ticker, round_num)
    contract = build_contract_arrays(
        ticker=ticker,
        prices=train["Close"].to_numpy(dtype=float),
        dates=train["Date"].to_numpy(),
        source=source,
        model_version=model_version,
    )
    window = get_feature_window(contract.features, WARMUP + 8)

    if contract.features.shape[1] != FEATURE_DIM:
        raise AssertionError(f"Feature dim mismatch: {contract.features.shape[1]} vs {FEATURE_DIM}")
    if window.shape != (SEQ_LEN, FEATURE_DIM):
        raise AssertionError(f"Window shape mismatch: {window.shape} vs {(SEQ_LEN, FEATURE_DIM)}")
    if not np.isfinite(contract.features).all():
        raise AssertionError("Feature matrix contains non-finite values")

    return {
        "ticker": ticker,
        "source": source,
        "train_days": len(train),
        "feature_shape": tuple(contract.features.shape),
        "window_shape": tuple(window.shape),
        "window_checksum": float(window.sum()),
    }


def _npz_scalar(data, key: str, default=None):
    if key not in data:
        return default
    value = data[key]
    if getattr(value, "shape", None) == ():
        return value.item()
    return value


def verify_prepared_round_data(ticker: str, round_num: int, model_version: str = ACTIVE_MODEL_VERSION) -> dict:
    prepared_path = feature_data_path(round_num, ticker, model_version=model_version)
    if not prepared_path.exists():
        raise FileNotFoundError(f"Prepared state file missing: {prepared_path}")

    train, source = _load_train_frame(ticker, round_num)
    contract = build_contract_arrays(
        ticker=ticker,
        prices=train["Close"].to_numpy(dtype=float),
        dates=train["Date"].to_numpy(),
        source=source,
        model_version=model_version,
    )

    data = np.load(prepared_path, allow_pickle=True)
    checks = {
        "prices": np.allclose(data["prices"], contract.prices),
        "returns": np.allclose(data["returns"], contract.returns),
        "sigma": np.allclose(data["sigma"], contract.sigma, equal_nan=True),
        "features": np.allclose(data["features"], contract.features),
        "source": str(_npz_scalar(data, "source", "")) == source,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"Prepared round data mismatch for {ticker}: {failed}")

    return {
        "prepared_path": str(prepared_path),
        "checks": checks,
        "rows": len(contract.prices),
    }


def verify_checkpoint_presence(ticker: str, round_num: int, model_version: str = ACTIVE_MODEL_VERSION) -> dict:
    checkpoint = resolve_checkpoint_path(round_num, ticker, model_version=model_version)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Contract checkpoint missing: {checkpoint}. "
            "This is expected before training, but DQN backtest cannot run yet."
        )
    return {"checkpoint": str(checkpoint)}


def verify_backtest_long(asset_name: str) -> dict:
    metrics = portfolio_metrics(asset_name, "Long")
    for key in ("MDD", "Calmar"):
        if key not in metrics:
            raise AssertionError(f"Backtest metrics missing {key}")
    return {"asset": asset_name, "metrics": metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="Forex")
    parser.add_argument("--round", type=int, default=1, choices=sorted(RETRAIN_ROUNDS))
    parser.add_argument("--ticker", default="AN")
    parser.add_argument("--model-version", default=ACTIVE_MODEL_VERSION)
    parser.add_argument("--require-prepared", action="store_true")
    parser.add_argument("--require-checkpoint", action="store_true")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    print(f"Verify DQN pipeline — asset={args.asset} round=r{args.round} ticker={ticker}")

    state = verify_state_schema(ticker, args.round, model_version=args.model_version)
    print(
        f"[OK] state schema: train_days={state['train_days']} "
        f"feature_shape={state['feature_shape']} window_shape={state['window_shape']} "
        f"checksum={state['window_checksum']:.6f}"
    )

    long_metrics = verify_backtest_long(args.asset)
    print(
        f"[OK] backtest long: MDD={long_metrics['metrics']['MDD']:+.3f} "
        f"Calmar={long_metrics['metrics']['Calmar']:+.3f}"
    )

    if args.require_prepared:
        prepared = verify_prepared_round_data(ticker, args.round, model_version=args.model_version)
        print(f"[OK] prepared round data: {prepared['prepared_path']} rows={prepared['rows']}")
    else:
        prepared_path = feature_data_path(args.round, ticker, model_version=args.model_version)
        print(f"[INFO] prepared round data not required. expected path: {prepared_path}")

    if args.require_checkpoint:
        checkpoint = verify_checkpoint_presence(ticker, args.round, model_version=args.model_version)
        print(f"[OK] contract checkpoint: {checkpoint['checkpoint']}")
    else:
        checkpoint = resolve_checkpoint_path(args.round, ticker, model_version=args.model_version)
        print(f"[INFO] checkpoint not required. expected path: {checkpoint}")


if __name__ == "__main__":
    main()
