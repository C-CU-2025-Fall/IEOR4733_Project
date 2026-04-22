#!/usr/bin/env python3
"""Prepare shared-model DQN training data by round and universe."""
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
from dqn.pipeline import ContractArrays, build_feature_matrix, compute_additive_returns, compute_ewma_sigma
from dqn.spec import SHARED_ROUNDS, round_data_dir, round_name, universe_tickers


def prepare_contract_round_data(ticker: str, round_num: int, asset_name: str) -> bool:
    round_info = SHARED_ROUNDS[round_num]
    source = SOURCE_OVERRIDES.get(ticker, "RAD")
    df = load_clc_full(
        ticker,
        source=source,
        start_date=round_info["train_start"],
        anchor_date=round_info["test_start"],
    )
    if df is None:
        print(f"  {ticker}: ❌ no data (source={source})")
        return False

    train_mask = (df["Date"] >= round_info["train_start"]) & (df["Date"] <= round_info["train_end"])
    df_train = df.loc[train_mask].reset_index(drop=True)
    if len(df_train) < 500:
        print(f"  {ticker}: ❌ train period too short ({len(df_train)} days)")
        return False

    prices = df_train["Close"].to_numpy(dtype=float)
    returns = compute_additive_returns(prices)
    sigma = compute_ewma_sigma(returns)
    features = build_feature_matrix(prices, returns, sigma)
    contract = ContractArrays(
        ticker=ticker,
        prices=prices,
        returns=returns,
        sigma=sigma,
        features=features,
        dates=df_train["Date"].to_numpy(),
        source=source,
    )

    out_dir = round_data_dir(round_num, asset_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ticker}.npz"
    np.savez_compressed(
        out_path,
        ticker=contract.ticker,
        prices=contract.prices,
        returns=contract.returns,
        sigma=contract.sigma,
        features=contract.features,
        dates=contract.dates,
        source=contract.source,
        round=round_name(round_num),
        train_start=round_info["train_start"],
        train_end=round_info["train_end"],
        test_start=round_info["test_start"],
        test_end=round_info["test_end"],
    )
    print(
        f"  {ticker}: ✅ train={len(df_train)}d "
        f"({round_info['train_start']}~{round_info['train_end']}), "
        f"source={source}"
    )
    return True


def prepare_round(asset_name: str, round_num: int) -> tuple[int, int]:
    tickers = universe_tickers(asset_name)
    ok = fail = 0
    print(f"\n{'=' * 70}")
    print(f"Shared DQN Data Preparation — {asset_name} — {round_name(round_num)}")
    print(f"Train: {SHARED_ROUNDS[round_num]['train_start']} ~ {SHARED_ROUNDS[round_num]['train_end']}")
    print(f"Test : {SHARED_ROUNDS[round_num]['test_start']} ~ {SHARED_ROUNDS[round_num]['test_end']}")
    print(f"{'=' * 70}")
    for ticker in tickers:
        if prepare_contract_round_data(ticker, round_num, asset_name):
            ok += 1
        else:
            fail += 1
    print(f"\nOutput: {round_data_dir(round_num, asset_name)}/")
    print(f"Prepared: {ok}/{len(tickers)}")
    return ok, fail


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="Forex", help='Asset universe or "All"')
    parser.add_argument("--round", type=int, choices=sorted(SHARED_ROUNDS), default=None)
    parser.add_argument("--all-rounds", action="store_true")
    args = parser.parse_args()

    rounds = sorted(SHARED_ROUNDS) if args.all_rounds or args.round is None else [args.round]
    for round_num in rounds:
        prepare_round(args.asset, round_num)
