#!/usr/bin/env python3
"""Global shared feature preparation for DRL models (DQN/PG/A2C)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import SOURCE_OVERRIDES
from data_loader import load_clc_full
from drl_shared.spec import (
    RETRAIN_ROUNDS,
    current_source_policy,
    feature_data_path,
    feature_spec,
    round_name,
    universe_tickers,
)
from drl_shared.state_space import build_contract_arrays


def prepare_contract_round_features(
    ticker: str,
    round_num: int,
    source_overrides: dict | None = None,
) -> bool:
    ticker = ticker.upper()
    round_info = RETRAIN_ROUNDS[round_num]
    policy = current_source_policy()
    _overrides = source_overrides or policy["source_overrides"] or SOURCE_OVERRIDES
    source = _overrides.get(ticker, "RAD")
    df = load_clc_full(
        ticker,
        source=source,
        start_date=round_info["train_start"],
        anchor_date=round_info["test_start"],
    )
    if df is None:
        print(f"  {ticker}: no data (source={source})")
        return False

    train_mask = (df["Date"] >= round_info["train_start"]) & (df["Date"] <= round_info["train_end"])
    df_train = df.loc[train_mask].reset_index(drop=True)
    if len(df_train) < 500:
        print(f"  {ticker}: train period too short ({len(df_train)} days)")
        return False

    contract = build_contract_arrays(
        ticker=ticker,
        prices=df_train["Close"].to_numpy(dtype=float),
        dates=df_train["Date"].to_numpy(),
        source=source,
    )

    spec = feature_spec()
    out_path = feature_data_path(round_num, ticker)
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
        feature_line=spec["feature_line"],
        state_spec_version=spec["state_spec_version"],
        feature_spec=json.dumps(spec, sort_keys=True),
        preset=policy["preset"],
        excluded_contracts=json.dumps(policy["excluded_contracts"]),
        train_start=round_info["train_start"],
        train_end=round_info["train_end"],
        test_start=round_info["test_start"],
        test_end=round_info["test_end"],
        source_overrides=json.dumps(_overrides),
    )
    print(
        f"  {ticker}: prepared mainline features train={len(df_train)}d "
        f"({round_info['train_start']}~{round_info['train_end']}), source={source}, "
        f"state={spec['state_spec_version']}"
    )
    return True


def prepare_round_features(
    asset_name: str,
    round_num: int,
    excluded: set[str] | None = None,
    source_overrides: dict | None = None,
) -> tuple[int, int]:
    policy = current_source_policy()
    excluded_set = excluded if excluded is not None else set(policy["excluded_contracts"])
    tickers = [t for t in universe_tickers(asset_name) if t.upper() not in excluded_set]
    ok = fail = 0
    print(f"\n{'=' * 70}")
    print(f"Shared DRL Feature Preparation — {asset_name} — {round_name(round_num)}")
    print(f"Train: {RETRAIN_ROUNDS[round_num]['train_start']} ~ {RETRAIN_ROUNDS[round_num]['train_end']}")
    print(f"Test : {RETRAIN_ROUNDS[round_num]['test_start']} ~ {RETRAIN_ROUNDS[round_num]['test_end']}")
    print(f"{'=' * 70}")
    for ticker in tqdm(tickers, desc=f"Features {asset_name} {round_name(round_num)}", unit="tk"):
        if prepare_contract_round_features(ticker, round_num, source_overrides=source_overrides):
            ok += 1
        else:
            fail += 1
    print(f"Prepared features: {ok}/{len(tickers)}")
    return ok, fail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="Forex", help='Asset universe or "All"')
    parser.add_argument("--ticker", default=None, help="Optional single ticker override")
    parser.add_argument("--round", type=int, choices=sorted(RETRAIN_ROUNDS), default=None)
    parser.add_argument("--all-rounds", action="store_true")
    args = parser.parse_args()

    policy = current_source_policy()
    overrides = policy["source_overrides"] or SOURCE_OVERRIDES
    excluded = set(policy["excluded_contracts"])
    print(f"Preset: {policy['preset']} | excluded={sorted(excluded)} | sigma_tgt=0.058")

    rounds = sorted(RETRAIN_ROUNDS) if args.all_rounds or args.round is None else [args.round]
    for round_num in rounds:
        if args.ticker:
            prepare_contract_round_features(args.ticker.upper(), round_num, source_overrides=overrides)
        else:
            prepare_round_features(args.asset, round_num, excluded=excluded, source_overrides=overrides)


if __name__ == "__main__":
    main()
