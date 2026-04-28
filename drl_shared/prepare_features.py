#!/usr/bin/env python3
"""Global shared feature preparation for DRL models (DQN/PG/A2C)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import SOURCE_OVERRIDES
from config import ASSET_CLASSES
from data_loader import load_clc_full
from drl_shared.spec import (
    RETRAIN_ROUNDS,
    asset_index_path,
    current_source_policy,
    feature_data_path,
    feature_spec,
    round_name,
    universe_tickers,
)
from drl_shared.state_space import build_contract_arrays


def _derive_round_split_indices(dates: np.ndarray, round_info: dict[str, str]) -> dict[str, int]:
    dt = np.asarray(dates)
    train_start = np.datetime64(round_info["train_start"])
    train_end = np.datetime64(round_info["train_end"])
    test_start = np.datetime64(round_info["test_start"])
    test_end = np.datetime64(round_info["test_end"])

    train_start_idx_arr = np.where(dt >= train_start)[0]
    train_idx = np.where(dt <= train_end)[0]
    test_idx = np.where((dt >= test_start) & (dt <= test_end))[0]
    if len(train_start_idx_arr) == 0:
        raise ValueError(f"No rows found on/after train_start={round_info['train_start']}")
    if len(train_idx) == 0:
        raise ValueError(f"No rows found on/before train_end={round_info['train_end']}")
    if len(test_idx) == 0:
        raise ValueError(f"No rows found in test window [{round_info['test_start']}, {round_info['test_end']}]")

    train_start_idx = int(train_start_idx_arr[0])
    train_end_idx = int(train_idx[-1])
    test_start_idx = int(test_idx[0])
    test_end_idx = int(test_idx[-1])
    if train_start_idx > train_end_idx:
        raise ValueError(
            f"Invalid split: train_start_idx={train_start_idx} exceeds train_end_idx={train_end_idx}"
        )
    if train_end_idx >= test_start_idx:
        raise ValueError(
            f"Invalid split: train_end_idx={train_end_idx} overlaps test_start_idx={test_start_idx}"
        )
    return {
        "train_start_idx": train_start_idx,
        "train_end_idx": train_end_idx,
        "test_start_idx": test_start_idx,
        "test_end_idx": test_end_idx,
    }


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

    # Load burn-in data: 1 year before train_start to ensure 252+ trading days
    train_start_dt = datetime.strptime(round_info["train_start"], "%Y-%m-%d")
    burnin_start = (train_start_dt - timedelta(days=365)).strftime("%Y-%m-%d")

    df = load_clc_full(
        ticker,
        source=source,
        start_date=burnin_start,
        anchor_date=round_info["test_start"],
    )
    if df is None:
        print(f"  {ticker}: no data (source={source})")
        return False

    # Compute features on full data (burn-in + train + test)
    df_full = df.loc[df["Date"] <= round_info["test_end"]].reset_index(drop=True)

    spec = feature_spec()
    contract_full = build_contract_arrays(
        ticker=ticker,
        prices=df_full["Close"].to_numpy(dtype=float),
        dates=df_full["Date"].to_numpy(),
        source=source,
        feature_spec_override=spec,
    )

    # Drop burn-in period (first 252 trading days)
    BURNIN_DAYS = 252
    n_burnin = min(BURNIN_DAYS, len(contract_full.prices))
    contract = type(contract_full)(
        ticker=contract_full.ticker,
        prices=contract_full.prices[n_burnin:],
        returns=contract_full.returns[n_burnin:],
        sigma=contract_full.sigma[n_burnin:],
        features=contract_full.features[n_burnin:],
        dates=contract_full.dates[n_burnin:],
        source=contract_full.source,
    )
    split_meta = _derive_round_split_indices(contract.dates, round_info)

    if len(contract.prices) < 500:
        print(f"  {ticker}: train period too short after burn-in ({len(contract.prices)} days)")
        return False

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
        burnin_days=BURNIN_DAYS,
        train_start_idx=split_meta["train_start_idx"],
        train_end_idx=split_meta["train_end_idx"],
        test_start_idx=split_meta["test_start_idx"],
        test_end_idx=split_meta["test_end_idx"],
    )
    print(
        f"  {ticker}: prepared features total={len(contract.prices)}d "
        f"train_rows={split_meta['train_end_idx'] - split_meta['train_start_idx'] + 1} "
        f"test_rows={split_meta['test_end_idx'] - split_meta['test_start_idx'] + 1} "
        f"({round_info['train_start']}~{round_info['train_end']}, burnin={BURNIN_DAYS}d), source={source}, "
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
    prepared_tickers = []
    failed_tickers = []
    ok = fail = 0
    print(f"\n{'=' * 70}")
    print(f"Shared DRL Feature Preparation — {asset_name} — {round_name(round_num)}")
    print(f"Train: {RETRAIN_ROUNDS[round_num]['train_start']} ~ {RETRAIN_ROUNDS[round_num]['train_end']}")
    print(f"Test : {RETRAIN_ROUNDS[round_num]['test_start']} ~ {RETRAIN_ROUNDS[round_num]['test_end']}")
    print(f"{'=' * 70}")
    for ticker in tqdm(tickers, desc=f"Features {asset_name} {round_name(round_num)}", unit="tk"):
        if prepare_contract_round_features(ticker, round_num, source_overrides=source_overrides):
            ok += 1
            prepared_tickers.append(ticker)
        else:
            fail += 1
            failed_tickers.append(ticker)
    spec = feature_spec()
    index_path = asset_index_path(asset_name, round_num)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "asset_class": asset_name,
                "round": round_name(round_num),
                "round_num": round_num,
                "member_tickers": prepared_tickers,
                "failed_tickers": failed_tickers,
                "expected_tickers": tickers,
                "excluded_contracts": sorted(excluded_set),
                "source_overrides": source_overrides or policy["source_overrides"] or SOURCE_OVERRIDES,
                "preset": policy["preset"],
                "train_start": RETRAIN_ROUNDS[round_num]["train_start"],
                "train_end": RETRAIN_ROUNDS[round_num]["train_end"],
                "test_start": RETRAIN_ROUNDS[round_num]["test_start"],
                "test_end": RETRAIN_ROUNDS[round_num]["test_end"],
                "feature_spec": spec,
                "state_spec_version": spec["state_spec_version"],
                "expected_count": len(tickers),
                "loaded_count": ok,
                "failed_count": fail,
            },
            fh,
            indent=2,
            sort_keys=True,
        )
    print(f"Asset index: {index_path}")
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
        elif args.asset == "All":
            for asset_name in ASSET_CLASSES:
                prepare_round_features(asset_name, round_num, excluded=excluded, source_overrides=overrides)
        else:
            prepare_round_features(args.asset, round_num, excluded=excluded, source_overrides=overrides)


if __name__ == "__main__":
    main()
