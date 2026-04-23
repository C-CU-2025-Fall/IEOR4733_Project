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
    ACTIVE_FEATURE_VERSION,
    RETRAIN_ROUNDS,
    feature_data_path,
    feature_spec,
    round_name,
    universe_tickers,
)
from drl_shared.state_space import build_contract_arrays


def _resolve_preset(preset_name: str | None) -> tuple[dict, set[str], str]:
    """Return (source_overrides, excluded_set, preset_label) for a named preset."""
    if preset_name is None or preset_name.lower() == "default":
        return dict(SOURCE_OVERRIDES), set(), "default"
    if preset_name.upper() == "STRUCTURAL_38":
        from frontier_presets import STRUCTURAL_38_OVERRIDES, STRUCTURAL_38_EXCLUDED
        return dict(STRUCTURAL_38_OVERRIDES), set(STRUCTURAL_38_EXCLUDED), "structural_38"
    raise ValueError(f"Unknown preset: {preset_name}. Available: default, structural_38")


def prepare_contract_round_features(
    ticker: str,
    round_num: int,
    model_version: str = ACTIVE_FEATURE_VERSION,
    source_overrides: dict | None = None,
) -> bool:
    ticker = ticker.upper()
    round_info = RETRAIN_ROUNDS[round_num]
    _overrides = source_overrides or SOURCE_OVERRIDES
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
        model_version=model_version,
    )

    spec = feature_spec(model_version)
    out_path = feature_data_path(round_num, ticker, model_version=model_version)
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
        model_version=model_version,
        state_spec_version=spec["state_spec_version"],
        feature_spec=json.dumps(spec, sort_keys=True),
        train_start=round_info["train_start"],
        train_end=round_info["train_end"],
        test_start=round_info["test_start"],
        test_end=round_info["test_end"],
        source_overrides=json.dumps(_overrides),
    )
    print(
        f"  {ticker}: prepared {model_version} features train={len(df_train)}d "
        f"({round_info['train_start']}~{round_info['train_end']}), source={source}, "
        f"state={spec['state_spec_version']}"
    )
    return True


def prepare_round_features(
    asset_name: str,
    round_num: int,
    model_version: str = ACTIVE_FEATURE_VERSION,
    excluded: set[str] | None = None,
    source_overrides: dict | None = None,
) -> tuple[int, int]:
    tickers = [t for t in universe_tickers(asset_name) if t.upper() not in (excluded or set())]
    ok = fail = 0
    print(f"\n{'=' * 70}")
    print(f"Shared DRL Feature Preparation — {asset_name} — {round_name(round_num)}")
    print(f"Train: {RETRAIN_ROUNDS[round_num]['train_start']} ~ {RETRAIN_ROUNDS[round_num]['train_end']}")
    print(f"Test : {RETRAIN_ROUNDS[round_num]['test_start']} ~ {RETRAIN_ROUNDS[round_num]['test_end']}")
    print(f"{'=' * 70}")
    for ticker in tqdm(tickers, desc=f"Features {asset_name} {round_name(round_num)}", unit="tk"):
        if prepare_contract_round_features(ticker, round_num, model_version=model_version, source_overrides=source_overrides):
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
    parser.add_argument("--model-version", default=ACTIVE_FEATURE_VERSION)
    parser.add_argument("--preset", default=None, help="Named preset: structural_38, default")
    args = parser.parse_args()

    overrides, excluded, preset_label = _resolve_preset(args.preset)
    if args.preset:
        print(f"Preset: {preset_label} | excluded={sorted(excluded)} | sigma_tgt varies by preset")

    rounds = sorted(RETRAIN_ROUNDS) if args.all_rounds or args.round is None else [args.round]
    for round_num in rounds:
        if args.ticker:
            prepare_contract_round_features(args.ticker.upper(), round_num, model_version=args.model_version, source_overrides=overrides)
        else:
            prepare_round_features(args.asset, round_num, model_version=args.model_version, excluded=excluded, source_overrides=overrides)


if __name__ == "__main__":
    main()
