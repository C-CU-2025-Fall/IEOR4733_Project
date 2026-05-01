#!/usr/bin/env python3
"""Unified feature data loader for DQN training and backtest.

Single source of truth for loading .npz feature artifacts:
- TRAINING: slice to train period, split 90/10 for train/val
- BACKTEST: slice to test period

All date boundaries come from RETRAIN_ROUNDS in spec.
Raises on mismatch — no silent data corruption.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from drl_shared.spec import (
    FEATURE_DIM,
    MARKET_FEATURE_DIM,
    RETRAIN_ROUNDS,
    WARMUP,
    feature_spec,
)
from drl_shared.state_space import ContractArrays


def _npz_scalar(data, key: str, default=None):
    if key not in data:
        return default
    value = data[key]
    if getattr(value, "shape", None) == ():
        return value.item()
    return value


def _derive_round_indices(
    dates: np.ndarray,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> dict[str, int]:
    dt = pd.to_datetime(dates)
    train_start_ts = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)
    train_start_idx_arr = np.where(dt >= train_start_ts)[0]
    train_idx = np.where(dt <= train_end_ts)[0]
    test_idx = np.where((dt >= test_start_ts) & (dt <= test_end_ts))[0]
    assert len(train_start_idx_arr) > 0, f"no train rows on/after {train_start}"
    assert len(train_idx) > 0, f"no train rows on/before {train_end}"
    assert len(test_idx) > 0, f"no test rows in [{test_start}, {test_end}]"
    return {
        "train_start_idx": int(train_start_idx_arr[0]),
        "train_end_idx": int(train_idx[-1]),
        "test_start_idx": int(test_idx[0]),
        "test_end_idx": int(test_idx[-1]),
    }


class DataSlice:
    """A contiguous slice of a ContractArrays, with validation metadata."""
    __slots__ = ("contract", "start_idx", "end_idx", "label")

    def __init__(self, contract: ContractArrays, start_idx: int, end_idx: int, label: str):
        self.contract = contract
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.label = label

    @property
    def usable_steps(self) -> int:
        return max(0, self.end_idx - self.start_idx)

    def __repr__(self):
        return f"DataSlice({self.label}, steps={self.usable_steps})"


def load_npz(ticker: str, round_num: int) -> tuple[np.lib.npyio.NpzFile, ContractArrays, dict]:
    """Load raw npz, return (data, full_contract, meta).
    
    Validates: file exists, feature dim matches, state spec matches,
    dates are monotonically increasing, no NaN/Inf, no duplicate dates,
    train/test boundaries exist and fall within data range.
    """
    from drl.dqn.spec import contract_data_path
    path = contract_data_path(round_num, ticker)
    if not path.exists():
        raise FileNotFoundError(f"No features for {ticker} r{round_num}: {path}")

    data = np.load(path, allow_pickle=True)

    # Build meta
    expected = feature_spec()
    actual_state = _npz_scalar(data, "state_spec_version")
    if actual_state and actual_state != expected["state_spec_version"]:
        raise ValueError(
            f"Feature spec mismatch for {ticker} r{round_num}: "
            f"npz={actual_state!r} != expected={expected['state_spec_version']!r}"
        )

    meta = {
        "path": str(path),
        "state_spec_version": actual_state,
        "train_start": str(_npz_scalar(data, "train_start", "")),
        "train_end": str(_npz_scalar(data, "train_end", "")),
        "test_start": str(_npz_scalar(data, "test_start", "")),
        "test_end": str(_npz_scalar(data, "test_end", "")),
        "train_start_idx": _npz_scalar(data, "train_start_idx", None),
        "train_end_idx": _npz_scalar(data, "train_end_idx", None),
        "test_start_idx": _npz_scalar(data, "test_start_idx", None),
        "test_end_idx": _npz_scalar(data, "test_end_idx", None),
    }

    # Build full contract
    contract = ContractArrays(
        ticker=ticker,
        prices=data["prices"],
        returns=data["returns"],
        sigma=data["sigma"],
        features=data["features"],
        dates=data["dates"],
        source=str(_npz_scalar(data, "source", "")),
    )

    # --- Validation ---
    n = len(contract.prices)
    assert n > 0, f"{ticker} r{round_num}: 0 rows"

    # Feature dim
    assert contract.features.shape == (n, MARKET_FEATURE_DIM), (
        f"{ticker} r{round_num}: features shape {contract.features.shape} != ({n}, {MARKET_FEATURE_DIM})"
    )

    # No NaN/Inf in critical arrays
    for attr in ("prices", "returns", "sigma"):
        arr = getattr(contract, attr)
        assert not np.any(np.isnan(arr)), f"{ticker} r{round_num}: {attr} has NaN"
        assert not np.any(np.isinf(arr)), f"{ticker} r{round_num}: {attr} has Inf"

    assert not np.any(np.isnan(contract.features)), f"{ticker} r{round_num}: features has NaN"
    assert not np.any(np.isinf(contract.features)), f"{ticker} r{round_num}: features has Inf"

    # No duplicate dates
    dates = pd.to_datetime(contract.dates)
    n_unique = len(dates.drop_duplicates())
    assert n_unique == n, f"{ticker} r{round_num}: {n - n_unique} duplicate dates"

    # Monotonic dates
    assert (dates.diff()[1:] >= pd.Timedelta(0)).all(), (
        f"{ticker} r{round_num}: dates not monotonically increasing"
    )

    # Train/test boundaries present
    for key in ("train_start", "train_end", "test_start", "test_end"):
        assert meta[key], f"{ticker} r{round_num}: missing {key} in npz"

    if any(meta[key] is None for key in ("train_start_idx", "train_end_idx", "test_start_idx", "test_end_idx")):
        meta.update(
            _derive_round_indices(
                contract.dates,
                meta["train_start"],
                meta["train_end"],
                meta["test_start"],
                meta["test_end"],
            )
        )
    else:
        for key in ("train_start_idx", "train_end_idx", "test_start_idx", "test_end_idx"):
            meta[key] = int(meta[key])

    # Train/test boundaries within data range
    first_date, last_date = dates[0], dates[-1]
    train_start = pd.Timestamp(meta["train_start"])
    train_end = pd.Timestamp(meta["train_end"])
    test_start = pd.Timestamp(meta["test_start"])
    test_end = pd.Timestamp(meta["test_end"])

    # Data should start near train_start (allow trading calendar offset)
    assert first_date <= train_start + pd.Timedelta(days=7), (
        f"{ticker} r{round_num}: data starts {first_date} too far after train_start {train_start}"
    )
    assert train_end < test_start, (
        f"{ticker} r{round_num}: train_end {train_end} >= test_start {test_start}"
    )
    assert test_end <= last_date + pd.Timedelta(days=7), (
        f"{ticker} r{round_num}: data ends {last_date} before test_end {test_end}"
    )
    assert 0 <= meta["train_start_idx"] <= meta["train_end_idx"], (
        f"{ticker} r{round_num}: bad train_start_idx={meta['train_start_idx']}"
    )
    assert 0 <= meta["train_end_idx"] < n, f"{ticker} r{round_num}: bad train_end_idx={meta['train_end_idx']}"
    assert 0 <= meta["test_start_idx"] < n, f"{ticker} r{round_num}: bad test_start_idx={meta['test_start_idx']}"
    assert 0 <= meta["test_end_idx"] < n, f"{ticker} r{round_num}: bad test_end_idx={meta['test_end_idx']}"
    assert meta["train_end_idx"] < meta["test_start_idx"], (
        f"{ticker} r{round_num}: split overlap train_end_idx={meta['train_end_idx']} "
        f"test_start_idx={meta['test_start_idx']}"
    )

    return data, contract, meta


def _slice_by_date(contract: ContractArrays, start_date: str, end_date: str, label: str) -> DataSlice:
    """Slice contract by date range, return DataSlice with WARMUP offset."""
    dates = pd.to_datetime(contract.dates)
    ts = pd.Timestamp(start_date)
    te = pd.Timestamp(end_date)

    mask = (dates >= ts) & (dates <= te)
    n_matching = mask.sum()
    assert n_matching > 0, f"{label}: no dates in range [{start_date}, {end_date}]"

    # Get the indices in the original array
    indices = np.where(mask)[0]
    start_idx = int(indices[0])
    end_idx = int(indices[-1]) + 1  # exclusive

    # Apply WARMUP: skip first WARMUP rows for feature burn-in
    warmup_start = start_idx + WARMUP
    if warmup_start >= end_idx:
        raise ValueError(
            f"{label}: after WARMUP={WARMUP}, no usable steps "
            f"(range has {end_idx - start_idx} rows)"
        )

    return DataSlice(contract, start_idx=warmup_start, end_idx=end_idx, label=label)


def get_train_slice(ticker: str, round_num: int) -> tuple[ContractArrays, DataSlice, DataSlice, dict]:
    """Load npz, slice to train period, split 90/10 for train/val.
    
    Returns: (full_train_contract, train_slice, val_slice, meta)
    - full_train_contract: ContractArrays sliced to train dates
    - train_slice: DataSlice for training (90% of train period)
    - val_slice: DataSlice for validation (last 10%, with SEQ_LEN overlap)
    - meta: dict with metadata
    """
    from drl.dqn.spec import SEQ_LEN

    data, full_contract, meta = load_npz(ticker, round_num)
    dates = pd.to_datetime(full_contract.dates)
    train_start_idx = int(meta["train_start_idx"])
    train_end_idx = int(meta["train_end_idx"])
    n_train = train_end_idx + 1
    assert n_train > WARMUP + 50, (
        f"{ticker} r{round_num}: train period too short ({n_train} rows)"
    )

    train_contract = ContractArrays(
        ticker=full_contract.ticker,
        prices=full_contract.prices[:n_train],
        returns=full_contract.returns[:n_train],
        sigma=full_contract.sigma[:n_train],
        features=full_contract.features[:n_train],
        dates=full_contract.dates[:n_train],
        source=full_contract.source,
    )

    n = len(train_contract.prices)
    val_split = 0.10
    n_train_period = train_end_idx - train_start_idx + 1
    split_idx = train_start_idx + int(n_train_period * (1 - val_split))
    val_start = max(WARMUP, train_start_idx, split_idx - SEQ_LEN)

    train_slice = DataSlice(
        train_contract,
        start_idx=max(WARMUP, train_start_idx),
        end_idx=split_idx,
        label=f"{ticker} r{round_num} train",
    )
    val_slice = DataSlice(train_contract, start_idx=val_start, end_idx=n, label=f"{ticker} r{round_num} val")

    assert train_slice.usable_steps > 0, f"{ticker} r{round_num}: 0 train steps"
    assert val_slice.usable_steps > 0, f"{ticker} r{round_num}: 0 val steps"

    meta["n_total"] = len(full_contract.prices)
    meta["n_train_period"] = n_train
    meta["n_train"] = split_idx
    meta["n_val"] = n - val_start
    meta["train_dates"] = f"{dates[train_start_idx].date()} ~ {dates[train_end_idx].date()}"

    return train_contract, train_slice, val_slice, meta


def get_test_slice(ticker: str, round_num: int) -> tuple[ContractArrays, int, dict]:
    """Load npz, slice to test period.
    
    Returns: (test_contract, start_idx, meta)
    - test_contract: ContractArrays sliced to test dates
    - start_idx: WARMUP offset within test_contract
    - meta: dict with metadata
    """
    data, full_contract, meta = load_npz(ticker, round_num)

    dates = pd.to_datetime(full_contract.dates)
    test_start_idx = int(meta["test_start_idx"])
    test_end_idx = int(meta["test_end_idx"])
    n_test = test_end_idx - test_start_idx + 1
    assert n_test > WARMUP + 10, (
        f"{ticker} r{round_num}: test period too short ({n_test} rows)"
    )

    test_contract = ContractArrays(
        ticker=full_contract.ticker,
        prices=full_contract.prices[test_start_idx:test_end_idx + 1],
        returns=full_contract.returns[test_start_idx:test_end_idx + 1],
        sigma=full_contract.sigma[test_start_idx:test_end_idx + 1],
        features=full_contract.features[test_start_idx:test_end_idx + 1],
        dates=full_contract.dates[test_start_idx:test_end_idx + 1],
        source=full_contract.source,
    )

    meta["n_test"] = len(test_contract.prices)
    meta["test_dates"] = f"{dates[test_start_idx].date()} ~ {dates[test_end_idx].date()}"

    return test_contract, WARMUP, meta
