"""Global shared DRL spec constants and paths."""
from __future__ import annotations

from pathlib import Path

from config import ASSET_CLASSES

REPO_ROOT = Path(__file__).resolve().parents[1]

SEQ_LEN = 60
FEATURE_DIM = 8
HORIZONS = (21, 42, 63, 252)
RSI_WINDOW = 30
WARMUP = 252
STATE_SPEC_VERSION = "v2_ewma60_close_deviation"
ACTIVE_FEATURE_VERSION = "v2"

DISCRETE_ACTION_VALUES = (-1.0, 0.0, 1.0)
CONTINUOUS_ACTION_RANGE = (-1.0, 1.0)

SIGMA_TGT_DEFAULT = 0.058

RETRAIN_ROUNDS = {
    1: {
        "train_start": "2005-01-01",
        "train_end": "2010-12-31",
        "test_start": "2011-01-01",
        "test_end": "2015-12-31",
    },
    2: {
        "train_start": "2005-01-01",
        "train_end": "2015-12-31",
        "test_start": "2016-01-01",
        "test_end": "2019-12-31",
    },
}

FEATURE_ROOT = REPO_ROOT / "drl" / "features"
LEGACY_FEATURE_ROOT = FEATURE_ROOT / "contract_rounds"


def ticker_asset_class(ticker: str) -> str:
    ticker = ticker.upper()
    for asset_name, tickers in ASSET_CLASSES.items():
        if ticker in tickers:
            return asset_name
    raise ValueError(f"Unknown ticker: {ticker}")


def universe_tickers(asset_name: str = "All") -> list[str]:
    if asset_name in (None, "", "All"):
        tickers = []
        for asset_tickers in ASSET_CLASSES.values():
            tickers.extend(asset_tickers)
        return tickers
    if asset_name not in ASSET_CLASSES:
        raise ValueError(f"Unknown asset universe: {asset_name}")
    return list(ASSET_CLASSES[asset_name])


def ticker_slug(ticker: str) -> str:
    return ticker.upper()


def round_name(round_num: int) -> str:
    if round_num not in RETRAIN_ROUNDS:
        raise ValueError(f"Unknown retrain round: {round_num}")
    return f"r{round_num}"


def feature_data_path(round_num: int, ticker: str, model_version: str = ACTIVE_FEATURE_VERSION) -> Path:
    version = model_version.lower()
    return FEATURE_ROOT / version / ticker_slug(ticker) / f"{round_name(round_num)}.npz"


def legacy_feature_data_path(round_num: int, ticker: str) -> Path:
    return LEGACY_FEATURE_ROOT / ticker_slug(ticker) / f"{round_name(round_num)}.npz"


def resolve_feature_data_path(round_num: int, ticker: str, model_version: str = ACTIVE_FEATURE_VERSION) -> Path:
    primary = feature_data_path(round_num, ticker, model_version=model_version)
    if primary.exists():
        return primary
    if model_version.lower() == "v0":
        legacy = legacy_feature_data_path(round_num, ticker)
        if legacy.exists():
            return legacy
    return primary


def feature_spec(model_version: str = ACTIVE_FEATURE_VERSION) -> dict:
    return {
        "model_version": model_version.lower(),
        "state_spec_version": STATE_SPEC_VERSION if model_version.lower() != "v0" else "v0_full_sample_zscore",
        "seq_len": SEQ_LEN,
        "feature_dim": FEATURE_DIM,
        "close_feature": {
            "name": "ewma60_close_deviation" if model_version.lower() != "v0" else "full_sample_zscore",
            "formula": "(p_t - EMA60(p)_t) / (EWMA60(r)_t * sqrt(60))"
            if model_version.lower() != "v0"
            else "(p_t - mean(p)) / std(p)",
            "causal": model_version.lower() != "v0",
        },
        "return_horizons": list(HORIZONS),
        "return_feature_formula": "(p_t - p_{t-H}) / (EWMA60(r)_t * sqrt(H))",
        "volatility_estimator": "EWMA(60) std of additive r_t",
        "macd_feature": "averaged MACD normalized by 63-day price volatility",
        "rsi_window": RSI_WINDOW,
        "volatility_feature": "EWMA60(r_t) / mean(EWMA60(r_t))",
    }
