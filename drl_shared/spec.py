"""Global shared DRL spec constants and paths for the active mainline."""
from __future__ import annotations

from pathlib import Path

from config import ASSET_CLASSES

REPO_ROOT = Path(__file__).resolve().parents[1]

SEQ_LEN = 60
FEATURE_DIM = 8
HORIZONS = (21, 42, 63, 252)
RSI_WINDOW = 30
WARMUP = 252

ACTIVE_FEATURE_LINE = "structural_38_mainline"
STATE_SPEC_VERSION = "structural_38_ewma60_close_deviation_no_sqrt60"

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


def asset_slug(asset_name: str) -> str:
    if asset_name not in ASSET_CLASSES:
        raise ValueError(f"Unknown asset universe: {asset_name}")
    return asset_name.replace(" ", "_")


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


def feature_data_path(round_num: int, ticker: str) -> Path:
    return FEATURE_ROOT / ticker_slug(ticker) / f"{round_name(round_num)}.npz"


def asset_index_path(asset_name: str, round_num: int) -> Path:
    root = FEATURE_ROOT / asset_slug(asset_name) / round_name(round_num)
    return root / "index.json"


def resolve_feature_data_path(round_num: int, ticker: str) -> Path:
    return feature_data_path(round_num, ticker)


def current_source_policy() -> dict:
    from frontier_presets import STRUCTURAL_38_EXCLUDED, STRUCTURAL_38_OVERRIDES

    return {
        "preset": "structural_38",
        "source_overrides": dict(STRUCTURAL_38_OVERRIDES),
        "excluded_contracts": sorted(STRUCTURAL_38_EXCLUDED),
    }


def feature_spec() -> dict:
    policy = current_source_policy()
    close_feature = {
        "name": "ewma60_close_deviation",
        "formula": "(p_t - EMA60(p)_t) / EWMA60(r)_t",
        "causal": True,
    }
    return {
        "feature_line": ACTIVE_FEATURE_LINE,
        "state_spec_version": STATE_SPEC_VERSION,
        "seq_len": SEQ_LEN,
        "feature_dim": FEATURE_DIM,
        "close_feature": close_feature,
        "return_horizons": list(HORIZONS),
        "return_feature_formula": "(p_t - p_{t-H}) / (EWMA60(r)_t * sqrt(H))",
        "volatility_estimator": "EWMA(60) std of additive r_t",
        "macd_feature": "averaged MACD normalized by 63-day price volatility",
        "rsi_window": RSI_WINDOW,
        "volatility_feature": "EWMA60(r_t) / mean(EWMA60(r_t))",
        "preset": policy["preset"],
    }
