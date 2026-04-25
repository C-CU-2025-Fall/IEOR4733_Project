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
ACTIVE_FEATURE_VERSION = ACTIVE_FEATURE_LINE  # Backward-compatible alias.
STATE_SPEC_VERSION = "structural_38_ewma60_close_deviation"
STATE_SPEC_VERSION_V4 = "v4_ewma60_close_deviation"

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


def feature_data_path(round_num: int, ticker: str, version: str | None = None) -> Path:
    if version:
        return FEATURE_ROOT / version / ticker_slug(ticker) / f"{round_name(round_num)}.npz"
    return FEATURE_ROOT / ticker_slug(ticker) / f"{round_name(round_num)}.npz"


def resolve_feature_data_path(round_num: int, ticker: str, version: str | None = None) -> Path:
    return feature_data_path(round_num, ticker, version=version)


def normalize_model_version(model_version: str | None = None) -> str:
    """Archive-only compatibility alias.

    Mainline DRL no longer routes behavior by version. The active line is the
    single structural-38-aligned feature spec.
    """
    if model_version in (None, "", "current", ACTIVE_FEATURE_LINE, ACTIVE_FEATURE_VERSION):
        return ACTIVE_FEATURE_LINE
    raise ValueError(
        f"Versioned DRL lines are archive-only and unsupported in the mainline: {model_version}"
    )


def current_source_policy() -> dict:
    from frontier_presets import STRUCTURAL_38_EXCLUDED, STRUCTURAL_38_OVERRIDES

    return {
        "preset": "structural_38",
        "source_overrides": dict(STRUCTURAL_38_OVERRIDES),
        "excluded_contracts": sorted(STRUCTURAL_38_EXCLUDED),
    }


def preset_policy(model_version: str | None = None) -> dict:
    _ = model_version
    return current_source_policy()


def feature_spec(model_version: str | None = None, version: str | None = None) -> dict:
    _ = model_version
    policy = current_source_policy()
    ver = version or "mainline"
    if ver == "v4":
        close_feature = {
            "name": "ewma60_close_deviation_v4",
            "formula": "(p_t - EMA60(p)_t) / EWMA60(r)_t",
            "causal": True,
        }
        state_spec = STATE_SPEC_VERSION_V4
    else:
        close_feature = {
            "name": "ewma60_close_deviation",
            "formula": "(p_t - EMA60(p)_t) / (EWMA60(r)_t * sqrt(60))",
            "causal": True,
        }
        state_spec = STATE_SPEC_VERSION
    return {
        "feature_line": ACTIVE_FEATURE_LINE,
        "state_spec_version": state_spec,
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
