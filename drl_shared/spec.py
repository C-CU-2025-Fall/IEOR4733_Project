"""Global shared DRL spec constants and paths for the active mainline."""
from __future__ import annotations

from pathlib import Path

from config import ASSET_CLASSES

REPO_ROOT = Path(__file__).resolve().parents[1]

SEQ_LEN = 60
FEATURE_DIM = 9  # 1(price norm) + 4(returns) + 3(MACD pairs) + 1(RSI)
MARKET_FEATURE_DIM = FEATURE_DIM  # 9D: all features are market features

MACD_PAIRS_ACTIVE = [(8, 24), (16, 48), (32, 96)]  # alias for drl/dqn/spec.py compatibility
HORIZONS = (21, 42, 63, 252)
RSI_WINDOW = 30
WARMUP = 252

ACTIVE_FEATURE_LINE = "structural_38_mainline"
STATE_SPEC_VERSION = "structural_38_close_norm_9d"

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
    return {
        "feature_line": ACTIVE_FEATURE_LINE,
        "state_spec_version": STATE_SPEC_VERSION,
        "seq_len": SEQ_LEN,
        "feature_dim": FEATURE_DIM,
        "close_feature": {
            "name": "normalized_close_price_60d_rolling_std",
            "formula": "p_t / std_60(p)_t",
            "note": "Repo convention for current 9-feature line.",
        },
        "return_horizons": list(HORIZONS),
        "return_feature_formula": "(p_t - p_{t-H}) / (sigma_t * sqrt(H))",
        "volatility_estimator": "EWMA(60) std of additive r_t",
        "macd_feature": {
            "name": "three_pair_macd_stack",
            "pairs": [[8, 24], [16, 48], [32, 96]],
            "formula": "q_t / std_252(q_t), where q_t = (EMA_short - EMA_long) / std_63(p)",
        },
        "rsi_window": RSI_WINDOW,
        "preset": policy["preset"],
    }
