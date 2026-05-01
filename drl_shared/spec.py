"""Global shared DRL spec constants and paths for the active mainline."""
from __future__ import annotations

from pathlib import Path

from config import ASSET_CLASSES

REPO_ROOT = Path(__file__).resolve().parents[1]

SEQ_LEN = 60
MARKET_FEATURE_DIM = 5  # ret_1d/sigma + ret_21d + MACD(8,24) + MACD(16,48) + RSI(30)
FEATURE_DIM = 5  # = MARKET_FEATURE_DIM (prev_action removed)
HORIZONS = (21,)
MACD_PAIRS_ACTIVE = [(8, 24), (16, 48)]
RSI_WINDOW = 30
WARMUP = SEQ_LEN

ACTIVE_FEATURE_LINE = "structural_38_pruned_v2"
STATE_SPEC_VERSION = "structural_38_pruned_v2_5d"

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
        "market_feature_dim": MARKET_FEATURE_DIM,
        "features": [
            {"index": 0, "name": "ret_1d_vol_norm", "formula": "r_t / sigma_t", "note": "replaces non-stationary p_t/rolling_std(p,60)"},
            {"index": 1, "name": "ret_21d_vol_norm", "formula": "(p_t - p_{t-21}) / (sigma_t * sqrt(21))"},
            {"index": 2, "name": "macd_8_24", "formula": "q_t / std_252(q_t), q_t = (EMA_8 - EMA_24) / std_63(p)"},
            {"index": 3, "name": "macd_16_48", "formula": "q_t / std_252(q_t), q_t = (EMA_16 - EMA_48) / std_63(p)"},
            {"index": 4, "name": "rsi_30_norm", "formula": "(RSI_30 - 50) / 50, Wilder smoothing"},
        ],
        "return_horizons": list(HORIZONS),
        "volatility_estimator": "EWMA(60) std of additive r_t",
        "macd_feature": {
            "name": "two_pair_macd_stack",
            "pairs": [list(p) for p in MACD_PAIRS_ACTIVE],
        },
        "rsi_window": RSI_WINDOW,
        "preset": policy["preset"],
    }
