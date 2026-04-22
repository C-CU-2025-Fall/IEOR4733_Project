"""Canonical DQN spec for shared-model training and inference."""
from __future__ import annotations

from pathlib import Path

from config import (
    ASSET_CLASSES,
    BP,
    EWMA_SPAN,
    MACD_PAIRS,
    MACD_STD_WINDOW,
    MACD_VOL_WINDOW,
    SOURCE_OVERRIDES,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DQN_ROOT = REPO_ROOT / "dqn"
DATA_ROOT = DQN_ROOT / "data"
MODEL_ROOT = DQN_ROOT / "models"
SHARED_DATA_ROOT = DATA_ROOT / "shared_rounds"
SHARED_MODEL_ROOT = MODEL_ROOT / "shared_rounds"

DQN_SPEC_VERSION = "shared_v1"
TRAINING_MODE = "shared_model"

SEQ_LEN = 60
FEATURE_DIM = 8
ACTION_VALUES = (-1.0, 0.0, 1.0)
ACTION_DIM = len(ACTION_VALUES)
RSI_WINDOW = 30
WARMUP = 252

# Paper Table 1 hyperparameters
LR = 0.0001
GAMMA = 0.3
BATCH_SIZE = 64
MEMORY_SIZE = 5000
TAU = 1000
EPS_START = 0.3
EPS_END = 0.05
EPS_DECAY_STEPS = 50000
EPISODES = 200
MAX_STEPS_PER_EP = 1500
SIGMA_TGT = 0.063

HORIZONS = (21, 42, 63, 252)
LEAKY_RELU_SLOPE = 0.01
LSTM_HIDDEN_SIZES = (64, 32)

SHARED_ROUNDS = {
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


def universe_tickers(asset_name: str = "All") -> list[str]:
    if asset_name in (None, "", "All"):
        tickers = []
        for asset_tickers in ASSET_CLASSES.values():
            tickers.extend(asset_tickers)
        return tickers
    if asset_name not in ASSET_CLASSES:
        raise ValueError(f"Unknown asset universe: {asset_name}")
    return list(ASSET_CLASSES[asset_name])


def universe_slug(asset_name: str = "All") -> str:
    return (asset_name or "All").lower().replace(" ", "_")


def round_name(round_num: int) -> str:
    if round_num not in SHARED_ROUNDS:
        raise ValueError(f"Unknown shared DQN round: {round_num}")
    return f"r{round_num}"


def round_data_dir(round_num: int, asset_name: str = "All") -> Path:
    return SHARED_DATA_ROOT / universe_slug(asset_name) / round_name(round_num)


def round_model_path(round_num: int, asset_name: str = "All") -> Path:
    return SHARED_MODEL_ROOT / universe_slug(asset_name) / f"{round_name(round_num)}.pt"


def checkpoint_metadata(round_num: int, asset_name: str, extra: dict | None = None) -> dict:
    meta = {
        "spec_version": DQN_SPEC_VERSION,
        "training_mode": TRAINING_MODE,
        "round": round_num,
        "round_info": SHARED_ROUNDS[round_num],
        "asset_universe": asset_name,
        "tickers": universe_tickers(asset_name),
        "feature_dim": FEATURE_DIM,
        "seq_len": SEQ_LEN,
        "action_values": list(ACTION_VALUES),
        "sigma_tgt": SIGMA_TGT,
        "bp": BP,
        "ewma_span": EWMA_SPAN,
        "macd_pairs": list(MACD_PAIRS),
        "macd_vol_window": MACD_VOL_WINDOW,
        "macd_std_window": MACD_STD_WINDOW,
        "rsi_window": RSI_WINDOW,
        "source_overrides": SOURCE_OVERRIDES,
        "hyperparameters": {
            "lr": LR,
            "gamma": GAMMA,
            "batch_size": BATCH_SIZE,
            "memory_size": MEMORY_SIZE,
            "tau": TAU,
            "eps_start": EPS_START,
            "eps_end": EPS_END,
            "eps_decay_steps": EPS_DECAY_STEPS,
            "episodes": EPISODES,
            "max_steps_per_ep": MAX_STEPS_PER_EP,
            "lstm_hidden_sizes": list(LSTM_HIDDEN_SIZES),
        },
    }
    if extra:
        meta.update(extra)
    return meta

