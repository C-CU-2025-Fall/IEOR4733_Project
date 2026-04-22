"""Canonical DQN spec on top of shared DRL state-space."""
from __future__ import annotations

from pathlib import Path

from config import BP, EWMA_SPAN, MACD_PAIRS, MACD_VOL_WINDOW, SOURCE_OVERRIDES
from drl_shared.spec import (
    CONTINUOUS_ACTION_RANGE,
    DISCRETE_ACTION_VALUES,
    FEATURE_DIM,
    HORIZONS,
    RETRAIN_ROUNDS,
    RSI_WINDOW,
    SEQ_LEN,
    SIGMA_TGT_DEFAULT,
    WARMUP,
    feature_data_path,
    round_name,
    ticker_asset_class,
    ticker_slug,
    universe_tickers,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DQN_ROOT = REPO_ROOT / "drl" / "dqn"
MODEL_ROOT = DQN_ROOT / "models"
LOG_ROOT = REPO_ROOT / "logs" / "rl"

CONTRACT_MODEL_ROOT = MODEL_ROOT / "contract_rounds"

DQN_SPEC_VERSION = "single_contract_v1"
TRAINING_MODE = "single_contract"

DISCRETE_ACTION_DIM = len(DISCRETE_ACTION_VALUES)

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
SIGMA_TGT = SIGMA_TGT_DEFAULT
LEAKY_RELU_SLOPE = 0.01
LSTM_HIDDEN_SIZES = (64, 32)


def contract_data_path(round_num: int, ticker: str) -> Path:
    return feature_data_path(round_num, ticker)


def contract_model_path(round_num: int, ticker: str, algorithm: str = "dqn") -> Path:
    return CONTRACT_MODEL_ROOT / algorithm.lower() / ticker_slug(ticker) / f"{round_name(round_num)}.pt"


def run_log_dir(algorithm: str, ticker: str, round_num: int, run_id: str) -> Path:
    return LOG_ROOT / algorithm.lower() / ticker_slug(ticker) / round_name(round_num) / run_id


def checkpoint_metadata(round_num: int, ticker: str, algorithm: str = "dqn", extra: dict | None = None) -> dict:
    ticker = ticker.upper()
    meta = {
        "spec_version": DQN_SPEC_VERSION,
        "training_mode": TRAINING_MODE,
        "algorithm": algorithm.lower(),
        "ticker": ticker,
        "asset_class": ticker_asset_class(ticker),
        "round": round_num,
        "round_info": RETRAIN_ROUNDS[round_num],
        "feature_dim": FEATURE_DIM,
        "seq_len": SEQ_LEN,
        "discrete_action_values": list(DISCRETE_ACTION_VALUES),
        "continuous_action_range": list(CONTINUOUS_ACTION_RANGE),
        "sigma_tgt": SIGMA_TGT,
        "bp": BP,
        "ewma_span": EWMA_SPAN,
        "macd_pairs": list(MACD_PAIRS),
        "macd_vol_window": MACD_VOL_WINDOW,
        "rsi_window": RSI_WINDOW,
        "source_override": SOURCE_OVERRIDES.get(ticker, "RAD"),
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
