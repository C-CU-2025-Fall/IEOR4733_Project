"""Canonical DQN spec on top of shared DRL state-space."""
from __future__ import annotations

import json
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
    feature_spec,
    feature_data_path,
    resolve_feature_data_path,
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
LEGACY_WALKFORWARD_MODEL_ROOT = MODEL_ROOT / "walkforward"

DQN_SPEC_VERSION = "single_contract_v2"
ACTIVE_MODEL_VERSION = "v2"
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


def contract_data_path(round_num: int, ticker: str, model_version: str = ACTIVE_MODEL_VERSION) -> Path:
    return resolve_feature_data_path(round_num, ticker, model_version=model_version)


def contract_model_path(round_num: int, ticker: str, algorithm: str = "dqn") -> Path:
    return CONTRACT_MODEL_ROOT / algorithm.lower() / ticker_slug(ticker) / f"{round_name(round_num)}.pt"


def model_version_root(model_version: str = ACTIVE_MODEL_VERSION) -> Path:
    return MODEL_ROOT / model_version.lower()


def model_bundle_root(
    round_num: int,
    ticker: str,
    run_id: str,
    model_version: str = ACTIVE_MODEL_VERSION,
) -> Path:
    return model_version_root(model_version) / ticker_slug(ticker) / round_name(round_num) / run_id


def checkpoint_path_for_bundle(bundle_dir: str | Path) -> Path:
    return Path(bundle_dir) / "checkpoint.pt"


def legacy_walkforward_model_path(round_num: int, ticker: str) -> Path:
    return LEGACY_WALKFORWARD_MODEL_ROOT / f"{ticker_slug(ticker)}_{round_name(round_num)}.pt"


def resolve_contract_model_path(round_num: int, ticker: str, algorithm: str = "dqn") -> Path:
    """Return the available checkpoint path for a contract/round.

    New training writes to ``contract_rounds/dqn/<ticker>/r<k>.pt``. Some
    GPU-trained Forex checkpoints were produced before that path was locked and
    live under ``models/walkforward/<ticker>_r<k>.pt``; keep that as a read-only
    compatibility fallback for backtests and status checks.
    """
    primary = contract_model_path(round_num, ticker, algorithm=algorithm)
    if primary.exists():
        return primary
    if algorithm.lower() == "dqn":
        legacy = legacy_walkforward_model_path(round_num, ticker)
        if legacy.exists():
            return legacy
    return primary


def _valid_bundle_dirs(round_num: int, ticker: str, model_version: str = ACTIVE_MODEL_VERSION) -> list[Path]:
    root = model_version_root(model_version) / ticker_slug(ticker) / round_name(round_num)
    if not root.exists():
        return []
    bundles = []
    for child in root.iterdir():
        if child.is_dir() and (child / "checkpoint.pt").exists() and (child / "manifest.json").exists():
            bundles.append(child)
    return sorted(bundles, key=lambda p: p.name)


def resolve_model_bundle(
    round_num: int,
    ticker: str,
    model_version: str = ACTIVE_MODEL_VERSION,
    run_id: str = "latest",
) -> Path:
    if model_version.lower() == "v0":
        legacy = legacy_walkforward_model_path(round_num, ticker)
        if legacy.exists():
            return legacy.parent
    if run_id == "latest":
        bundles = _valid_bundle_dirs(round_num, ticker, model_version=model_version)
        if not bundles:
            return model_version_root(model_version) / ticker_slug(ticker) / round_name(round_num) / "latest"
        return bundles[-1]
    return model_bundle_root(round_num, ticker, run_id=run_id, model_version=model_version)


def resolve_checkpoint_path(
    round_num: int,
    ticker: str,
    model_version: str = ACTIVE_MODEL_VERSION,
    run_id: str = "latest",
    checkpoint_bundle: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> Path:
    if checkpoint:
        return Path(checkpoint)
    if checkpoint_bundle:
        return checkpoint_path_for_bundle(checkpoint_bundle)
    if model_version.lower() == "v0":
        return resolve_contract_model_path(round_num, ticker, algorithm="dqn")
    return checkpoint_path_for_bundle(resolve_model_bundle(round_num, ticker, model_version=model_version, run_id=run_id))


def load_manifest(bundle_dir: str | Path) -> dict:
    path = Path(bundle_dir) / "manifest.json"
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def maybe_load_manifest_for_checkpoint(checkpoint: str | Path) -> dict | None:
    path = Path(checkpoint)
    manifest = path.parent / "manifest.json"
    if manifest.exists():
        with manifest.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def run_log_dir(algorithm: str, ticker: str, round_num: int, run_id: str) -> Path:
    return LOG_ROOT / algorithm.lower() / ticker_slug(ticker) / round_name(round_num) / run_id


def checkpoint_metadata(
    round_num: int,
    ticker: str,
    algorithm: str = "dqn",
    model_version: str = ACTIVE_MODEL_VERSION,
    extra: dict | None = None,
) -> dict:
    ticker = ticker.upper()
    f_spec = feature_spec(model_version)
    meta = {
        "model_version": model_version.lower(),
        "spec_version": DQN_SPEC_VERSION,
        "state_spec_version": f_spec["state_spec_version"],
        "training_mode": TRAINING_MODE,
        "algorithm": algorithm.lower(),
        "ticker": ticker,
        "asset_class": ticker_asset_class(ticker),
        "round": round_num,
        "round_info": RETRAIN_ROUNDS[round_num],
        "feature_dim": FEATURE_DIM,
        "seq_len": SEQ_LEN,
        "feature_spec": f_spec,
        "feature_artifact_path": str(feature_data_path(round_num, ticker, model_version=model_version)),
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
            "leaky_relu_slope": LEAKY_RELU_SLOPE,
        },
        "reward_spec": {
            "name": "eq4_additive_price_difference",
            "return_definition": "r_t = p_t - p_{t-1}",
            "position_scale": "sigma_tgt / sigma_{t-1}",
            "transaction_cost": "bp * p_{t-1} * abs(scaled_position_t - scaled_position_{t-1})",
        },
        "architecture": {
            "family": "lstm_dqn",
            "lstm_hidden_sizes": list(LSTM_HIDDEN_SIZES),
            "head": "dueling_value_advantage",
            "target_network_update": f"hard_copy_every_{TAU}_learn_steps",
            "double_dqn": True,
        },
    }
    if extra:
        meta.update(extra)
    return meta
