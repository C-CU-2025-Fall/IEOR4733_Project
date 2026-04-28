"""Active DQN spec on top of the shared structural-38 DRL state-space."""
from __future__ import annotations

import json
from pathlib import Path

from config import BP, EWMA_SPAN, MACD_PAIRS, MACD_VOL_WINDOW
from drl_shared.spec import (
    ACTIVE_FEATURE_LINE,
    asset_slug,
    CONTINUOUS_ACTION_RANGE,
    DISCRETE_ACTION_VALUES,
    FEATURE_DIM,
    HORIZONS,
    RETRAIN_ROUNDS,
    RSI_WINDOW,
    SEQ_LEN,
    SIGMA_TGT_DEFAULT,
    current_source_policy,
    feature_data_path,
    feature_spec,
    round_name,
    ticker_asset_class,
    ticker_slug,
    universe_tickers,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DQN_ROOT = REPO_ROOT / "drl" / "dqn"
MODEL_ROOT = DQN_ROOT / "models"
LOG_ROOT = REPO_ROOT / "logs" / "rl"

DQN_SPEC_VERSION = "asset_class_structural38_mainline"
ACTIVE_MODEL_LINE = ACTIVE_FEATURE_LINE
TRAINING_MODE = "asset_class_shared"

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
DROPOUT = 0.2
VALIDATION_SPLIT = 0.1
EARLY_STOPPING_PATIENCE = 20
WARMUP = 252


def contract_data_path(round_num: int, ticker: str) -> Path:
    return feature_data_path(round_num, ticker)


def model_bundle_root(round_num: int, asset_name: str, run_id: str) -> Path:
    return MODEL_ROOT / asset_slug(asset_name) / round_name(round_num) / run_id


def legacy_contract_model_bundle_root(round_num: int, ticker: str, run_id: str) -> Path:
    return MODEL_ROOT / ticker_slug(ticker) / round_name(round_num) / run_id


def checkpoint_path_for_bundle(bundle_dir: str | Path) -> Path:
    return Path(bundle_dir) / "checkpoint.pt"


def _valid_bundle_dirs(round_num: int, asset_name: str) -> list[Path]:
    root = MODEL_ROOT / asset_slug(asset_name) / round_name(round_num)
    if not root.exists():
        return []
    bundles = []
    for child in root.iterdir():
        if child.is_dir() and (child / "checkpoint.pt").exists():
            bundles.append(child)
    return sorted(bundles, key=lambda p: p.name)


def resolve_model_bundle(round_num: int, asset_name: str, run_id: str = "latest") -> Path:
    if run_id == "latest":
        # Prefer best_seed.json (val-reward selected) over alphabetical sort
        root = MODEL_ROOT / asset_slug(asset_name) / round_name(round_num)
        best_json = root / "best_seed.json"
        if best_json.exists():
            try:
                info = json.loads(best_json.read_text())
                best_dir = Path(info["best_model_dir"])
                if best_dir.exists() and (best_dir / "checkpoint.pt").exists():
                    return best_dir
            except Exception:
                pass
        bundles = _valid_bundle_dirs(round_num, asset_name)
        if not bundles:
            return MODEL_ROOT / asset_slug(asset_name) / round_name(round_num) / "latest"
        return bundles[-1]
    return model_bundle_root(round_num, asset_name, run_id=run_id)


def resolve_checkpoint_path(
    round_num: int,
    asset_name: str,
    run_id: str = "latest",
    checkpoint_bundle: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> Path:
    if checkpoint:
        return Path(checkpoint)
    if checkpoint_bundle:
        return checkpoint_path_for_bundle(checkpoint_bundle)
    return checkpoint_path_for_bundle(resolve_model_bundle(round_num, asset_name, run_id=run_id))


def load_manifest(bundle_dir: str | Path) -> dict | None:
    path = Path(bundle_dir) / "manifest.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def maybe_load_manifest_for_checkpoint(checkpoint: str | Path) -> dict | None:
    path = Path(checkpoint)
    manifest = path.parent / "manifest.json"
    if manifest.exists():
        with manifest.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def run_log_dir(algorithm: str, asset_or_ticker: str, round_num: int, run_id: str) -> Path:
    try:
        slug = asset_slug(asset_or_ticker)
    except ValueError:
        slug = ticker_slug(asset_or_ticker)
    return LOG_ROOT / algorithm.lower() / slug / round_name(round_num) / run_id


def checkpoint_metadata(
    round_num: int,
    asset_or_ticker: str,
    algorithm: str = "dqn",
    extra: dict | None = None,
) -> dict:
    f_spec = feature_spec()
    policy = current_source_policy()
    if asset_or_ticker in universe_tickers("All"):
        ticker = asset_or_ticker.upper()
        asset_class = ticker_asset_class(ticker)
    else:
        ticker = None
        asset_class = asset_or_ticker
    meta = {
        "feature_line": ACTIVE_FEATURE_LINE,
        "spec_version": DQN_SPEC_VERSION,
        "state_spec_version": f_spec["state_spec_version"],
        "training_mode": TRAINING_MODE,
        "algorithm": algorithm.lower(),
        "ticker": ticker,
        "asset_class": asset_class,
        "round": round_num,
        "round_info": RETRAIN_ROUNDS[round_num],
        "feature_dim": FEATURE_DIM,
        "seq_len": SEQ_LEN,
        "feature_spec": f_spec,
        "feature_artifact_path": None,
        "preset": policy["preset"],
        "source_overrides": policy["source_overrides"],
        "excluded_contracts": policy["excluded_contracts"],
        "discrete_action_values": list(DISCRETE_ACTION_VALUES),
        "continuous_action_range": list(CONTINUOUS_ACTION_RANGE),
        "sigma_tgt": SIGMA_TGT,
        "bp": BP,
        "ewma_span": EWMA_SPAN,
        "return_horizons": list(HORIZONS),
        "macd_pairs": list(MACD_PAIRS),
        "macd_vol_window": MACD_VOL_WINDOW,
        "rsi_window": RSI_WINDOW,
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
            "dropout": DROPOUT,
            "validation_split": VALIDATION_SPLIT,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
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
            "fixed_q_targets": True,
            "target_network_update": f"hard_copy_every_{TAU}_learn_steps",
            "target_update_tau": TAU,
            "double_dqn": True,
            "dueling_dqn": True,
            "paper_reference_ids": [49, 18, 50],
        },
    }
    if extra:
        meta.update(extra)
    return meta
