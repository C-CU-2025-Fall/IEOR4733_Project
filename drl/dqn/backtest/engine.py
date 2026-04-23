"""DQN backtest adapter: model inference only, baseline engine for metrics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import EXCLUDED_CONTRACTS, SOURCE_OVERRIDES
from drl.dqn.model import DQNAgent
from drl.dqn.spec import (
    ACTIVE_MODEL_VERSION,
    RETRAIN_ROUNDS,
    SIGMA_TGT,
    WARMUP,
    maybe_load_manifest_for_checkpoint,
    resolve_checkpoint_path,
)
from drl_shared.state_space import action_id_to_position, build_feature_matrix, get_feature_window
from strategy_backtester import backtest_strategy_metrics, paper_table3_reference

STRATEGY_LABELS = {
    "long": "Long",
    "sign_r": "Sign(R)",
    "sign(r)": "Sign(R)",
    "macd": "MACD",
    "dqn": "DQN",
}


def canonical_strategy_name(strategy: str) -> str:
    key = strategy.strip().lower()
    if key not in STRATEGY_LABELS:
        raise ValueError(f"Unsupported strategy: {strategy}")
    return STRATEGY_LABELS[key]


def _load_dqn_agent(
    round_num: int,
    ticker: str,
    checkpoint: str | None = None,
    checkpoint_bundle: str | None = None,
    model_version: str = ACTIVE_MODEL_VERSION,
    run_id: str = "latest",
    device: str | None = None,
) -> tuple[DQNAgent, dict | None, Path]:
    ckpt_path = resolve_checkpoint_path(
        round_num,
        ticker,
        model_version=model_version,
        run_id=run_id,
        checkpoint_bundle=checkpoint_bundle,
        checkpoint=checkpoint,
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing DQN checkpoint for {ticker}: {ckpt_path}")
    agent = DQNAgent(device=device)
    checkpoint_meta = agent.load(ckpt_path)
    agent.q_net.eval()
    manifest = maybe_load_manifest_for_checkpoint(ckpt_path) or checkpoint_meta or None
    if manifest and model_version.lower() != "v0":
        actual_version = str(manifest.get("model_version", model_version)).lower()
        if actual_version != model_version.lower():
            raise ValueError(f"Model version mismatch for {ticker}: {actual_version} != {model_version}")
    return agent, manifest, ckpt_path


def _batched_action_ids(agent: DQNAgent, states: np.ndarray, batch_size: int, progress: bool, desc: str) -> np.ndarray:
    if len(states) == 0:
        return np.zeros(0, dtype=np.int64)
    out = []
    starts = range(0, len(states), batch_size)
    iterator = tqdm(starts, desc=desc, unit="batch", leave=False, disable=not progress)
    for start in iterator:
        out.append(agent.predict_action_ids(states[start:start + batch_size]))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def _resolve_rounds_for_dates(
    dates: pd.DatetimeIndex,
    round_num: int | None = None,
) -> list[tuple[np.ndarray, int]]:
    if round_num is not None:
        return [(np.ones(len(dates), dtype=bool), round_num)]
    masks = []
    for rn in sorted(RETRAIN_ROUNDS):
        info = RETRAIN_ROUNDS[rn]
        mask = (dates >= pd.Timestamp(info["test_start"])) & (dates <= pd.Timestamp(info["test_end"]))
        if np.any(mask):
            masks.append((np.asarray(mask, dtype=bool), rn))
    return masks


def dqn_position_provider(
    round_num: int | None = None,
    checkpoint: str | None = None,
    checkpoint_bundle: str | None = None,
    model_version: str = ACTIVE_MODEL_VERSION,
    run_id: str = "latest",
    device: str | None = None,
    progress: bool = False,
    batch_size: int = 2048,
    expected_sigma_tgt: float | None = None,
):
    """Return a provider(rd)->positions for baseline_run.compute_strategy_metrics."""
    if (checkpoint or checkpoint_bundle) and round_num is None:
        raise ValueError("Explicit DQN checkpoint/checkpoint-bundle requires --round.")
    cache: dict[tuple[str, int], tuple[DQNAgent, dict | None, Path]] = {}

    def provider(rd) -> np.ndarray:
        ticker = rd["tk"]
        prices = np.asarray(rd["prices"], dtype=float)
        returns = np.asarray(rd["rt"], dtype=float)
        sigma = np.asarray(rd["sigma"], dtype=float)
        start, t1 = int(rd["start"]), int(rd["t1"])
        eval_dates = pd.to_datetime(rd["dates"])
        eval_len = min(len(eval_dates), max(0, t1 - start + 1))

        features = build_feature_matrix(prices, returns, sigma, model_version=model_version)
        positions = np.zeros(len(prices), dtype=float)
        if eval_len <= 0:
            return positions

        round_masks = _resolve_rounds_for_dates(eval_dates[:eval_len], round_num=round_num)
        if not round_masks:
            return positions

        for mask, rn in round_masks:
            agent_key = (ticker, rn)
            if agent_key not in cache:
                cache[agent_key] = _load_dqn_agent(
                    rn,
                    ticker=ticker,
                    checkpoint=checkpoint if round_num is not None else None,
                    checkpoint_bundle=checkpoint_bundle if round_num is not None else None,
                    model_version=model_version,
                    run_id=run_id,
                    device=device,
                )
            agent, manifest, _ = cache[agent_key]
            if manifest and model_version.lower() != "v0":
                expected_state = "v2_ewma60_close_deviation"
                actual_state = manifest.get("state_spec_version")
                if actual_state != expected_state:
                    raise ValueError(f"State spec mismatch for {ticker} r{rn}: {actual_state} != {expected_state}")
                manifest_sigma = manifest.get("sigma_tgt")
                if expected_sigma_tgt is not None and manifest_sigma is not None and not np.isclose(float(manifest_sigma), expected_sigma_tgt):
                    raise ValueError(
                        f"sigma_tgt mismatch for {ticker} r{rn}: manifest={manifest_sigma} "
                        f"backtest={expected_sigma_tgt}"
                    )
            full_indices = np.array([start + int(i) for i in np.where(mask)[0]], dtype=int)
            valid = full_indices[(full_indices >= WARMUP) & (full_indices < len(prices))]
            if len(valid) == 0:
                continue
            states = np.stack([get_feature_window(features, int(full_idx)) for full_idx in valid]).astype(np.float32)
            action_ids = _batched_action_ids(
                agent,
                states,
                batch_size=max(1, int(batch_size)),
                progress=progress,
                desc=f"DQN {ticker} r{rn}",
            )
            positions[valid] = [action_id_to_position(action_id) for action_id in action_ids]
        return positions

    return provider


def portfolio_metrics(
    asset_name: str,
    strategy: str,
    round_num: int | None = None,
    checkpoint: str | None = None,
    checkpoint_bundle: str | None = None,
    model_version: str = ACTIVE_MODEL_VERSION,
    run_id: str = "latest",
    device: str | None = None,
    progress: bool = False,
    batch_size: int = 2048,
    sigma_tgt: float = SIGMA_TGT,
    excluded_contracts: list[str] | None = None,
    source_overrides: dict[str, str] | None = None,
) -> dict[str, float]:
    strategy_name = canonical_strategy_name(strategy)
    excluded = EXCLUDED_CONTRACTS if excluded_contracts is None else excluded_contracts
    overrides = SOURCE_OVERRIDES if source_overrides is None else source_overrides
    provider = (
        dqn_position_provider(
            round_num=round_num,
            checkpoint=checkpoint,
            checkpoint_bundle=checkpoint_bundle,
            model_version=model_version,
            run_id=run_id,
            device=device,
            progress=progress,
            batch_size=batch_size,
            expected_sigma_tgt=sigma_tgt,
        )
        if strategy_name == "DQN"
        else None
    )
    return backtest_strategy_metrics(
        asset_name=asset_name,
        strategy=strategy_name,
        sigma_tgt=sigma_tgt,
        position_provider=provider,
        excluded_contracts=excluded,
        source_overrides=overrides,
    )


def paper_reference(asset_name: str, strategy: str) -> dict[str, float] | None:
    strategy_name = canonical_strategy_name(strategy)
    return paper_table3_reference(asset_name, strategy_name)
