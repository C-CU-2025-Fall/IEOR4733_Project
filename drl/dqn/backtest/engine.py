"""DQN backtest adapter: model inference only, baseline engine for metrics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from drl.dqn.model import DQNAgent
from drl.dqn.spec import (
    RETRAIN_ROUNDS,
    SIGMA_TGT,
    WARMUP,
    maybe_load_manifest_for_checkpoint,
    resolve_checkpoint_path,
)
from drl_shared.spec import current_source_policy, feature_spec
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


def current_dqn_policy() -> tuple[dict[str, str], list[str]]:
    policy = current_source_policy()
    return dict(policy["source_overrides"]), list(policy["excluded_contracts"])


def _load_dqn_agent(
    round_num: int,
    asset_name: str,
    checkpoint: str | None = None,
    checkpoint_bundle: str | None = None,
    run_id: str = "latest",
    device: str | None = None,
) -> tuple[DQNAgent, dict | None, Path]:
    ckpt_path = resolve_checkpoint_path(
        round_num,
        asset_name,
        run_id=run_id,
        checkpoint_bundle=checkpoint_bundle,
        checkpoint=checkpoint,
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing DQN checkpoint for {asset_name} r{round_num}: {ckpt_path}. "
            "Mainline DQN expects one asset-class checkpoint per round."
        )
    agent = DQNAgent(device=device)
    checkpoint_meta = agent.load(ckpt_path)
    agent.q_net.eval()
    manifest = maybe_load_manifest_for_checkpoint(ckpt_path) or checkpoint_meta or None
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
    asset_name: str,
    round_num: int | None = None,
    checkpoint: str | None = None,
    checkpoint_bundle: str | None = None,
    run_id: str = "latest",
    device: str | None = None,
    progress: bool = False,
    batch_size: int = 2048,
    expected_sigma_tgt: float | None = None,
):
    """Return a provider(rd)->positions for baseline_run.compute_strategy_metrics."""
    if (checkpoint or checkpoint_bundle) and round_num is None:
        raise ValueError("Explicit DQN checkpoint/checkpoint-bundle requires --round.")
    cache: dict[int, tuple[DQNAgent, dict | None, Path]] = {}

    def provider(rd) -> np.ndarray:
        ticker = rd["tk"]
        prices = np.asarray(rd["prices"], dtype=float)
        returns = np.asarray(rd["rt"], dtype=float)
        start, t1 = int(rd["start"]), int(rd["t1"])
        eval_dates = pd.to_datetime(rd["dates"])
        eval_len = min(len(eval_dates), max(0, t1 - start + 1))

        positions = np.zeros(len(prices), dtype=float)
        if eval_len <= 0:
            return positions

        round_masks = _resolve_rounds_for_dates(eval_dates[:eval_len], round_num=round_num)
        if not round_masks:
            return positions

        sigma = np.asarray(rd["sigma"], dtype=float)
        features = build_feature_matrix(prices, returns, sigma, feature_spec_override=feature_spec())

        for mask, rn in round_masks:
            if rn not in cache:
                cache[rn] = _load_dqn_agent(
                    rn,
                    asset_name=asset_name,
                    checkpoint=checkpoint if round_num is not None else None,
                    checkpoint_bundle=checkpoint_bundle if round_num is not None else None,
                    run_id=run_id,
                    device=device,
                )
            agent, manifest, _ = cache[rn]
            if manifest:
                manifest_asset = manifest.get("asset_class")
                if manifest_asset and manifest_asset != asset_name:
                    raise ValueError(
                        f"Asset-class checkpoint mismatch for r{rn}: manifest={manifest_asset!r}, "
                        f"backtest={asset_name!r}"
                    )
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
    run_id: str = "latest",
    device: str | None = None,
    progress: bool = False,
    batch_size: int = 2048,
    sigma_tgt: float = SIGMA_TGT,
    excluded_contracts: list[str] | None = None,
    source_overrides: dict[str, str] | None = None,
) -> dict[str, float]:
    strategy_name = canonical_strategy_name(strategy)
    if strategy_name == "DQN":
        overrides, excluded = current_dqn_policy()
    else:
        overrides = source_overrides
        excluded = excluded_contracts
    provider = (
        dqn_position_provider(
            asset_name=asset_name,
            round_num=round_num,
            checkpoint=checkpoint,
            checkpoint_bundle=checkpoint_bundle,
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
