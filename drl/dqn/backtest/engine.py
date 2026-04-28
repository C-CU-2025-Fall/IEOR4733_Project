"""DQN backtest adapter: model inference only, baseline engine for metrics.

Single unified pipeline for all strategies:
  - Baseline strategies (Long, Sign(R), MACD) — no model needed
  - DQN with per_contract or asset_class training mode
  - DQN ensemble: best (top-1) or top3 (avg Q-values across top-3 seeds)

All modes share:
  - Same portfolio_metrics entry point
  - Same preset (source_overrides + excluded_contracts)
  - Same metric computation (9 metrics from one R_port)
  - Same batched inference (no step-by-step env loops)

Round/test periods driven by RETRAIN_ROUNDS config.
"""
from __future__ import annotations

import json
import re as _re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from drl.dqn.model import DQNAgent
from drl.dqn.spec import (
    MODEL_ROOT,
    RETRAIN_ROUNDS,
    SIGMA_TGT,
    WARMUP,
    maybe_load_manifest_for_checkpoint,
    resolve_checkpoint_path,
    asset_slug,
    round_name,
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

# Unified agent tuple: (agent, manifest_or_none, path_or_none)
# Works for both per-contract and asset-class modes.
AgentTuple = tuple[DQNAgent, dict | None, Path | None]


def canonical_strategy_name(strategy: str) -> str:
    key = strategy.strip().lower()
    if key not in STRATEGY_LABELS:
        raise ValueError(f"Unsupported strategy: {strategy}")
    return STRATEGY_LABELS[key]


def current_dqn_policy() -> tuple[dict[str, str], list[str]]:
    """Single source of truth for preset (source_overrides + excluded)."""
    policy = current_source_policy()
    return dict(policy["source_overrides"]), list(policy["excluded_contracts"])


# ─── Model loading (unified) ───


def _load_agent_from_checkpoint(ckpt_path: Path, device: str | None = None) -> AgentTuple:
    """Load a single DQN agent from a checkpoint path."""
    agent = DQNAgent(device=device)
    agent.load(ckpt_path)
    agent.q_net.eval()
    manifest = maybe_load_manifest_for_checkpoint(ckpt_path) or None
    return agent, manifest, ckpt_path


def _read_seed_rankings(root: Path) -> list[dict]:
    """Read seed rankings from best_seed.json or parse train.log files.

    Returns list sorted by val_reward descending:
      [{"dir": "/path/to/seed_dir", "val_reward": float}, ...]
    """
    best_json = root / "best_seed.json"
    if best_json.exists():
        with best_json.open() as f:
            info = json.load(f)
        return info.get("all_seeds", [])

    # Fallback: parse train.log for val_reward
    picks = []
    for d in root.iterdir():
        if not d.is_dir() or not (d / "checkpoint.pt").exists():
            continue
        log_path = d / "train.log"
        best_val = None
        if log_path.exists():
            m = _re.findall(r"best=([+-]?\d+\.\d+)", log_path.read_text())
            if m:
                best_val = float(m[-1])
        picks.append({"dir": str(d), "val_reward": best_val or 0.0})
    picks.sort(key=lambda x: x.get("val_reward", 0.0), reverse=True)
    return picks


def load_agents(
    round_num: int,
    asset_name: str,
    training_mode: str = "asset_class",
    ensemble_mode: str = "best",
    device: str | None = None,
) -> list[AgentTuple]:
    """Load DQN agents — unified for per-contract and asset-class modes.

    Returns list of AgentTuple. Length depends on ensemble_mode:
      - "best": 1 agent (highest val_reward)
      - "top3": up to 3 agents (top-3 by val_reward)

    For per-contract mode, asset_name is the ticker symbol.
    For asset-class mode, asset_name is the asset class name.
    """
    slug = asset_slug(asset_name) if training_mode == "asset_class" else asset_name.upper()
    root = MODEL_ROOT / slug / round_name(round_num)

    rankings = _read_seed_rankings(root)
    if not rankings:
        raise FileNotFoundError(f"No seeds found for {asset_name} r{round_num} ({training_mode})")

    if ensemble_mode not in ("best", "top3"):
        raise ValueError(f"Unknown ensemble_mode: {ensemble_mode!r}, expected 'best' or 'top3'")
    n = 3 if ensemble_mode == "top3" else 1
    picks = rankings[:n]

    results = []
    for entry in picks:
        d = Path(entry["dir"])
        ckpt = d / "checkpoint.pt"
        if not ckpt.exists():
            continue
        results.append(_load_agent_from_checkpoint(ckpt, device=device))

    if not results:
        raise FileNotFoundError(
            f"No valid checkpoints for {asset_name} r{round_num} "
            f"mode={training_mode} ensemble={ensemble_mode}"
        )
    return results


# ─── Batched inference ───


def _batched_action_ids(
    agents: list[DQNAgent],
    states: np.ndarray,
    batch_size: int,
    progress: bool,
    desc: str,
) -> np.ndarray:
    """Batched action prediction. Single agent or ensemble (avg Q-values)."""
    if len(states) == 0:
        return np.zeros(0, dtype=np.int64)
    out = []
    starts = range(0, len(states), batch_size)
    iterator = tqdm(starts, desc=desc, unit="batch", leave=False, disable=not progress)
    for start in iterator:
        batch = states[start:start + batch_size]
        batch_t = torch.from_numpy(batch).float()
        if len(agents) == 1:
            with torch.no_grad():
                q = agents[0].q_net(batch_t.to(agents[0].device))
            out.append(q.argmax(dim=1).cpu().numpy())
        else:
            q_sum = None
            for ag in agents:
                with torch.no_grad():
                    q = ag.q_net(batch_t.to(ag.device))
                q_sum = q if q_sum is None else q_sum + q
            avg_q = q_sum / len(agents)
            out.append(avg_q.argmax(dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


# ─── Round resolution ───


def _resolve_rounds_for_dates(
    dates: pd.DatetimeIndex,
    round_num: int | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
) -> list[tuple[np.ndarray, int]]:
    """Map dates to (mask, round_num) pairs."""
    if round_num is not None:
        if test_start is not None and test_end is not None:
            mask = (dates >= pd.Timestamp(test_start)) & (dates <= pd.Timestamp(test_end))
            return [(np.asarray(mask, dtype=bool), round_num)]
        return [(np.ones(len(dates), dtype=bool), round_num)]
    masks = []
    for rn in sorted(RETRAIN_ROUNDS):
        info = RETRAIN_ROUNDS[rn]
        ts = test_start or info["test_start"]
        te = test_end or info["test_end"]
        mask = (dates >= pd.Timestamp(ts)) & (dates <= pd.Timestamp(te))
        if np.any(mask):
            masks.append((np.asarray(mask, dtype=bool), rn))
    return masks


# ─── Position provider ───


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
    training_mode: str = "per_contract",
    ensemble_mode: str = "best",
    test_start: str | None = None,
    test_end: str | None = None,
):
    """Return a provider(rd)->positions for baseline_run.compute_strategy_metrics."""
    if (checkpoint or checkpoint_bundle) and round_num is None:
        raise ValueError("Explicit DQN checkpoint/checkpoint-bundle requires --round.")

    # Unified cache: key -> list[AgentTuple]
    cache: dict[tuple[str, int], list[AgentTuple]] = {}

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

        round_masks = _resolve_rounds_for_dates(
            eval_dates[:eval_len], round_num=round_num,
            test_start=test_start, test_end=test_end,
        )
        if not round_masks:
            return positions

        sigma = np.asarray(rd["sigma"], dtype=float)
        features = build_feature_matrix(prices, returns, sigma, feature_spec_override=feature_spec())

        for mask, rn in round_masks:
            # For per-contract: load per ticker. For asset-class: load per asset class.
            lookup_key = ticker if training_mode == "per_contract" else asset_name
            cache_key = (lookup_key, rn)
            if cache_key not in cache:
                if checkpoint and round_num is not None:
                    # Explicit checkpoint path overrides everything
                    cache[cache_key] = [_load_agent_from_checkpoint(Path(checkpoint), device=device)]
                else:
                    cache[cache_key] = load_agents(
                        rn,
                        asset_name=lookup_key,
                        training_mode=training_mode,
                        ensemble_mode=ensemble_mode,
                        device=device,
                    )

            agents_tuples = cache[cache_key]

            # Manifest validation (asset-class only)
            if training_mode == "asset_class":
                manifest = agents_tuples[0][1]
                if manifest:
                    manifest_asset = manifest.get("asset_class")
                    if manifest_asset and manifest_asset != asset_name:
                        raise ValueError(
                            f"Asset-class checkpoint mismatch for r{rn}: "
                            f"manifest={manifest_asset!r}, backtest={asset_name!r}"
                        )
                    manifest_sigma = manifest.get("sigma_tgt")
                    if (expected_sigma_tgt is not None and manifest_sigma is not None
                            and not np.isclose(float(manifest_sigma), expected_sigma_tgt)):
                        raise ValueError(
                            f"sigma_tgt mismatch for {ticker} r{rn}: "
                            f"manifest={manifest_sigma} backtest={expected_sigma_tgt}"
                        )

            full_indices = np.array([start + int(i) for i in np.where(mask)[0]], dtype=int)
            valid = full_indices[(full_indices >= WARMUP) & (full_indices < len(prices))]
            if len(valid) == 0:
                continue
            states = np.stack([get_feature_window(features, int(idx)) for idx in valid]).astype(np.float32)

            agents_only = [t[0] for t in agents_tuples]
            action_ids = _batched_action_ids(
                agents_only, states,
                batch_size=max(1, int(batch_size)),
                progress=progress,
                desc=f"DQN {ticker} r{rn}" + (f" ({len(agents_only)}ens)" if len(agents_only) > 1 else ""),
            )
            positions[valid] = [action_id_to_position(aid) for aid in action_ids]
        return positions

    return provider


# ─── Portfolio metrics (main entry) ───


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
    training_mode: str = "per_contract",
    ensemble_mode: str = "best",
    test_start: str | None = None,
    test_end: str | None = None,
) -> dict[str, float]:
    """Compute 9 portfolio-level metrics for any strategy.

    All strategies use the same preset (source_overrides + excluded_contracts).
    For baseline strategies (Long/Sign(R)/MACD), DQN params are ignored.
    For DQN, training_mode and ensemble_mode control model loading.
    """
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
            training_mode=training_mode,
            ensemble_mode=ensemble_mode,
            test_start=test_start,
            test_end=test_end,
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
