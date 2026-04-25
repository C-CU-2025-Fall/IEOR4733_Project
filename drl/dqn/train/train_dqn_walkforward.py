#!/usr/bin/env python3
"""Single-contract walk-forward DQN training."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drl.dqn.logging_utils import RunLogger, make_run_id
from drl.dqn.model import DQNAgent
from drl.dqn.spec import (
    EPISODES,
    MAX_STEPS_PER_EP,
    RETRAIN_ROUNDS,
    checkpoint_metadata,
    contract_data_path,
    checkpoint_path_for_bundle,
    model_bundle_root,
    round_name,
)
from drl_shared.spec import current_source_policy, feature_spec
from drl_shared.state_space import ContractArrays, ContractEnv


def _npz_scalar(data, key: str, default=None):
    if key not in data:
        return default
    value = data[key]
    if getattr(value, "shape", None) == ():
        return value.item()
    return value


def load_contract_round(ticker: str, round_num: int, version: str | None = None) -> tuple[ContractArrays, dict]:
    ticker = ticker.upper()
    path = contract_data_path(round_num, ticker, version=version)
    if not path.exists():
        raise FileNotFoundError(
            f"No prepared shared feature data found at {path}. "
            "Run python drl_shared/prepare_features.py first."
        )
    data = np.load(path, allow_pickle=True)
    expected = feature_spec(version=version)
    actual_state = _npz_scalar(data, "state_spec_version")
    if actual_state != expected["state_spec_version"]:
        raise ValueError(
            f"Feature spec mismatch for {ticker} r{round_num}: "
            f"{actual_state!r} != {expected['state_spec_version']!r}. "
            "Regenerate features with drl_shared/prepare_features.py."
        )
    meta = {
        "feature_artifact_path": str(path),
        "feature_spec": json.loads(str(_npz_scalar(data, "feature_spec", "{}"))),
        "state_spec_version": actual_state,
        "feature_line": str(_npz_scalar(data, "feature_line", expected["feature_line"])),
        "preset": _npz_scalar(data, "preset", None),
        "source_overrides": json.loads(str(_npz_scalar(data, "source_overrides", "{}"))),
        "excluded_contracts": json.loads(str(_npz_scalar(data, "excluded_contracts", "[]"))),
        "train_start": str(_npz_scalar(data, "train_start", "")),
        "train_end": str(_npz_scalar(data, "train_end", "")),
        "test_start": str(_npz_scalar(data, "test_start", "")),
        "test_end": str(_npz_scalar(data, "test_end", "")),
    }
    contract = ContractArrays(
        ticker=ticker,
        prices=data["prices"],
        returns=data["returns"],
        sigma=data["sigma"],
        features=data["features"],
        dates=data["dates"],
        source=str(_npz_scalar(data, "source", "")),
    )
    return contract, meta


def train_contract_round(
    ticker: str,
    round_num: int,
    episodes: int = EPISODES,
    early_stop_patience: int = 0,
    device: str | None = None,
    seed: int | None = None,
    sigma_tgt: float = 0.058,
    version: str | None = None,
) -> tuple[Path, Path]:
    ticker = ticker.upper()
    round_info = RETRAIN_ROUNDS[round_num]
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    contract, feature_meta = load_contract_round(ticker, round_num, version=version)
    expected_policy = current_source_policy()
    resolved_preset = feature_meta.get("preset") or expected_policy["preset"]
    if resolved_preset != expected_policy["preset"]:
        raise ValueError(
            f"Mainline DQN requires preset={expected_policy['preset']!r}, got {resolved_preset!r}"
        )
    if feature_meta.get("source_overrides") != expected_policy["source_overrides"]:
        raise ValueError("Feature artifact source_overrides do not match structural_38")
    if sorted(feature_meta.get("excluded_contracts", [])) != sorted(expected_policy["excluded_contracts"]):
        raise ValueError("Feature artifact excluded_contracts do not match structural_38")
    env = ContractEnv(contract, sigma_tgt=sigma_tgt)
    agent = DQNAgent(device=device)
    run_id = make_run_id()
    bundle_dir = model_bundle_root(round_num, ticker, run_id=run_id, version=version)
    logger = RunLogger("dqn", ticker, round_num, run_id=run_id, base_dir=bundle_dir)
    metadata = checkpoint_metadata(
        round_num,
        ticker,
        algorithm="dqn",
        extra={
            "run_id": logger.run_id,
            "episodes": episodes,
            "device": agent.device,
            "log_dir": str(logger.dir),
            "bundle_dir": str(bundle_dir),
            "seed": seed,
            "preset": resolved_preset,
            "sigma_tgt": sigma_tgt,
            **feature_meta,
        },
    )
    logger.write_json("manifest.json", metadata)
    logger.write_json("train_config.json", metadata["hyperparameters"] | {"episodes": episodes, "seed": seed})
    logger.write_json("feature_spec.json", metadata["feature_spec"])

    t0 = time.time()
    report_interval = max(1, episodes // 10)
    global_step = 0
    episode_rows = []

    # Early stopping state
    best_avg = -np.inf
    best_state = None
    best_ep = 0
    patience_counter = 0

    logger.log(f"{'=' * 70}")
    logger.log(f"DQN Training — {ticker} — {round_name(round_num)}")
    logger.log(f"Train: {round_info['train_start']} ~ {round_info['train_end']}")
    logger.log(f"Test : {round_info['test_start']} ~ {round_info['test_end']}")
    logger.log(f"Source: {contract.source} | Train days: {len(contract.prices)} | Device: {agent.device}")
    logger.log(f"Model bundle: {bundle_dir}")
    logger.log(f"State spec: {metadata['state_spec_version']}")
    logger.log("Architecture: LSTM[64,32] + Leaky-ReLU + Fixed Q-targets + Double DQN + Dueling DQN")
    if early_stop_patience > 0:
        logger.log(f"Early Stop: patience={early_stop_patience}")
    logger.log(f"{'=' * 70}")

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        last_loss = 0.0
        done = False
        steps = 0
        while not done and steps < MAX_STEPS_PER_EP:
            eps = agent.epsilon_for_step(global_step)
            action_id = agent.act(state, eps)
            next_state, reward, done = env.step(action_id)
            agent.push(state, action_id, reward, next_state, float(done))
            loss = agent.learn()
            if loss > 0:
                last_loss = loss
            state = next_state
            total_reward += reward
            steps += 1
            global_step += 1

        row = {
            "episode": ep + 1,
            "reward": round(total_reward, 6),
            "steps": steps,
            "epsilon_end": round(agent.epsilon_for_step(global_step), 6),
            "last_loss": round(last_loss, 6),
        }
        episode_rows.append(row)

        # Early stopping: check every episode
        if early_stop_patience > 0:
            if total_reward > best_avg:
                best_avg = total_reward
                best_state = {k: v.clone() for k, v in agent.q_net.state_dict().items()}
                best_ep = ep + 1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:

                    logger.log(f"Early stop @ ep{ep + 1} (best={best_avg:+.2f} @ ep{best_ep})")
                    break


        if (ep + 1) % report_interval == 0:
            elapsed = time.time() - t0
            logger.log(
                f"ep {ep + 1}/{episodes} reward={total_reward:+.4f} "
                f"steps={steps} epsilon={row['epsilon_end']:.4f} "
                f"loss={last_loss:.6f} ({elapsed:.0f}s)"
            )

    # Restore best model if early stopped
    if best_state is not None and patience_counter >= early_stop_patience:
        agent.q_net.load_state_dict(best_state)
        agent.target.load_state_dict(best_state)
        logger.log(f"Restored best model (ep={best_ep})")

    out_path = checkpoint_path_for_bundle(bundle_dir)
    agent.save(out_path, metadata=metadata)
    logger.write_csv("episode_metrics.csv", episode_rows)
    logger.write_json("checkpoint_metadata.json", metadata)
    logger.log(f"Saved checkpoint: {out_path}")
    logger.log(f"Saved logs: {logger.dir}")
    return out_path, logger.dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--round", type=int, required=True, choices=sorted(RETRAIN_ROUNDS))
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--early-stop", type=int, default=0, help="Early stopping patience (0=disabled)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sigma-tgt", type=float, default=0.058)
    parser.add_argument("--version", default=None, help="Feature version (e.g., v4)")
    args = parser.parse_args()

    train_contract_round(
        args.ticker,
        args.round,
        args.episodes,
        early_stop_patience=args.early_stop,
        device=args.device,
        seed=args.seed,
        sigma_tgt=args.sigma_tgt,
        version=args.version,
    )
