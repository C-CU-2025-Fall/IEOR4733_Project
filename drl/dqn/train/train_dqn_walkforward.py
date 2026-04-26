#!/usr/bin/env python3
"""Asset-class walk-forward DQN training."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import ASSET_CLASSES
from drl.dqn.logging_utils import RunLogger, make_run_id
from drl.dqn.model import DQNAgent
from drl.dqn.spec import (
    EPISODES,
    MAX_STEPS_PER_EP,
    MODEL_ROOT,
    RETRAIN_ROUNDS,
    asset_slug,
    checkpoint_metadata,
    checkpoint_path_for_bundle,
    contract_data_path,
    model_bundle_root,
    round_name,
)
from drl_shared.spec import (
    SIGMA_TGT_DEFAULT,
    asset_index_path,
    current_source_policy,
    feature_spec,
    universe_tickers,
)
from drl_shared.state_space import (
    ContractArrays,
    ContractEnv,
    action_id_to_position,
    compute_eq4_reward,
    get_feature_window,
)


# Early stop and validation removed — paper trains all episodes without validation split


def _npz_scalar(data, key: str, default=None):
    if key not in data:
        return default
    value = data[key]
    if getattr(value, "shape", None) == ():
        return value.item()
    return value


def parse_rounds(value: str | int | None) -> list[int]:
    if value is None or value == "" or str(value).lower() == "both":
        return sorted(RETRAIN_ROUNDS)
    round_num = int(value)
    if round_num not in RETRAIN_ROUNDS:
        raise ValueError(f"Unknown retrain round: {value}")
    return [round_num]


def load_contract_round(ticker: str, round_num: int) -> tuple[ContractArrays, dict]:
    ticker = ticker.upper()
    path = contract_data_path(round_num, ticker)
    if not path.exists():
        raise FileNotFoundError(
            f"No prepared shared feature data found at {path}. "
            "Run python drl_shared/prepare_features.py first."
        )
    data = np.load(path, allow_pickle=True)
    expected = feature_spec()
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


def _asset_tickers_from_index(asset_name: str, round_num: int) -> list[str]:
    index_path = asset_index_path(asset_name, round_num)
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return [str(t).upper() for t in payload.get("member_tickers", [])]
    policy = current_source_policy()
    excluded = set(policy["excluded_contracts"])
    return [t for t in universe_tickers(asset_name) if t.upper() not in excluded]


def _validate_feature_policy(feature_meta: dict, ticker: str):
    expected_policy = current_source_policy()
    resolved_preset = feature_meta.get("preset") or expected_policy["preset"]
    if resolved_preset != expected_policy["preset"]:
        raise ValueError(
            f"Mainline DQN requires preset={expected_policy['preset']!r}, got {resolved_preset!r}"
        )
    if feature_meta.get("source_overrides") != expected_policy["source_overrides"]:
        raise ValueError(f"Feature artifact source_overrides do not match structural_38 for {ticker}")
    if sorted(feature_meta.get("excluded_contracts", [])) != sorted(expected_policy["excluded_contracts"]):
        raise ValueError(f"Feature artifact excluded_contracts do not match structural_38 for {ticker}")


# (validation split removed)


def _run_training_episode(env: ContractEnv, agent: DQNAgent, global_step: int) -> tuple[float, int, list[float], int]:
    state = env.reset()
    total_reward = 0.0
    losses: list[float] = []
    done = False
    steps = 0
    while not done and steps < MAX_STEPS_PER_EP:
        eps = agent.epsilon_for_step(global_step)
        action_id = agent.act(state, eps)
        next_state, reward, done = env.step(action_id)
        agent.push(state, action_id, reward, next_state, float(done))
        loss = agent.learn()
        if loss > 0:
            losses.append(loss)
        state = next_state
        total_reward += reward
        steps += 1
        global_step += 1
    return float(total_reward), steps, losses, global_step


# (_validation_reward removed — no validation in paper)


def train_asset_round(
    asset_name: str,
    round_num: int,
    episodes: int = EPISODES,
    device: str | None = None,
    seed: int | None = None,
    sigma_tgt: float = SIGMA_TGT_DEFAULT,
    resume: bool = False,
) -> tuple[Path, Path]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    tickers = _asset_tickers_from_index(asset_name, round_num)
    if not tickers:
        raise ValueError(f"No eligible contracts found for {asset_name} {round_name(round_num)}")

    contracts: dict[str, ContractArrays] = {}
    envs: dict[str, ContractEnv] = {}
    feature_meta_by_ticker: dict[str, dict] = {}
    skipped: list[str] = []

    for ticker in tickers:
        try:
            contract, feature_meta = load_contract_round(ticker, round_num)
            _validate_feature_policy(feature_meta, ticker)
            contracts[ticker] = contract
            feature_meta_by_ticker[ticker] = feature_meta
            envs[ticker] = ContractEnv(contract, sigma_tgt=sigma_tgt)
        except Exception as exc:
            skipped.append(f"{ticker}: {exc}")

    if not contracts:
        raise ValueError(f"No loadable contracts for {asset_name} {round_name(round_num)}")

    agent = DQNAgent(device=device)
    if resume:
        # Find latest checkpoint for this asset+round (look in parent dir of bundles)
        asset_model_dir = Path(MODEL_ROOT) / asset_slug(asset_name) / round_name(round_num)
        if asset_model_dir.exists():
            existing_bundles = sorted(
                [d for d in asset_model_dir.iterdir() if d.is_dir() and (d / "checkpoint.pt").exists()],
                key=lambda d: d.name, reverse=True,
            )
            for prev_bundle in existing_bundles:
                ckpt_path = prev_bundle / "checkpoint.pt"
                agent.load(ckpt_path, resume=True)
                print(f"Resumed from {ckpt_path} (train_steps={agent.train_steps}, replay={len(agent.replay)})")
                break
    run_id = make_run_id()
    bundle_dir = model_bundle_root(round_num, asset_name, run_id=run_id)
    logger = RunLogger("dqn", asset_name, round_num, run_id=run_id, base_dir=bundle_dir)
    round_info = RETRAIN_ROUNDS[round_num]
    f_spec = feature_spec()
    metadata = checkpoint_metadata(
        round_num,
        asset_name,
        algorithm="dqn",
        extra={
            "run_id": logger.run_id,
            "cycles": episodes,
            "episodes_per_cycle": len(contracts),
            "total_planned_contract_episodes": episodes * len(contracts),
            "device": agent.device,
            "log_dir": str(logger.dir),
            "bundle_dir": str(bundle_dir),
            "seed": seed,
            "sigma_tgt": sigma_tgt,
            "member_tickers": list(contracts),
            "loaded_contracts": list(contracts),
            "skipped_contracts": skipped,
            "asset_class_count": len(contracts),
            "feature_artifact_paths": {
                ticker: meta["feature_artifact_path"] for ticker, meta in feature_meta_by_ticker.items()
            },
            "state_spec_version": f_spec["state_spec_version"],
        },
    )
    logger.write_json("manifest.json", metadata)
    logger.write_json("train_config.json", metadata["hyperparameters"] | {
        "cycles": episodes,
        "seed": seed,
    })
    logger.write_json("feature_spec.json", metadata["feature_spec"])

    logger.log(f"{'=' * 70}")
    logger.log(f"DQN Asset-Class Training — {asset_name} — {round_name(round_num)}")
    logger.log(f"Train: {round_info['train_start']} ~ {round_info['train_end']}")
    logger.log(f"Test : {round_info['test_start']} ~ {round_info['test_end']}")
    logger.log(f"Contracts: {len(contracts)}/{len(tickers)} loaded | Device: {agent.device}")
    if skipped:
        logger.log(f"Skipped: {skipped}")
    logger.log(f"Model bundle: {bundle_dir}")
    logger.log(f"State spec: {metadata['state_spec_version']}")
    logger.log(
        "DQN stabilizers [49]/[18]/[50]: fixed Q-targets, Double DQN, "
        "Dueling DQN; target hard-copy every 1000 learn steps"
    )
    logger.log(f"{'=' * 70}")

    t0 = time.time()
    report_interval = max(1, episodes // 10)
    global_step = agent.train_steps if resume else 0
    global_episode = 0
    start_cycle = 1
    episode_rows: list[dict] = []
    contract_stats = defaultdict(lambda: {
        "episodes_seen": 0,
        "transitions_added": 0,
        "reward_sum": 0.0,
        "loss_sum": 0.0,
        "loss_count": 0,
        "last_reward": 0.0,
        "last_loss": 0.0,
    })

    ordered_tickers = list(contracts)

    for cycle in range(1, episodes + 1):
        random.shuffle(ordered_tickers)
        cycle_rewards = []
        cycle_losses = []
        for ticker in ordered_tickers:
            global_episode += 1
            reward, steps, losses, global_step = _run_training_episode(envs[ticker], agent, global_step)
            mean_loss = float(np.mean(losses)) if losses else 0.0
            last_loss = float(losses[-1]) if losses else 0.0
            cycle_rewards.append(reward)
            cycle_losses.extend(losses)

            stats = contract_stats[ticker]
            stats["episodes_seen"] += 1
            stats["transitions_added"] += steps
            stats["reward_sum"] += reward
            stats["loss_sum"] += sum(losses)
            stats["loss_count"] += len(losses)
            stats["last_reward"] = reward
            stats["last_loss"] = last_loss

            row = {
                "episode": global_episode,
                "cycle": cycle,
                "round": round_num,
                "ticker": ticker,
                "reward": round(reward, 6),
                "steps": steps,
                "epsilon_end": round(agent.epsilon_for_step(global_step), 6),
                "mean_loss": round(mean_loss, 6),
                "last_loss": round(last_loss, 6),
                "learn_steps": agent.train_steps,
                "target_updates": agent.target_updates,
                "replay_size": len(agent.replay),
            }
            episode_rows.append(row)

        if cycle % report_interval == 0 or cycle == 1:
            elapsed = time.time() - t0
            avg_loss = float(np.mean(cycle_losses)) if cycle_losses else 0.0
            eta = (elapsed / cycle) * max(0, episodes - cycle) if cycle > 0 else 0.0
            logger.log(
                f"cycle {cycle}/{episodes} reward_avg={np.mean(cycle_rewards):+.4f} "
                f"loss={avg_loss:.6f} "
                f"epsilon={agent.epsilon_for_step(global_step):.4f} replay={len(agent.replay)} "
                f"target_updates={agent.target_updates} coverage={len(contracts)}/{len(tickers)} "
                f"elapsed={elapsed:.0f}s eta={eta:.0f}s"
            )

    contract_rows = []
    for ticker, stats in sorted(contract_stats.items()):
        loss_count = max(1, stats["loss_count"])
        episodes_seen = max(1, stats["episodes_seen"])
        contract_rows.append({
            "ticker": ticker,
            "round": round_num,
            "episodes_seen": stats["episodes_seen"],
            "transitions_added": stats["transitions_added"],
            "avg_reward": round(stats["reward_sum"] / episodes_seen, 6),
            "avg_loss": round(stats["loss_sum"] / loss_count, 6),
            "last_reward": round(stats["last_reward"], 6),
            "last_loss": round(stats["last_loss"], 6),
        })

    metadata["completed_cycles"] = episodes
    metadata["learn_steps"] = agent.train_steps
    metadata["target_updates"] = agent.target_updates
    out_path = checkpoint_path_for_bundle(bundle_dir)
    agent.save(out_path, metadata=metadata, include_training_state=True)
    logger.write_csv("episode_metrics.csv", episode_rows)
    logger.write_csv("contract_metrics.csv", contract_rows)
    logger.write_json("checkpoint_metadata.json", metadata)
    logger.write_json("manifest.json", metadata)
    logger.log(f"Saved checkpoint: {out_path}")
    logger.log(f"Saved logs: {logger.dir}")
    return out_path, logger.dir


def train_contract_round(*args, **kwargs):
    raise RuntimeError(
        "Mainline DQN training is asset-class based. "
        "Use train_asset_round(asset_name, round_num, ...) or the CLI with --asset."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="Forex", help='Asset universe or "All"')
    parser.add_argument("--round", default="both", help='1, 2, or "both" (default)')
    parser.add_argument("--episodes", type=int, default=EPISODES, help="Training cycles; each cycle visits every contract once.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sigma-tgt", type=float, default=SIGMA_TGT_DEFAULT)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint for each round")
    args = parser.parse_args()

    rounds = parse_rounds(args.round)
    assets = list(ASSET_CLASSES) if args.asset in (None, "", "All") else [args.asset]
    for round_num in rounds:
        for asset_name in assets:
            train_asset_round(
                asset_name,
                round_num,
                episodes=args.episodes,
                device=args.device,
                seed=args.seed,
                sigma_tgt=args.sigma_tgt,
                resume=args.resume,
            )


if __name__ == "__main__":
    main()
