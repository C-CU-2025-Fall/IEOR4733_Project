#!/usr/bin/env python3
"""Shared-model walk-forward DQN training."""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dqn.model import DEVICE, SharedDQNAgent
from dqn.pipeline import ContractArrays, ContractEnv
from dqn.spec import EPISODES, MAX_STEPS_PER_EP, SHARED_ROUNDS, checkpoint_metadata, round_data_dir, round_model_path, round_name, universe_tickers


def load_round_contracts(asset_name: str, round_num: int) -> list[ContractArrays]:
    tickers = universe_tickers(asset_name)
    data_dir = round_data_dir(round_num, asset_name)
    contracts: list[ContractArrays] = []
    for ticker in tickers:
        path = data_dir / f"{ticker}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        contracts.append(
            ContractArrays(
                ticker=ticker,
                prices=data["prices"],
                returns=data["returns"],
                sigma=data["sigma"],
                features=data["features"],
                dates=data["dates"],
                source=str(data["source"]),
            )
        )
    return contracts


def train_round_shared(asset_name: str, round_num: int, episodes: int = EPISODES) -> Path:
    contracts = load_round_contracts(asset_name, round_num)
    if not contracts:
        raise FileNotFoundError(
            f"No prepared shared DQN data found in {round_data_dir(round_num, asset_name)}. "
            "Run prepare_dqn_walkforward.py first."
        )

    envs = [ContractEnv(contract) for contract in contracts]
    agent = SharedDQNAgent()
    t0 = time.time()
    report_interval = max(1, episodes // 10)
    global_step = 0

    print(f"\n{'=' * 70}")
    print(f"Shared DQN Training — {asset_name} — {round_name(round_num)}")
    print(f"Contracts: {len(contracts)} | Episodes: {episodes} | Device: {DEVICE}")
    print(f"{'=' * 70}")

    for ep in range(episodes):
        env = random.choice(envs)
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        while not done and steps < MAX_STEPS_PER_EP:
            eps = agent.epsilon_for_step(global_step)
            action_id = agent.act(state, eps)
            next_state, reward, done = env.step(action_id)
            agent.push(state, action_id, reward, next_state, float(done))
            agent.learn()
            state = next_state
            total_reward += reward
            steps += 1
            global_step += 1

        if (ep + 1) % report_interval == 0:
            elapsed = time.time() - t0
            print(
                f"ep {ep + 1}/{episodes} reward={total_reward:+.4f} "
                f"steps={steps} ({elapsed:.0f}s)"
            )

    metadata = checkpoint_metadata(
        round_num,
        asset_name,
        extra={"contracts_seen": [contract.ticker for contract in contracts], "episodes": episodes},
    )
    out_path = round_model_path(round_num, asset_name)
    agent.save(out_path, metadata=metadata)
    print(f"\nSaved shared checkpoint: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True, choices=sorted(SHARED_ROUNDS))
    parser.add_argument("--asset", default="Forex", help='Asset universe or "All"')
    parser.add_argument("--episodes", type=int, default=EPISODES)
    args = parser.parse_args()

    train_round_shared(args.asset, args.round, args.episodes)
