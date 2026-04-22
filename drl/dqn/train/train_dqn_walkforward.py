#!/usr/bin/env python3
"""Single-contract walk-forward DQN training."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drl.dqn.logging_utils import RunLogger
from drl.dqn.model import DEVICE, DQNAgent
from drl.dqn.spec import (
    EPISODES,
    MAX_STEPS_PER_EP,
    RETRAIN_ROUNDS,
    checkpoint_metadata,
    contract_data_path,
    contract_model_path,
    round_name,
)
from drl_shared.state_space import ContractArrays, ContractEnv


def load_contract_round(ticker: str, round_num: int) -> ContractArrays:
    ticker = ticker.upper()
    path = contract_data_path(round_num, ticker)
    if not path.exists():
        raise FileNotFoundError(
            f"No prepared shared feature data found at {path}. "
            "Run python drl_shared/prepare_features.py first."
        )
    data = np.load(path, allow_pickle=True)
    return ContractArrays(
        ticker=ticker,
        prices=data["prices"],
        returns=data["returns"],
        sigma=data["sigma"],
        features=data["features"],
        dates=data["dates"],
        source=str(data["source"]),
    )


def train_contract_round(ticker: str, round_num: int, episodes: int = EPISODES) -> tuple[Path, Path]:
    ticker = ticker.upper()
    round_info = RETRAIN_ROUNDS[round_num]
    contract = load_contract_round(ticker, round_num)
    env = ContractEnv(contract)
    agent = DQNAgent()
    logger = RunLogger("dqn", ticker, round_num)
    metadata = checkpoint_metadata(
        round_num,
        ticker,
        algorithm="dqn",
        extra={
            "run_id": logger.run_id,
            "episodes": episodes,
            "device": DEVICE,
            "log_dir": str(logger.dir),
        },
    )
    logger.write_json("config.json", metadata)

    t0 = time.time()
    report_interval = max(1, episodes // 10)
    global_step = 0
    episode_rows = []
    logger.log(f"{'=' * 70}")
    logger.log(f"DQN Training — {ticker} — {round_name(round_num)}")
    logger.log(f"Train: {round_info['train_start']} ~ {round_info['train_end']}")
    logger.log(f"Test : {round_info['test_start']} ~ {round_info['test_end']}")
    logger.log(f"Source: {contract.source} | Train days: {len(contract.prices)} | Device: {DEVICE}")
    logger.log("Architecture: LSTM[64,32] + Leaky-ReLU + Fixed Q-targets + Double DQN + Dueling DQN")
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

        if (ep + 1) % report_interval == 0:
            elapsed = time.time() - t0
            logger.log(
                f"ep {ep + 1}/{episodes} reward={total_reward:+.4f} "
                f"steps={steps} epsilon={row['epsilon_end']:.4f} "
                f"loss={last_loss:.6f} ({elapsed:.0f}s)"
            )

    out_path = contract_model_path(round_num, ticker, algorithm="dqn")
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
    args = parser.parse_args()

    train_contract_round(args.ticker, args.round, args.episodes)
