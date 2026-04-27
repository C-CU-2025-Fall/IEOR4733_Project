#!/usr/bin/env python3
"""Train a single contract DQN model (called by parallel scheduler)."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from config import BP
from drl.dqn.model import DQNAgent
from drl.dqn.spec import (
    BATCH_SIZE,
    DISCRETE_ACTION_DIM,
    DROPOUT,
    DQN_SPEC_VERSION,
    EPS_DECAY_STEPS,
    EPS_END,
    EPS_START,
    FEATURE_DIM,
    GAMMA,
    LEAKY_RELU_SLOPE,
    LR,
    LSTM_HIDDEN_SIZES,
    MAX_STEPS_PER_EP,
    MEMORY_SIZE,
    MODEL_ROOT,
    SEQ_LEN,
    TAU,
    contract_data_path,
)
from drl_shared.spec import (
    SIGMA_TGT_DEFAULT,
    SEQ_LEN as SHARED_SEQ_LEN,
    current_source_policy,
    feature_spec,
    universe_tickers,
    ticker_asset_class,
)
from drl_shared.state_space import (
    WARMUP,
    ContractArrays,
    ContractEnv,
    action_id_to_position,
    compute_eq4_reward,
    get_feature_window,
)


def load_contract(ticker: str, round_num: int) -> tuple[ContractArrays, dict]:
    path = contract_data_path(round_num, ticker)
    if not path.exists():
        raise FileNotFoundError(f"No features for {ticker} r{round_num}: {path}")
    data = np.load(path, allow_pickle=True)
    meta = {
        "feature_artifact_path": str(path),
        "state_spec_version": data.get("state_spec_version", "").item() if hasattr(data.get("state_spec_version"), "item") else str(data.get("state_spec_version", "")),
    }
    contract = ContractArrays(
        ticker=ticker,
        prices=data["prices"],
        returns=data["returns"],
        sigma=data["sigma"],
        features=data["features"],
        dates=data["dates"],
        source=str(data.get("source", "").item() if hasattr(data.get("source", ""), "item") else data.get("source", "")),
    )
    return contract, meta


import sys as _sys
def _log(msg: str):
    print(msg, flush=True)


def train_single(ticker: str, round_num: int, episodes: int, device: str, sigma_tgt: float, early_stop_patience: int = 0):
    _log(f"  {ticker} r{round_num} START episodes={episodes} sigma_tgt={sigma_tgt} early_stop={early_stop_patience} device={device}")
    contract, meta = load_contract(ticker, round_num)
    n = len(contract.prices)
    _log(f"  {ticker} r{round_num} data: {n} steps, dates {contract.dates[0]} → {contract.dates[-1]}")

    # 10% chronological validation split (paper: JFDS 2020)
    val_split = 0.10
    n_train = int(n * (1 - val_split))
    _log(f"  {ticker} r{round_num} split: train [0, {n_train}), val [{n_train}, {n})")
    train_env = ContractEnv(contract, sigma_tgt=sigma_tgt, max_idx=n_train - 1)
    val_env = ContractEnv(contract, sigma_tgt=sigma_tgt, max_idx=n - 1, start_idx=n_train)

    agent = DQNAgent(device=device)
    global_step = 0
    rewards_log = []
    losses_log = []
    val_rewards_log = []

    # Early stopping state (based on validation reward)
    best_val_reward = None
    best_agent_state = None
    no_improve_count = 0

    for ep in range(episodes):
        # --- Train ---
        state = train_env.reset()
        total_reward = 0.0
        ep_losses = []
        done = False
        steps = 0
        while not done and steps < MAX_STEPS_PER_EP:
            eps = agent.epsilon_for_step(global_step)
            action_id = agent.act(state, eps)
            next_state, reward, done = train_env.step(action_id)
            agent.push(state, action_id, reward, next_state, float(done))
            loss = agent.learn()
            if loss > 0:
                ep_losses.append(loss)
            state = next_state
            total_reward += reward
            steps += 1
            global_step += 1

        avg_loss = np.mean(ep_losses) if ep_losses else 0.0
        rewards_log.append(total_reward)
        losses_log.append(avg_loss)

        # --- Validate (greedy, no exploration) ---
        if early_stop_patience > 0:
            v_state = val_env.reset()
            val_reward = 0.0
            v_done = False
            v_steps = 0
            while not v_done and v_steps < MAX_STEPS_PER_EP:
                v_action = agent.act(v_state, eps=0.0)  # greedy
                v_state, v_r, v_done = val_env.step(v_action)
                val_reward += v_r
                v_steps += 1
            val_rewards_log.append(val_reward)

            if best_val_reward is None or val_reward > best_val_reward:
                best_val_reward = val_reward
                no_improve_count = 0
                import torch
                best_agent_state = {k: v.cpu().clone() for k, v in agent.q_net.state_dict().items()}
            else:
                no_improve_count += 1
            if no_improve_count >= early_stop_patience:
                _log(f"  {ticker} r{round_num} EARLY STOP @ ep{ep+1} (best_val={best_val_reward:.4f}, no improve {no_improve_count} eps)")
                if best_agent_state is not None:
                    agent.q_net.load_state_dict(best_agent_state)
                break

        if (ep + 1) % 20 == 0:
            val_str = f" val={val_rewards_log[-1]:.4f}" if val_rewards_log else ""
            _log(f"  {ticker} r{round_num} ep{ep+1}/{episodes} reward={total_reward:.4f} loss={avg_loss:.4f} eps={agent.epsilon_for_step(global_step):.3f}{val_str}")

    # Save to per-contract path: models/<TICKER>/r<round>/<run_id>/checkpoint.pt
    from drl.dqn.logging_utils import make_run_id
    run_id = "per_" + make_run_id()
    asset_class = ticker_asset_class(ticker)
    save_dir = MODEL_ROOT / ticker / f"r{round_num}" / run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    agent.save(save_dir / "checkpoint.pt", include_training_state=False, metadata={
        "ticker": ticker,
        "round": round_num,
        "episodes": episodes,
        "global_step": global_step,
        "device": device,
        "sigma_tgt": sigma_tgt,
        "final_avg_reward_10": float(np.mean(rewards_log[-10:])) if rewards_log else 0.0,
        "training_mode": "per_contract",
        "asset_class": asset_class,
    })

    # Save manifest
    manifest = {
        "ticker": ticker,
        "round": round_num,
        "episodes": episodes,
        "device": device,
        "sigma_tgt": sigma_tgt,
        "training_mode": "per_contract",
        "asset_class": asset_class,
        "final_avg_reward_10": float(np.mean(rewards_log[-10:])) if rewards_log else 0.0,
        "learning_curve": {"rewards": rewards_log, "losses": losses_log},
    }
    with open(save_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    _log(f"  {ticker} r{round_num} DONE: {ep} eps, avg10={np.mean(rewards_log[-10:]):.4f}, saved → {save_dir}")
    return save_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sigma-tgt", type=float, default=0.0063)
    parser.add_argument("--early-stop", type=int, default=0, help="Early stopping patience (0=disabled)")
    args = parser.parse_args()
    train_single(args.ticker.upper(), args.round, args.episodes, args.device, args.sigma_tgt, early_stop_patience=args.early_stop)


if __name__ == "__main__":
    main()
