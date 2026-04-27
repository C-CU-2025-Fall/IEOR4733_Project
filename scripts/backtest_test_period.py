#!/usr/bin/env python3
"""Per-contract DQN vs Long-only backtest on TEST periods.

r1 model → tested on 2011-2015
r2 model → tested on 2016-2019

Usage:
    python3 scripts/backtest_test_period.py --asset Forex
    python3 scripts/backtest_test_period.py --ticker AN
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from config import BP
from drl.dqn.model import DQNAgent
from drl.dqn.spec import SEQ_LEN, MODEL_ROOT
from drl_shared.spec import universe_tickers, current_source_policy, RETRAIN_ROUNDS, ticker_asset_class
from drl_shared.state_space import ContractArrays, ContractEnv, WARMUP, build_feature_matrix, compute_ewma_sigma
from drl_shared.spec import feature_spec
from data_loader import load_clc_full


def get_best_seed_model(ticker: str, round_num: int) -> Path | None:
    """Read best_seed.json to find the best model, or fall back to latest."""
    base = MODEL_ROOT / ticker / f"r{round_num}"
    best_json = base / "best_seed.json"
    if best_json.exists():
        with open(best_json) as f:
            info = json.load(f)
        best_dir = Path(info["best_model_dir"])
        if best_dir.exists() and (best_dir / "checkpoint.pt").exists():
            return best_dir
    # Fallback: latest per_* directory
    if not base.exists():
        return None
    per_dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("per_")])
    return per_dirs[-1] if per_dirs else None


def get_all_seed_models(ticker: str, round_num: int) -> list[Path]:
    """Return all seed model dirs for ensemble mode."""
    best_json = MODEL_ROOT / ticker / f"r{round_num}" / "best_seed.json"
    if best_json.exists():
        with open(best_json) as f:
            info = json.load(f)
        return [Path(s["dir"]) for s in info["all_seeds"] if Path(s["dir"]).exists()]
    return []


def load_test_data(ticker: str, round_num: int) -> tuple[ContractArrays, int] | None:
    """Load test-period data from pre-computed npz via shared data_loader."""
    from drl_shared.data_loader import get_test_slice
    try:
        contract, start_idx, _meta = get_test_slice(ticker, round_num)
        return contract, start_idx
    except Exception:
        return None


def compute_mdd(rewards):
    cum = np.cumsum(rewards)
    peak = np.maximum.accumulate(cum)
    return float(np.min(cum - peak))


def _run_greedy_episode(agent, env, sigma_tgt):
    """Run one greedy episode, return (rewards, positions)."""
    state = env.reset()
    rewards, positions = [], []
    done = False
    while not done:
        action_id = agent.act(state, eps=0.0)
        positions.append(float(action_id) - 1.0)
        next_state, reward, done = env.step(action_id)
        rewards.append(reward)
        state = next_state
    return np.array(rewards), np.array(positions)


def backtest_dqn(ticker, round_num, contract, start_idx, sigma_tgt=0.058, mode="best"):
    if mode == "ensemble":
        return backtest_dqn_ensemble(ticker, round_num, contract, start_idx, sigma_tgt)

    model_dir = get_best_seed_model(ticker, round_num)
    if model_dir is None:
        return None
    ckpt = model_dir / "checkpoint.pt"
    if not ckpt.exists():
        return None
    agent = DQNAgent(device="cpu")
    agent.load(ckpt)

    env = ContractEnv(contract, sigma_tgt=sigma_tgt, start_idx=start_idx)
    rewards, positions = _run_greedy_episode(agent, env, sigma_tgt)
    return {"rewards": rewards, "positions": positions,
            "cum_return": float(np.sum(rewards)),
            "sharpe": float(np.mean(rewards) / (np.std(rewards) + 1e-10)),
            "mdd": compute_mdd(rewards), "steps": len(rewards)}


def backtest_dqn_ensemble(ticker, round_num, contract, start_idx, sigma_tgt=0.058):
    """Ensemble: average Q-values across all seeds, pick argmax action."""
    import torch
    model_dirs = get_all_seed_models(ticker, round_num)
    if not model_dirs:
        return backtest_dqn(ticker, round_num, contract, start_idx, sigma_tgt, mode="best")

    agents = []
    for md in model_dirs:
        ckpt = md / "checkpoint.pt"
        if not ckpt.exists():
            continue
        ag = DQNAgent(device="cpu")
        ag.load(ckpt)
        agents.append(ag)
    if not agents:
        return None

    env = ContractEnv(contract, sigma_tgt=sigma_tgt, start_idx=start_idx)
    state = env.reset()
    rewards, positions = [], []
    done = False
    while not done:
        # Average Q-values across seeds
        q_sum = None
        for ag in agents:
            state_t = torch.from_numpy(state).unsqueeze(0).float()
            with torch.no_grad():
                q = ag.q_net(state_t)
            q_sum = q if q_sum is None else q_sum + q
        action_id = (q_sum / len(agents)).argmax().item()
        positions.append(float(action_id) - 1.0)
        next_state, reward, done = env.step(action_id)
        rewards.append(reward)
        state = next_state
    rewards = np.array(rewards)
    return {"rewards": rewards, "positions": np.array(positions),
            "cum_return": float(np.sum(rewards)),
            "sharpe": float(np.mean(rewards) / (np.std(rewards) + 1e-10)),
            "mdd": compute_mdd(rewards), "steps": len(rewards)}


def backtest_long_only(contract, start_idx, sigma_tgt=0.058):
    env = ContractEnv(contract, sigma_tgt=sigma_tgt, start_idx=start_idx)
    env.reset()
    rewards = []
    done = False
    while not done:
        _, reward, done = env.step(2)
        rewards.append(reward)
    rewards = np.array(rewards)
    return {"cum_return": float(np.sum(rewards)),
            "sharpe": float(np.mean(rewards) / (np.std(rewards) + 1e-10)),
            "mdd": compute_mdd(rewards), "steps": len(rewards)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="Forex")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--sigma-tgt", type=float, default=0.058)
    parser.add_argument("--mode", default="best", choices=["best", "ensemble"],
                        help="best=val-selected seed, ensemble=avg Q-values across all seeds")
    args = parser.parse_args()

    tickers = [args.ticker.upper()] if args.ticker else universe_tickers(args.asset)
    policy = current_source_policy()
    excluded = set(policy.get("excluded_contracts", []))
    tickers = [t for t in tickers if t.upper() not in excluded]

    print(f"DQN vs Long-only Backtest (TEST periods) | mode={args.mode}")
    print(f"{'Ticker':>6} {'Rd':>3} {'Period':>22s} {'DQN_cum':>9} {'Long_cum':>9} {'DQN_Shp':>8} {'Long_Shp':>8} {'DQN_MDD':>8} {'Long_MDD':>8} {'Winner':>6}")
    print("-" * 100)

    dqn_wins = 0
    total = 0
    for t in tickers:
        for r in [1, 2]:
            ri = RETRAIN_ROUNDS[r]
            period = f"{ri['test_start']}~{ri['test_end']}"
            result = load_test_data(t, r)
            if result is None:
                print(f"{t:>6} r{r:>2} {period:>22s}  ERROR: no data")
                continue
            contract, start_idx = result

            dqn = backtest_dqn(t, r, contract, start_idx, args.sigma_tgt, mode=args.mode)
            if dqn is None:
                print(f"{t:>6} r{r:>2} {period:>22s}  ERROR: no model")
                continue

            lon = backtest_long_only(contract, start_idx, args.sigma_tgt)
            w = "DQN" if dqn["cum_return"] > lon["cum_return"] else "LONG"
            total += 1
            if w == "DQN":
                dqn_wins += 1

            print(f"{t:>6} r{r:>2} {period:>22s} {dqn['cum_return']:>+9.4f} {lon['cum_return']:>+9.4f} "
                  f"{dqn['sharpe']:>+8.3f} {lon['sharpe']:>+8.3f} "
                  f"{dqn['mdd']:>+8.4f} {lon['mdd']:>+8.4f} {w:>6}")

    print(f"\nDQN beats Long-only: {dqn_wins}/{total} ({dqn_wins/max(total,1)*100:.0f}%)")


if __name__ == "__main__":
    main()
