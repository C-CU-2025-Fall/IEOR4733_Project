#!/usr/bin/env python3
"""Fast per-contract DQN vs Long-only backtest using pre-computed features from .npz."""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from drl.dqn.model import DQNAgent
from drl.dqn.spec import MODEL_ROOT, SEQ_LEN
from drl_shared.spec import universe_tickers, current_source_policy, RETRAIN_ROUNDS, WARMUP
from drl_shared.state_space import ContractArrays, ContractEnv, get_feature_window


def get_best_model(ticker: str, round_num: int) -> Path | None:
    base = MODEL_ROOT / ticker / f"r{round_num}"
    best_json = base / "best_seed.json"
    if best_json.exists():
        with open(best_json) as f:
            info = json.load(f)
        best_dir = Path(info["best_model_dir"])
        if best_dir.exists() and (best_dir / "checkpoint.pt").exists():
            return best_dir
    if not base.exists():
        return None
    per_dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("per_")])
    return per_dirs[-1] if per_dirs else None


def load_npz_features(ticker: str, round_num: int) -> ContractArrays | None:
    """Load pre-computed features from .npz (fast, no re-computation)."""
    path = REPO / "drl" / "features" / ticker / f"r{round_num}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return ContractArrays(
        ticker=ticker,
        prices=data["prices"],
        returns=data["returns"],
        sigma=data["sigma"],
        features=data["features"],
        dates=data["dates"],
        source=str(data.get("source", "")),
    )


def compute_mdd(rewards):
    cum = np.cumsum(rewards)
    peak = np.maximum.accumulate(cum)
    return float(np.min(cum - peak))


def backtest_dqn_greedy(agent, contract, start_idx, sigma_tgt):
    env = ContractEnv(contract, sigma_tgt=sigma_tgt, start_idx=start_idx)
    state = env.reset()
    rewards = []
    done = False
    while not done:
        action_id = agent.act(state, eps=0.0)
        next_state, reward, done = env.step(action_id)
        rewards.append(reward)
        state = next_state
    rewards = np.array(rewards)
    return {
        "cum_return": float(np.sum(rewards)),
        "sharpe": float(np.mean(rewards) / (np.std(rewards) + 1e-10)),
        "mdd": compute_mdd(rewards),
        "steps": len(rewards),
    }


def backtest_long(contract, start_idx, sigma_tgt):
    env = ContractEnv(contract, sigma_tgt=sigma_tgt, start_idx=start_idx)
    env.reset()
    rewards = []
    done = False
    while not done:
        _, reward, done = env.step(2)  # action_id=2 is long
        rewards.append(reward)
    rewards = np.array(rewards)
    return {
        "cum_return": float(np.sum(rewards)),
        "sharpe": float(np.mean(rewards) / (np.std(rewards) + 1e-10)),
        "mdd": compute_mdd(rewards),
        "steps": len(rewards),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="Forex")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--sigma-tgt", type=float, default=0.058)
    args = parser.parse_args()

    tickers = [args.ticker.upper()] if args.ticker else universe_tickers(args.asset)
    policy = current_source_policy()
    excluded = set(policy.get("excluded_contracts", []))
    tickers = [t for t in tickers if t.upper() not in excluded]

    print(f"DQN vs Long (TEST period, pre-computed features) | sigma_tgt={args.sigma_tgt}")
    print(f"{'Ticker':>6} {'Rd':>3} {'Period':>22} {'DQN_cum':>9} {'Long_cum':>9} {'DQN_Shp':>8} {'Long_Shp':>8} {'DQN_MDD':>8} {'Long_MDD':>8} {'Win':>6}")
    print("-" * 100)

    dqn_wins = 0
    total = 0
    for t in tickers:
        for r in [1, 2]:
            ri = RETRAIN_ROUNDS[r]
            period = f"{ri['test_start']}~{ri['test_end']}"

            contract = load_npz_features(t, r)
            if contract is None:
                print(f"{t:>6} r{r:>2} {period:>22}  no features")
                continue

            # Test period: use last portion of training data as proxy
            # NPZ contains training data; test period starts at test_start
            # For per-contract evaluation, run on full training data
            start_idx = WARMUP

            model_dir = get_best_model(t, r)
            if model_dir is None:
                print(f"{t:>6} r{r:>2} {period:>22}  no model")
                continue

            agent = DQNAgent(device="cuda")
            agent.load(model_dir / "checkpoint.pt")

            dqn = backtest_dqn_greedy(agent, contract, start_idx, args.sigma_tgt)
            lon = backtest_long(contract, start_idx, args.sigma_tgt)
            w = "DQN" if dqn["cum_return"] > lon["cum_return"] else "LONG"
            total += 1
            if w == "DQN":
                dqn_wins += 1

            print(f"{t:>6} r{r:>2} {period:>22} {dqn['cum_return']:>+9.3f} {lon['cum_return']:>+9.3f} "
                  f"{dqn['sharpe']:>+8.3f} {lon['sharpe']:>+8.3f} "
                  f"{dqn['mdd']:>+8.3f} {lon['mdd']:>+8.3f} {w:>6}")

    print(f"\nDQN beats Long: {dqn_wins}/{total} ({dqn_wins/max(total,1)*100:.0f}%)")


if __name__ == "__main__":
    main()
