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
from data_loader import load_clc_full


def get_latest_model(ticker: str, round_num: int) -> Path | None:
    base = MODEL_ROOT / ticker / f"r{round_num}"
    if not base.exists():
        return None
    per_dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("per_")])
    return per_dirs[-1] if per_dirs else None


def load_test_data(ticker: str, round_num: int) -> tuple[ContractArrays, int] | None:
    """Load test-period data with burn-in for feature computation."""
    from drl_shared.spec import current_source_policy
    ri = RETRAIN_ROUNDS[round_num]
    test_start = ri["test_start"]
    test_end = ri["test_end"]

    # Burn-in: load from 1 year before test_start
    burnin_dt = datetime.strptime(test_start, "%Y-%m-%d") - __import__("datetime").timedelta(days=400)
    burnin_start = burnin_dt.strftime("%Y-%m-%d")

    # Determine source from policy
    policy = current_source_policy()
    source = policy.get("source_overrides", {}).get(ticker, policy.get("default_source", "RAD"))

    df = load_clc_full(ticker, start_date=burnin_start, source=source)
    if df is None or len(df) == 0:
        return None

    prices = df["Close"].values.astype(np.float64)
    dates = df["Date"].values

    returns = np.diff(prices, prepend=prices[0])
    sigma = compute_ewma_sigma(returns)
    features = build_feature_matrix(prices, returns, sigma)

    # Find test_start index
    if hasattr(dates[0], 'strftime'):
        date_strs = [d.strftime("%Y-%m-%d") for d in dates]
    else:
        date_strs = [np.datetime_as_string(d, unit='D')[:10] for d in dates]

    test_start_idx = 0
    for i, ds in enumerate(date_strs):
        if ds >= test_start:
            test_start_idx = i
            break

    start_idx = max(WARMUP, test_start_idx)

    contract = ContractArrays(
        ticker=ticker, prices=prices, returns=returns,
        sigma=sigma, features=features, dates=dates, source="test_period",
    )
    return contract, start_idx


def compute_mdd(rewards):
    cum = np.cumsum(rewards)
    peak = np.maximum.accumulate(cum)
    return float(np.min(cum - peak))


def backtest_dqn(ticker, round_num, contract, start_idx, sigma_tgt=0.058):
    model_dir = get_latest_model(ticker, round_num)
    if model_dir is None:
        return None
    ckpt = model_dir / "checkpoint.pt"
    if not ckpt.exists():
        return None
    agent = DQNAgent(device="cpu")
    agent.load(ckpt)

    env = ContractEnv(contract, sigma_tgt=sigma_tgt, start_idx=start_idx)
    state = env.reset()
    rewards = []
    positions = []
    done = False
    while not done:
        action_id = agent.act(state, eps=0.0)
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
    args = parser.parse_args()

    tickers = [args.ticker.upper()] if args.ticker else universe_tickers(args.asset)
    policy = current_source_policy()
    excluded = set(policy.get("excluded_contracts", []))
    tickers = [t for t in tickers if t.upper() not in excluded]

    print(f"DQN vs Long-only Backtest (TEST periods)")
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

            dqn = backtest_dqn(t, r, contract, start_idx, args.sigma_tgt)
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
