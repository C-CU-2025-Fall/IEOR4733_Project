#!/usr/bin/env python3
"""Per-contract DQN backtest: load each contract's own model, run inference, compute metrics."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from drl.dqn.model import DQNAgent
from drl.dqn.spec import SEQ_LEN, SIGMA_TGT, MODEL_ROOT
from drl_shared.spec import universe_tickers, current_source_policy
from drl_shared.state_space import ContractArrays, ContractEnv, WARMUP
from metrics import METRIC_NAMES
from strategy_backtester import paper_table3_reference


def find_best_per_contract_model(ticker: str, round_num: int) -> Path | None:
    """Find the best per-contract model via best_seed.json, fallback to latest per_* dir."""
    base = MODEL_ROOT / ticker / f"r{round_num}"
    if not base.exists():
        return None
    # Prefer best_seed.json
    best_json = base / "best_seed.json"
    if best_json.exists():
        with open(best_json) as f:
            info = json.load(f)
        best_dir = info.get("best_model_dir")
        if best_dir and Path(best_dir).exists():
            return Path(best_dir)
    # Fallback: latest per_* dir
    per_dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("per_")])
    return per_dirs[-1] if per_dirs else None


def backtest_single_contract(ticker: str, round_num: int, device: str = "auto") -> dict:
    """Run DQN backtest for one contract with its own model."""
    # Find model
    model_dir = find_best_per_contract_model(ticker, round_num)
    if model_dir is None:
        return {"ticker": ticker, "round": round_num, "error": "no model found"}

    checkpoint_path = model_dir / "checkpoint.pt"
    manifest_path = model_dir / "manifest.json"
    if not checkpoint_path.exists():
        return {"ticker": ticker, "round": round_num, "error": "no checkpoint.pt"}

    # Load manifest for metadata
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    sigma_tgt = manifest.get("sigma_tgt", 0.058)

    # Load agent
    agent = DQNAgent(device=device)
    agent.load(checkpoint_path)

    # Load features data
    from drl.dqn.spec import contract_data_path
    feat_path = contract_data_path(round_num, ticker)
    if not feat_path.exists():
        return {"ticker": ticker, "round": round_num, "error": f"no features at {feat_path}"}

    data = np.load(feat_path, allow_pickle=True)
    contract = ContractArrays(
        ticker=ticker,
        prices=data["prices"],
        returns=data["returns"],
        sigma=data["sigma"],
        features=data["features"],
        dates=data["dates"],
        source=str(data.get("source", "")),
    )

    # Run episode (greedy, eps=0)
    env = ContractEnv(contract, sigma_tgt=sigma_tgt)
    state = env.reset()
    rewards = []
    done = False
    steps = 0
    while not done and steps < 5000:
        action_id = agent.act(state, eps=0.0)  # greedy
        next_state, reward, done = env.step(action_id)
        rewards.append(reward)
        state = next_state
        steps += 1

    rewards = np.array(rewards)
    cum_return = np.sum(rewards)

    return {
        "ticker": ticker,
        "round": round_num,
        "steps": steps,
        "cum_return": float(cum_return),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "positive_pct": float(np.mean(rewards > 0) * 100),
        "sharpe": float(np.mean(rewards) / (np.std(rewards) + 1e-10)),
        "mdd": float(compute_mdd(rewards)),
        "model_dir": str(model_dir),
        "sigma_tgt": sigma_tgt,
    }


def compute_mdd(rewards: np.ndarray) -> float:
    """Max drawdown from reward series."""
    cum = np.cumsum(rewards)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(np.min(dd))


def main():
    parser = argparse.ArgumentParser(description="Per-contract DQN backtest")
    parser.add_argument("--asset", default="Forex")
    parser.add_argument("--round", type=int, default=None, help="Round number (default: both)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ticker", default=None, help="Single ticker override")
    args = parser.parse_args()

    # Resolve tickers
    if args.ticker:
        tickers = [args.ticker.upper()]
    else:
        tickers = universe_tickers(args.asset)

    policy = current_source_policy()
    excluded = set(policy.get("excluded_contracts", []))
    tickers = [t for t in tickers if t.upper() not in excluded]

    rounds = [args.round] if args.round else sorted(set(range(1, 3)))

    print(f"Per-contract DQN Backtest: {len(tickers)} tickers × {len(rounds)} rounds")
    print(f"{'Ticker':>6s} {'Round':>5s} {'Steps':>6s} {'CumR':>8s} {'MeanR':>8s} {'Sharpe':>8s} {'MDD':>8s} {'%+ve':>6s}")
    print("-" * 65)

    results = []
    for t in tickers:
        for r in rounds:
            res = backtest_single_contract(t, r, device=args.device)
            results.append(res)
            if "error" in res:
                print(f"{t:>6s} r{r:>4d}  ERROR: {res['error']}")
            else:
                print(f"{t:>6s} r{r:>4d} {res['steps']:>6d} {res['cum_return']:>+8.4f} {res['mean_reward']:>+8.5f} {res['sharpe']:>+8.3f} {res['mdd']:>+8.4f} {res['positive_pct']:>5.1f}%")

    # Summary
    ok_results = [r for r in results if "error" not in r]
    if ok_results:
        avg_cum = np.mean([r["cum_return"] for r in ok_results])
        avg_sharpe = np.mean([r["sharpe"] for r in ok_results])
        print(f"\n{'='*65}")
        print(f"Average across {len(ok_results)} models: CumR={avg_cum:+.4f} Sharpe={avg_sharpe:+.3f}")

    # Save JSON
    out_path = REPO / f"per_contract_backtest_{'_'.join(t.lower() for t in tickers[:3])}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
