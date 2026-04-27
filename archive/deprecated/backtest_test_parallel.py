#!/usr/bin/env python3
"""Parallel per-contract DQN vs Long-only backtest on TEST periods.

Uses test-period data (load_clc_full with burn-in) and runs inference.
Spawns N workers via subprocess for true CPU parallelism.

Usage:
    python3 scripts/backtest_test_parallel.py --asset Forex --workers 4
    python3 scripts/backtest_test_parallel.py --ticker AN
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from drl.dqn.model import DQNAgent
from drl.dqn.spec import MODEL_ROOT
from drl_shared.spec import universe_tickers, current_source_policy, RETRAIN_ROUNDS, WARMUP
from drl_shared.state_space import ContractArrays, ContractEnv, build_feature_matrix, compute_ewma_sigma
from data_loader import load_clc_full
from datetime import datetime, timedelta


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


def load_test_contract(ticker: str, round_num: int) -> tuple[ContractArrays, int] | None:
    """Load test-period data with burn-in for feature computation."""
    policy = current_source_policy()
    source = policy.get("source_overrides", {}).get(ticker, "RAD")
    ri = RETRAIN_ROUNDS[round_num]
    test_start = ri["test_start"]
    test_end = ri["test_end"]

    # Burn-in: 400 days before test_start
    burnin_dt = datetime.strptime(test_start, "%Y-%m-%d") - timedelta(days=400)
    burnin_start = burnin_dt.strftime("%Y-%m-%d")

    df = load_clc_full(ticker, start_date=burnin_start, source=source)
    if df is None or len(df) == 0:
        return None

    prices = df["Close"].values.astype(np.float64)
    returns = np.diff(prices, prepend=prices[0])
    sigma = compute_ewma_sigma(returns)
    features = build_feature_matrix(prices, returns, sigma)

    dates = df["Date"].values
    if hasattr(dates[0], 'strftime'):
        date_strs = [d.strftime("%Y-%m-%d") for d in dates]
    else:
        date_strs = [str(d)[:10] for d in dates]

    test_start_idx = 0
    for i, ds in enumerate(date_strs):
        if ds >= test_start:
            test_start_idx = i
            break

    start_idx = max(WARMUP, test_start_idx)

    contract = ContractArrays(
        ticker=ticker, prices=prices, returns=returns,
        sigma=sigma, features=features, dates=dates, source="test",
    )
    return contract, start_idx


def compute_mdd(rewards):
    cum = np.cumsum(rewards)
    peak = np.maximum.accumulate(cum)
    return float(np.min(cum - peak))


def run_one(ticker: str, round_num: int, sigma_tgt: float = 0.058) -> dict:
    """Run DQN + Long backtest for one contract-round on test period."""
    model_dir = get_best_model(ticker, round_num)
    if model_dir is None:
        return {"ticker": ticker, "round": round_num, "error": "no model"}

    result = load_test_contract(ticker, round_num)
    if result is None:
        return {"ticker": ticker, "round": round_num, "error": "no data"}

    contract, start_idx = result
    agent = DQNAgent(device="cuda")
    agent.load(model_dir / "checkpoint.pt")

    # DQN greedy
    env = ContractEnv(contract, sigma_tgt=sigma_tgt, start_idx=start_idx)
    state = env.reset()
    dqn_rewards = []
    done = False
    while not done:
        action_id = agent.act(state, eps=0.0)
        state, reward, done = env.step(action_id)
        dqn_rewards.append(reward)

    # Long only
    env2 = ContractEnv(contract, sigma_tgt=sigma_tgt, start_idx=start_idx)
    env2.reset()
    long_rewards = []
    done = False
    while not done:
        _, reward, done = env2.step(2)
        long_rewards.append(reward)

    dqn_r = np.array(dqn_rewards)
    lon_r = np.array(long_rewards)
    ri = RETRAIN_ROUNDS[round_num]

    return {
        "ticker": ticker,
        "round": round_num,
        "period": f"{ri['test_start']}~{ri['test_end']}",
        "steps": len(dqn_r),
        "dqn_cum": float(np.sum(dqn_r)),
        "long_cum": float(np.sum(lon_r)),
        "dqn_sharpe": float(np.mean(dqn_r) / (np.std(dqn_r) + 1e-10)),
        "long_sharpe": float(np.mean(lon_r) / (np.std(lon_r) + 1e-10)),
        "dqn_mdd": compute_mdd(dqn_r),
        "long_mdd": compute_mdd(lon_r),
        "dqn_pct_pos": float(np.mean(dqn_r > 0) * 100),
        "long_pct_pos": float(np.mean(lon_r > 0) * 100),
    }


def _worker_main():
    """Subprocess worker: reads JSON from stdin, writes result JSON to stdout."""
    import sys as _sys
    _sys.path.insert(0, str(REPO))
    task = json.loads(_sys.stdin.readline())
    result = run_one(task["ticker"], task["round"], task.get("sigma_tgt", 0.058))
    _sys.stdout.write(json.dumps(result) + "\n")
    _sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="Forex")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--sigma-tgt", type=float, default=0.058)
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--device", default="cuda", help="Device for inference")
    parser.add_argument("--worker-mode", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_mode:
        _worker_main()
        return

    tickers = [args.ticker.upper()] if args.ticker else universe_tickers(args.asset)
    policy = current_source_policy()
    excluded = set(policy.get("excluded_contracts", []))
    tickers = [t for t in tickers if t.upper() not in excluded]

    tasks = [(t, r) for t in tickers for r in [1, 2]]
    # Filter to only those with models
    pending = []
    for t, r in tasks:
        if get_best_model(t, r) is not None:
            pending.append((t, r))

    print(f"DQN vs Long TEST Period Backtest | {len(pending)} tasks | {args.workers} workers | {args.device}")
    print(f"{'Ticker':>6} {'Rd':>3} {'Period':>22} {'DQN_cum':>9} {'Long_cum':>9} {'DQN_Shp':>8} {'Long_Shp':>8} {'DQN_MDD':>8} {'Long_MDD':>8} {'Win':>6}")
    print("-" * 100)

    results = []
    running = []  # (proc, ticker, round, stdout_line_buffer)
    pending_q = list(pending)
    t0 = time.time()

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {}
        for t, r in pending:
            future = executor.submit(run_one, t, r, args.sigma_tgt)
            future_map[future] = (t, r)

        for future in concurrent.futures.as_completed(future_map):
            t, r = future_map[future]
            try:
                res = future.result()
            except Exception as e:
                res = {"ticker": t, "round": r, "error": str(e)}
            results.append(res)

            if "error" in res:
                print(f"{t:>6} r{r:>2}  ERROR: {res['error']}")
            else:
                w = "DQN" if res["dqn_cum"] > res["long_cum"] else "LONG"
                print(f"{t:>6} r{r:>2} {res['period']:>22} {res['dqn_cum']:>+9.3f} {res['long_cum']:>+9.3f} "
                      f"{res['dqn_sharpe']:>+8.3f} {res['long_sharpe']:>+8.3f} "
                      f"{res['dqn_mdd']:>+8.3f} {res['long_mdd']:>+8.3f} {w:>6}")

    elapsed = time.time() - t0
    ok = [r for r in results if "error" not in r]
    dqn_wins = sum(1 for r in ok if r["dqn_cum"] > r["long_cum"])
    print(f"\nDQN beats Long: {dqn_wins}/{len(ok)} ({dqn_wins/max(len(ok),1)*100:.0f}%) | {elapsed:.0f}s")

    # Save
    out_path = REPO / f"test_period_backtest_{tickers[0].lower()}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
