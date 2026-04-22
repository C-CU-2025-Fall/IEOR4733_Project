#!/usr/bin/env python3
"""Shared-model DQN walk-forward backtest."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline_run import compute_contract_returns_from_positions, load_contracts
from config import PAPER_TABLE3
from dqn.model import SharedDQNAgent
from dqn.pipeline import build_feature_matrix, compute_additive_returns, compute_ewma_sigma, get_feature_window
from dqn.spec import SHARED_ROUNDS, SIGMA_TGT, WARMUP, round_model_path


def load_round_agent(round_num: int, asset_name: str, checkpoint: str | None = None) -> SharedDQNAgent:
    ckpt_path = Path(checkpoint) if checkpoint else round_model_path(round_num, asset_name)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing shared DQN checkpoint: {ckpt_path}. "
            "Backtest no longer falls back to Long; provide a shared round checkpoint."
        )
    agent = SharedDQNAgent()
    agent.load(ckpt_path)
    agent.q_net.eval()
    return agent


def infer_contract_positions(agent: SharedDQNAgent, rd: dict) -> np.ndarray:
    prices = rd["prices"]
    returns = compute_additive_returns(prices)
    sigma = compute_ewma_sigma(returns)
    features = build_feature_matrix(prices, returns, sigma)
    positions = np.zeros(len(prices), dtype=float)
    for idx in range(WARMUP, len(prices)):
        state = get_feature_window(features, idx)
        positions[idx] = float(agent.predict_action_id(state) - 1)
    return positions


def compute_round_returns(round_num: int, asset_name: str, checkpoint: str | None = None) -> tuple[np.ndarray, list[str]]:
    round_info = SHARED_ROUNDS[round_num]
    raw = load_contracts(
        asset_name,
        test_start=round_info["test_start"],
        test_end=round_info["test_end"],
    )
    if not raw:
        raise ValueError(f"No contracts loaded for asset universe {asset_name}")

    agent = load_round_agent(round_num, asset_name, checkpoint=checkpoint)
    series = []
    tickers_used = []
    for rd in raw:
        positions = infer_contract_positions(agent, rd)
        contract_returns = compute_contract_returns_from_positions(rd, positions, sigma_tgt=SIGMA_TGT)
        start, t1, dates = rd["start"], rd["t1"], rd["dates"]
        slc = contract_returns[start:t1 + 1]
        series.append(pd.Series(slc[:len(dates)], index=dates[:len(slc)]))
        tickers_used.append(rd["tk"])

    df_all = pd.DataFrame(series).T
    port = df_all.mean(axis=1).to_numpy(dtype=float)
    return port, tickers_used


def compute_trade_metrics(port: np.ndarray, n_contracts: int) -> dict[str, float]:
    pos = port[port > 0]
    neg = port[port < 0]
    er = port.mean() * 252
    std = port.std(ddof=0) * np.sqrt(252)
    dd = neg.std(ddof=0) * np.sqrt(252) if len(neg) else 0.0
    sharpe = er / std if std > 0 else 0.0
    sortino = er / dd if dd > 0 else 0.0
    pct_ve = len(pos) / len(port) if len(port) else 0.0
    ave_pl = (pos.mean() / abs(neg.mean())) if len(pos) and len(neg) and abs(neg.mean()) > 0 else 0.0
    values = {
        "E(R)": er,
        "std(R)": std,
        "DD": dd,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "% +ve": pct_ve,
        "Ave P/L": ave_pl,
    }
    return {k: round(v, 3) for k, v in values.items()}


def backtest_round(round_num: int, asset_name: str = "Forex", checkpoint: str | None = None) -> dict[str, float]:
    round_info = SHARED_ROUNDS[round_num]
    print(f"\n{'=' * 70}")
    print(f"Shared DQN Walk-Forward Backtest — {asset_name} — r{round_num}")
    print(f"Train: {round_info['train_start']} ~ {round_info['train_end']}")
    print(f"Test : {round_info['test_start']} ~ {round_info['test_end']}")
    print(f"{'=' * 70}")

    port, tickers_used = compute_round_returns(round_num, asset_name, checkpoint=checkpoint)
    metrics = compute_trade_metrics(port, n_contracts=len(tickers_used))
    paper = PAPER_TABLE3[asset_name]["Long"]

    print(f"Contracts used: {len(tickers_used)}")
    for metric in ["E(R)", "std(R)", "DD", "Sharpe", "Sortino", "% +ve", "Ave P/L"]:
        ours = metrics[metric]
        target = paper[metric]
        err = abs((ours - target) / abs(target)) * 100 if target != 0 else 0.0
        print(f"  {metric:8s}: {ours:+.3f} vs {target:+.3f} err={err:.1f}%")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, choices=sorted(SHARED_ROUNDS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--asset", default="Forex")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    if args.all:
        for round_num in sorted(SHARED_ROUNDS):
            backtest_round(round_num, args.asset, checkpoint=args.checkpoint)
    elif args.round:
        backtest_round(args.round, args.asset, checkpoint=args.checkpoint)
    else:
        parser.print_help()
