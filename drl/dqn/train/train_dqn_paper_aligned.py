#!/usr/bin/env python3
"""Deprecated alias for the single-contract DQN trainer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drl.dqn.spec import ACTIVE_MODEL_VERSION, RETRAIN_ROUNDS
from drl.dqn.train.train_dqn_walkforward import train_contract_round


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--round", type=int, default=1, choices=sorted(RETRAIN_ROUNDS))
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--model-version", default=ACTIVE_MODEL_VERSION)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    print("Deprecated alias: use drl/dqn/train/train_dqn_walkforward.py directly.")
    train_contract_round(
        args.ticker,
        args.round,
        args.episodes,
        model_version=args.model_version,
        device=args.device,
        seed=args.seed,
    )
