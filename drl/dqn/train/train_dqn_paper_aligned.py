#!/usr/bin/env python3
"""Legacy alias for the single-contract DQN trainer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drl.dqn.spec import RETRAIN_ROUNDS
from drl.dqn.train.train_dqn_walkforward import train_contract_round


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--round", type=int, default=1, choices=sorted(RETRAIN_ROUNDS))
    parser.add_argument("--episodes", type=int, default=200)
    args = parser.parse_args()

    print("Single-contract DQN trainer (paper-faithful infrastructure alias)")
    train_contract_round(args.ticker, args.round, args.episodes)
