#!/usr/bin/env python3
"""Legacy alias for the shared-model DQN trainer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dqn.spec import SHARED_ROUNDS
from dqn.train.train_dqn_walkforward import train_round_shared


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=1, choices=sorted(SHARED_ROUNDS))
    parser.add_argument("--asset", default="All", help='Asset universe or "All"')
    parser.add_argument("--episodes", type=int, default=200)
    args = parser.parse_args()

    print("Shared-model DQN trainer (paper-faithful infrastructure alias)")
    train_round_shared(args.asset, args.round, args.episodes)
