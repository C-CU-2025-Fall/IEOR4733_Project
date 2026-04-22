#!/usr/bin/env python3
"""Compatibility wrapper for shared-round DQN data preparation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dqn.train.prepare_dqn_walkforward import prepare_round


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="All", help='Asset universe or "All"')
    parser.add_argument("--round", type=int, default=1, help="Shared DQN retrain round")
    args = parser.parse_args()
    prepare_round(args.asset, args.round)
