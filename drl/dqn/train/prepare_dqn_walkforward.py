#!/usr/bin/env python3
"""Deprecated compatibility CLI: delegate to global shared DRL feature preparation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drl_shared.prepare_features import prepare_contract_round_features, prepare_round_features
from drl_shared.spec import RETRAIN_ROUNDS


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="Forex", help='Asset universe or "All"')
    parser.add_argument("--ticker", default=None, help="Optional single ticker override")
    parser.add_argument("--round", type=int, choices=sorted(RETRAIN_ROUNDS), default=None)
    parser.add_argument("--all-rounds", action="store_true")
    args = parser.parse_args()

    rounds = sorted(RETRAIN_ROUNDS) if args.all_rounds or args.round is None else [args.round]
    for round_num in rounds:
        if args.ticker:
            prepare_contract_round_features(args.ticker.upper(), round_num)
        else:
            prepare_round_features(args.asset, round_num)
