#!/usr/bin/env python3
"""Deprecated compatibility wrapper for global shared DRL feature preparation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drl.dqn.train.prepare_dqn_walkforward import (
    prepare_contract_round_features as prepare_contract_round_data,
    prepare_round_features as prepare_round,
)
from drl_shared.spec import ACTIVE_FEATURE_VERSION, RETRAIN_ROUNDS


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="All", help='Asset universe or "All"')
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--round", type=int, default=1, choices=sorted(RETRAIN_ROUNDS))
    parser.add_argument("--model-version", default=ACTIVE_FEATURE_VERSION)
    args = parser.parse_args()
    if args.ticker:
        prepare_contract_round_data(args.ticker.upper(), args.round, model_version=args.model_version)
    else:
        prepare_round(args.asset, args.round, model_version=args.model_version)
