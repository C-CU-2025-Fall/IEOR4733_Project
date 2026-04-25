#!/usr/bin/env python3
"""Batch prepare features for all 50 contracts with burn-in."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drl_shared.prepare_features import prepare_round_features
from drl_shared.spec import universe_tickers

if __name__ == "__main__":
    # Prepare features for all asset classes
    asset_classes = ["Forex", "Fixed Income", "Equity Index", "Commodity"]
    
    for asset in asset_classes:
        tickers = universe_tickers(asset)
        print(f"\n{'='*70}")
        print(f"Preparing features for {asset} ({len(tickers)} contracts)")
        print(f"{'='*70}")
        
        ok, fail = prepare_round_features(
            asset_name=asset,
            round_num=1,
            excluded=set(),  # No exclusions for batch
        )
        print(f"{asset}: {ok} ok, {fail} failed")
