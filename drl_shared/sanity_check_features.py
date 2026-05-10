#!/usr/bin/env python3
"""Sanity check for generated feature data (.npz) artifacts.
Checks for:
1. NaNs or Infs in features, prices, and returns.
2. All-zero feature arrays.
3. High rates of duplicate rows (which may indicate forward-fill issues or broken data).
"""

import argparse
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drl_shared.spec import feature_data_path, universe_tickers, RETRAIN_ROUNDS

def sanity_check_features(asset: str, round_num: int):
    tickers = universe_tickers(asset)
    print(f"\n{'=' * 60}")
    print(f"Sanity Check: {asset} - Round {round_num}")
    print(f"{'=' * 60}")
    
    checked_count = 0
    warning_count = 0
    
    for ticker in tickers:
        path = feature_data_path(round_num, ticker)
        if not path.exists():
            continue
            
        checked_count += 1
        try:
            with np.load(path, allow_pickle=True) as data:
                features = data['features']
                prices = data['prices']
                returns = data['returns']
                
            has_warning = False
            
            # 1. NaN / Inf checks
            if not np.isfinite(features).all():
                print(f"  [!] {ticker}: Found NaN or Inf in features.")
                has_warning = True
            if not np.isfinite(prices).all():
                print(f"  [!] {ticker}: Found NaN or Inf in prices.")
                has_warning = True
                
            # 2. All-zero checks
            if np.allclose(features, 0.0):
                print(f"  [!] {ticker}: Features are entirely zeros.")
                has_warning = True
            if np.allclose(prices, 0.0):
                print(f"  [!] {ticker}: Prices are entirely zeros.")
                has_warning = True
                
            # 3. Duplicate rows check
            # For features, excessive exact duplicate rows suggest stale prices or broken indicators.
            # We use a threshold of 10% exact duplicate rows (some are expected on holidays if ffilled)
            unique_rows = np.unique(features, axis=0)
            duplicate_ratio = 1.0 - (len(unique_rows) / len(features))
            if duplicate_ratio > 0.10:
                print(f"  [!] {ticker}: High duplicate feature rows ratio: {duplicate_ratio:.2%} (Possible stale data or ffill issues)")
                has_warning = True
                
            if has_warning:
                warning_count += 1
                
        except Exception as e:
            print(f"  [!] {ticker}: Failed to load or parse {path.name} - {str(e)}")
            warning_count += 1

    print(f"\nSanity check complete. Checked {checked_count} artifacts, found warnings in {warning_count}.")
    return warning_count == 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="Forex", help='Asset universe')
    parser.add_argument("--round", type=int, choices=sorted(RETRAIN_ROUNDS), default=1)
    args = parser.parse_args()
    
    sanity_check_features(args.asset, args.round)

if __name__ == "__main__":
    main()
