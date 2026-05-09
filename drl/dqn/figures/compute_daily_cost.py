#!/usr/bin/env python3
"""
Compute real daily cost per contract for all BP levels.

Formula: daily_cost = sum(|position_t - position_{t-1}| * bp * price_t) / n_contracts

Outputs:
- exhibit5_daily_cost_{bp}.csv for each BP level
- exhibit5_daily_cost_all.csv (summary)
- Updates exhibit5_bp{XX}.csv with daily_cost column
"""
import csv
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from baseline_run import load_contracts

DATA_DIR = REPO_ROOT / "drl/dqn/figures/data"
REPORTS_DIR = REPO_ROOT / "drl/dqn/reports/ensemble_table2_bp"

BP_LEVELS = [1, 10, 20, 30, 45]
ASSETS = {
    "Commodity": "Commodity",
    "Equity_Index": "Equity Index",
    "Fixed_Income": "Fixed Income",
    "Forex": "Forex"
}
TEST_START = "2011-01-01"
TEST_END = "2019-12-31"


def load_positions(asset_slug, bp):
    """Load positions.csv for a given asset and BP level."""
    positions_file = REPORTS_DIR / asset_slug / f"bp{bp}" / "positions.csv"
    if not positions_file.exists():
        print(f"Warning: {positions_file} not found")
        return None
    
    df = pd.read_csv(positions_file)
    return df


def compute_daily_cost_for_asset(asset_slug, asset_name, bp):
    """
    Compute average daily cost per contract for a given asset and BP level.
    
    Formula: daily_cost = sum(|position_t - position_{t-1}| * bp * price_t) / n_contracts
    """
    print(f"  Computing daily cost for {asset_name} (BP={bp})...")
    
    # Load positions
    positions_df = load_positions(asset_slug, bp)
    if positions_df is None:
        return None
    
    # Load price data
    contracts_data = load_contracts(asset_name, test_start=TEST_START, test_end=TEST_END)
    if not contracts_data:
        print(f"  Warning: No contract data for {asset_name}")
        return None
    
    n_contracts = len(contracts_data)
    
    # Create a dictionary mapping (contract, date) to price
    price_lookup = {}
    for contract_data in contracts_data:
        ticker = contract_data['tk']
        dates = contract_data['dates']
        prices = contract_data['prices']
        for date, price in zip(dates, prices):
            date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
            price_lookup[(ticker, date_str)] = price
    
    # Group positions by contract and compute turnover
    total_daily_costs = []
    
    for contract in positions_df['contract'].unique():
        contract_positions = positions_df[positions_df['contract'] == contract].copy()
        contract_positions = contract_positions.sort_values('date')
        
        # Get positions and dates
        dates = contract_positions['date'].values
        positions = contract_positions['position'].values
        
        # Compute turnover (absolute change in position)
        if len(positions) > 1:
            turnover = np.abs(positions[1:] - positions[:-1])
            turnover_dates = dates[1:]
            
            # Compute daily cost for each day with turnover
            for i, (date, turn) in enumerate(zip(turnover_dates, turnover)):
                if turn > 0:
                    # Get price for this contract and date
                    price = price_lookup.get((contract, date))
                    if price is not None and not np.isnan(price):
                        # Cost = turnover * bp * price / 10000 (bp is in basis points)
                        daily_cost = turn * bp * price / 10000.0
                        total_daily_costs.append(daily_cost)
    
    # Average daily cost per contract
    if total_daily_costs:
        avg_daily_cost = np.mean(total_daily_costs)
    else:
        avg_daily_cost = 0.0
    
    print(f"    Average daily cost per contract: {avg_daily_cost:.6f}")
    print(f"    Total trading days with turnover: {len(total_daily_costs)}")
    
    return {
        'Asset': asset_name,
        'BP': bp,
        'avg_daily_cost': avg_daily_cost,
        'n_contracts': n_contracts,
        'n_trading_days': len(total_daily_costs)
    }


def compute_all_daily_costs():
    """Compute daily costs for all assets and BP levels."""
    results_by_bp = {}
    all_results = []
    
    for bp in BP_LEVELS:
        print(f"\nProcessing BP level: {bp}")
        bp_results = []
        
        for asset_slug, asset_name in ASSETS.items():
            result = compute_daily_cost_for_asset(asset_slug, asset_name, bp)
            if result:
                bp_results.append(result)
                all_results.append(result)
        
        # Compute "All" category as average across assets
        if bp_results:
            all_avg_cost = np.mean([r['avg_daily_cost'] for r in bp_results])
            all_result = {
                'Asset': 'All',
                'BP': bp,
                'avg_daily_cost': all_avg_cost,
                'n_contracts': sum(r['n_contracts'] for r in bp_results),
                'n_trading_days': sum(r['n_trading_days'] for r in bp_results)
            }
            bp_results.append(all_result)
            all_results.append(all_result)
        
        results_by_bp[bp] = bp_results
    
    return results_by_bp, all_results


def save_daily_cost_csv(results_by_bp):
    """Save daily cost data to CSV files for each BP level."""
    for bp, results in results_by_bp.items():
        output_file = DATA_DIR / f"exhibit5_daily_cost_bp{bp}.csv"
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Asset', 'BP', 'avg_daily_cost', 'n_contracts', 'n_trading_days'])
            for r in results:
                writer.writerow([
                    r['Asset'],
                    r['BP'],
                    f"{r['avg_daily_cost']:.6f}",
                    r['n_contracts'],
                    r['n_trading_days']
                ])
        print(f"Saved: {output_file}")


def save_daily_cost_all_csv(all_results):
    """Save summary CSV with all BP levels."""
    output_file = DATA_DIR / "exhibit5_daily_cost_all.csv"
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Asset', 'BP', 'avg_daily_cost', 'n_contracts', 'n_trading_days'])
        for r in all_results:
            writer.writerow([
                r['Asset'],
                r['BP'],
                f"{r['avg_daily_cost']:.6f}",
                r['n_contracts'],
                r['n_trading_days']
            ])
    print(f"Saved: {output_file}")


def update_exhibit5_csvs(results_by_bp):
    """Update existing exhibit5_bp{XX}.csv files with daily_cost column."""
    for bp, results in results_by_bp.items():
        input_file = DATA_DIR / f"exhibit5_bp{bp}.csv"
        if not input_file.exists():
            print(f"Warning: {input_file} not found, skipping")
            continue
        
        # Read existing data
        with open(input_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames
        
        # Add daily_cost column if not present
        if 'daily_cost' not in fieldnames:
            fieldnames = fieldnames + ['daily_cost']
        
        # Create lookup for daily costs
        cost_lookup = {r['Asset']: r['avg_daily_cost'] for r in results}
        
        # Update rows with daily_cost
        for row in rows:
            asset = row['Asset']
            row['daily_cost'] = f"{cost_lookup.get(asset, 0):.6f}"
        
        # Write updated data
        with open(input_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"Updated: {input_file}")


def main():
    print("=" * 60)
    print("Computing Real Daily Cost Per Contract")
    print("=" * 60)
    
    # Ensure output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Compute daily costs
    results_by_bp, all_results = compute_all_daily_costs()
    
    # Save results
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)
    
    save_daily_cost_csv(results_by_bp)
    save_daily_cost_all_csv(all_results)
    update_exhibit5_csvs(results_by_bp)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
