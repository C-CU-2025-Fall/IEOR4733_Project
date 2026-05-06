#!/usr/bin/env python3
"""
Generate Paper Figure 1 — Cumulative Trade Returns (5 panels)
DQN Top-5 Q-ensemble vs Long Only (2011-2019)

Usage:
    python3 drl/dqn/figures/paper_figure1_cumulative_returns.py

Output:
    drl/dqn/figures/paper_figure1_cumulative_returns.png
    drl/dqn/figures/paper_figure1_cumulative_returns.pdf
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from baseline_run import load_contracts, compute_portfolio_return_series
from drl.dqn.figures.figure_data import load_scaled_ensemble_series, scale_return_series
from drl_shared.spec import current_source_policy

ASSETS = ['Commodity', 'Equity Index', 'Fixed Income', 'Forex']
ASSET_PATH_MAP = {
    'Commodity': 'Commodity',
    'Equity Index': 'Equity_Index',
    'Fixed Income': 'Fixed_Income',
    'Forex': 'Forex'
}
SIGMA_TGT = 0.058
PORT_VOL_TARGET = 0.97  # Corrected target volatility matching paper Table 2
TEST_START = '2011-01-01'
TEST_END = '2019-12-31'
R1_R2_BOUNDARY = '2016-01-01'

DQN_COLOR = '#1f77b4'
LONG_COLOR = '#ff7f0e'
DPI = 300
FIGSIZE = (16, 10)
SOURCE_POLICY = current_source_policy()


def load_dqn_ensemble_returns(asset_name):
    """Load DQN ensemble returns from the corrected ensemble_table2 path."""
    path_name = ASSET_PATH_MAP[asset_name]
    npz_path = Path(f'drl/dqn/reports/ensemble_table2/{path_name}/top5_ensemble_R.npz')
    series = load_scaled_ensemble_series(npz_path, PORT_VOL_TARGET)
    return series.index, series.values


def compute_long_only_returns(asset_name):
    raw_data = load_contracts(
        asset_name,
        test_start=TEST_START,
        test_end=TEST_END,
        excluded_contracts=SOURCE_POLICY['excluded_contracts'],
        source_overrides=SOURCE_POLICY['source_overrides'],
    )
    
    series = compute_portfolio_return_series(raw_data, 'Long', SIGMA_TGT)
    series = scale_return_series(series, PORT_VOL_TARGET)
    return series.index, series.values


def compute_cumulative_returns(returns):
    return np.cumsum(returns)


def create_figure():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=DPI)
    fig.suptitle(
        'Figure 1: Cumulative Trade Returns — DQN Top-5 Ensemble vs Long Only (2011–2019)',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    for idx, asset in enumerate(ASSETS):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        print(f"Processing {asset}...")
        
        dqn_dates, dqn_returns = load_dqn_ensemble_returns(asset)
        dqn_dt = pd.to_datetime(dqn_dates)
        dqn_cum = compute_cumulative_returns(dqn_returns)
        
        long_dates, long_returns = compute_long_only_returns(asset)
        long_dt = pd.to_datetime(long_dates)
        long_cum = compute_cumulative_returns(long_returns)
        
        ax.plot(dqn_dt, dqn_cum, color=DQN_COLOR, linewidth=1.5, label='DQN Top-5 Ensemble')
        ax.plot(long_dt, long_cum, color=LONG_COLOR, linewidth=1.5, linestyle='--', label='Long Only')
        
        boundary_date = pd.Timestamp(R1_R2_BOUNDARY)
        ax.axvline(x=boundary_date, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        
        ax.set_title(asset, fontsize=11, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Cumulative Return ($\\sigma$ = 0.97)', fontsize=10)
        ax.legend(loc='upper left', fontsize=9)
        ax.tick_params(axis='both', labelsize=9)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig


def main():
    output_path = Path('drl/dqn/figures/paper_figure1_cumulative_returns.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Generating Figure 1: Cumulative Trade Returns")
    print("=" * 50)
    print(f"Using port_vol_target = {PORT_VOL_TARGET}")
    print(f"Loading from: drl/dqn/reports/ensemble_table2/")
    
    fig = create_figure()
    
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"\nFigure saved to: {output_path}")
    
    pdf_path = output_path.with_suffix('.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"PDF saved to: {pdf_path}")
    
    plt.close(fig)
    print("\nDone!")


if __name__ == '__main__':
    main()
