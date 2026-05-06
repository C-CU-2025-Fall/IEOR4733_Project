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

from baseline_run import load_contracts, compute_portfolio_returns
from vol_scaling import get_portfolio_bridge

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


def load_dqn_ensemble_returns(asset_name):
    """Load DQN ensemble returns from the corrected ensemble_table2 path."""
    path_name = ASSET_PATH_MAP[asset_name]
    npz_path = Path(f'drl/dqn/reports/ensemble_table2/{path_name}/top5_ensemble_R.npz')
    
    if not npz_path.exists():
        raise ValueError(f"DQN ensemble data not found at {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    returns = data['portfolio_returns']
    dates = data['dates']
    
    # Apply port_vol_target scaling using constant_posthoc method
    scaler = get_portfolio_bridge('constant_posthoc', PORT_VOL_TARGET)
    returns_scaled = scaler(returns)
    
    return dates, returns_scaled


def compute_long_only_returns(asset_name):
    raw_data = load_contracts(
        asset_name,
        test_start=TEST_START,
        test_end=TEST_END
    )
    
    returns = compute_portfolio_returns(raw_data, 'Long', SIGMA_TGT)
    dates = raw_data[0]['dates']
    min_len = min(len(returns), len(dates))
    returns = returns[:min_len]
    dates = dates[:min_len]
    
    # Apply same port_vol_target scaling for fair comparison
    scaler = get_portfolio_bridge('constant_posthoc', PORT_VOL_TARGET)
    returns_scaled = scaler(returns)
    
    return dates, returns_scaled


def compute_cumulative_returns(returns):
    return np.cumsum(returns)


def create_figure():
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE, dpi=DPI)
    fig.suptitle(
        'Figure 1: Cumulative Trade Returns — DQN Top-5 Ensemble vs Long Only (2011–2019)',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    all_dqn_returns = {}
    all_long_returns = {}
    
    for idx, asset in enumerate(ASSETS):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        print(f"Processing {asset}...")
        
        dqn_dates, dqn_returns = load_dqn_ensemble_returns(asset)
        dqn_dt = pd.to_datetime(dqn_dates)
        dqn_cum = compute_cumulative_returns(dqn_returns)
        
        long_dates, long_returns = compute_long_only_returns(asset)
        long_dt = pd.to_datetime(long_dates)
        long_cum = compute_cumulative_returns(long_returns)
        
        all_dqn_returns[asset] = (dqn_dt, dqn_returns)
        all_long_returns[asset] = (long_dt, long_returns)
        
        ax.plot(dqn_dt, dqn_cum, color=DQN_COLOR, linewidth=1.5, label='DQN Top-5 Ensemble')
        ax.plot(long_dt, long_cum, color=LONG_COLOR, linewidth=1.5, linestyle='--', label='Long Only')
        
        boundary_date = pd.Timestamp(R1_R2_BOUNDARY)
        ax.axvline(x=boundary_date, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        
        ax.set_title(asset, fontsize=11, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Cumulative Return (σ = 0.97)', fontsize=10)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=9)
        ax.tick_params(axis='x', rotation=45)
    
    ax_all = axes[1, 1]
    print("Processing All Portfolio...")
    
    all_dqn_aligned = []
    all_long_aligned = []
    
    for asset in ASSETS:
        dqn_dt, dqn_ret = all_dqn_returns[asset]
        long_dt, long_ret = all_long_returns[asset]
        
        dqn_series = pd.Series(dqn_ret, index=dqn_dt)
        long_series = pd.Series(long_ret, index=long_dt)
        
        all_dqn_aligned.append(dqn_series)
        all_long_aligned.append(long_series)
    
    dqn_df = pd.concat(all_dqn_aligned, axis=1, sort=True)
    long_df = pd.concat(all_long_aligned, axis=1, sort=True)
    
    dqn_avg = dqn_df.mean(axis=1)
    long_avg = long_df.mean(axis=1)
    
    dqn_all_cum = compute_cumulative_returns(dqn_avg.values)
    long_all_cum = compute_cumulative_returns(long_avg.values)

    ax_all.plot(dqn_avg.index, dqn_all_cum, color=DQN_COLOR, linewidth=1.5, label='DQN Top-5 Ensemble')
    ax_all.plot(long_avg.index, long_all_cum, color=LONG_COLOR, linewidth=1.5, linestyle='--', label='Long Only')
    
    boundary_date = pd.Timestamp(R1_R2_BOUNDARY)
    ax_all.axvline(x=boundary_date, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    
    ax_all.set_title('All Portfolio', fontsize=11, fontweight='bold')
    ax_all.set_xlabel('Date', fontsize=10)
    ax_all.set_ylabel('Cumulative Return (σ = 0.97)', fontsize=10)
    ax_all.legend(loc='upper left', fontsize=9)
    ax_all.grid(True, alpha=0.3)
    ax_all.tick_params(axis='both', labelsize=9)
    ax_all.tick_params(axis='x', rotation=45)
    
    axes[1, 2].axis('off')
    
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
