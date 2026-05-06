#!/usr/bin/env python3
"""
Generate Supplementary Figures from Ensemble Table 2 R data (port_vol_target=0.97)

Usage:
    python3 drl/dqn/figures/supplementary_figures.py

Output:
    drl/dqn/figures/supp_rolling_sharpe.png (300 DPI)
    drl/dqn/figures/supp_drawdown.png (300 DPI)
    drl/dqn/figures/supp_monthly_heatmap.png (300 DPI)
    drl/dqn/figures/supp_yearly_bars.png (300 DPI)
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
PORT_VOL_TARGET = 0.97
TEST_START = '2011-01-01'
TEST_END = '2019-12-31'
R1_R2_BOUNDARY = '2016-01-01'

DQN_COLOR = '#1f77b4'
LONG_COLOR = '#ff7f0e'
DPI = 300


def load_dqn_ensemble_returns(asset_name):
    path_name = ASSET_PATH_MAP[asset_name]
    npz_path = Path(f'drl/dqn/reports/ensemble_table2/{path_name}/top5_ensemble_R.npz')
    
    if not npz_path.exists():
        raise ValueError(f"DQN ensemble data not found at {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    returns = data['portfolio_returns']
    dates = data['dates']
    
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
    
    scaler = get_portfolio_bridge('constant_posthoc', PORT_VOL_TARGET)
    returns_scaled = scaler(returns)
    
    return dates, returns_scaled


def compute_rolling_sharpe(returns, window=252):
    """
    rolling_sharpe(t) = mean(R[t-251:t+1]) / std(R[t-251:t+1]) * sqrt(252)
    """
    returns_series = pd.Series(returns)
    rolling_mean = returns_series.rolling(window=window, min_periods=window).mean()
    rolling_std = returns_series.rolling(window=window, min_periods=window).std()
    rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)
    return rolling_sharpe.values


def compute_drawdown(returns):
    """
    cum = np.cumsum(R)
    running_max = np.maximum.accumulate(cum)
    drawdown = (cum - running_max) / (running_max + 1e-10) * 100
    """
    cum = np.cumsum(returns)
    running_max = np.maximum.accumulate(cum)
    drawdown = (cum - running_max) / (running_max + 1e-10) * 100
    return drawdown


def create_figure_rolling_sharpe():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=DPI)
    fig.suptitle(
        'Supplementary Figure A: Rolling 252-Day Sharpe Ratio (DQN vs Long Only)',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    for idx, asset in enumerate(ASSETS):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        print(f"  Processing rolling Sharpe for {asset}...")
        
        dqn_dates, dqn_returns = load_dqn_ensemble_returns(asset)
        dqn_dt = pd.to_datetime(dqn_dates)
        dqn_sharpe = compute_rolling_sharpe(dqn_returns)
        
        long_dates, long_returns = compute_long_only_returns(asset)
        long_dt = pd.to_datetime(long_dates)
        long_sharpe = compute_rolling_sharpe(long_returns)
        
        ax.plot(dqn_dt, dqn_sharpe, color=DQN_COLOR, linewidth=1.5, label='DQN Top-5 Ensemble')
        ax.plot(long_dt, long_sharpe, color=LONG_COLOR, linewidth=1.5, linestyle='--', label='Long Only')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        boundary_date = pd.Timestamp(R1_R2_BOUNDARY)
        ax.axvline(x=boundary_date, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        
        ax.set_title(asset, fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Rolling 252-Day Sharpe', fontsize=10)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=9)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def create_figure_drawdown():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=DPI)
    fig.suptitle(
        'Supplementary Figure B: Drawdown Curves (DQN vs Long Only)',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    for idx, asset in enumerate(ASSETS):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        print(f"  Processing drawdown for {asset}...")
        
        dqn_dates, dqn_returns = load_dqn_ensemble_returns(asset)
        dqn_dt = pd.to_datetime(dqn_dates)
        dqn_drawdown = compute_drawdown(dqn_returns)
        
        long_dates, long_returns = compute_long_only_returns(asset)
        long_dt = pd.to_datetime(long_dates)
        long_drawdown = compute_drawdown(long_returns)
        
        ax.fill_between(dqn_dt, dqn_drawdown, 0, color=DQN_COLOR, alpha=0.4, label='DQN Top-5 Ensemble')
        ax.plot(dqn_dt, dqn_drawdown, color=DQN_COLOR, linewidth=1.0)
        ax.fill_between(long_dt, long_drawdown, 0, color=LONG_COLOR, alpha=0.4, label='Long Only')
        ax.plot(long_dt, long_drawdown, color=LONG_COLOR, linewidth=1.0, linestyle='--')
        
        boundary_date = pd.Timestamp(R1_R2_BOUNDARY)
        ax.axvline(x=boundary_date, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        
        ax.set_title(asset, fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Drawdown (%)', fontsize=10)
        ax.legend(loc='lower left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=9)
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(bottom=min(dqn_drawdown.min(), long_drawdown.min()) * 1.1, top=5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def create_figure_monthly_heatmap():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=DPI)
    fig.suptitle(
        'Supplementary Figure C: Monthly Returns Heatmap',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    for idx, asset in enumerate(ASSETS):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        print(f"  Processing monthly heatmap for {asset}...")
        
        dqn_dates, dqn_returns = load_dqn_ensemble_returns(asset)
        dqn_series = pd.Series(dqn_returns, index=pd.to_datetime(dqn_dates))
        
        monthly_returns = dqn_series.resample('ME').sum()
        
        monthly_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'returns': monthly_returns.values
        })
        pivot = monthly_df.pivot(index='month', columns='year', values='returns')
        pivot = pivot.reindex(range(1, 13))
        
        vmax = max(abs(pivot.max().max()), abs(pivot.min().min()))
        vmin = -vmax
        
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', 
                       vmin=vmin, vmax=vmax, interpolation='nearest')
        
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=9)
        ax.set_yticks(range(12))
        ax.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=9)
        
        for i in range(12):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if not pd.isna(val):
                    text_color = 'white' if abs(val) > vmax * 0.5 else 'black'
                    ax.text(j, i, f'{val*100:.1f}', ha='center', va='center', 
                           fontsize=7, color=text_color)
        
        ax.set_title(f'{asset}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('Month', fontsize=10)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Monthly Return', fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def create_figure_yearly_bars():
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), dpi=DPI)
    fig.suptitle(
        'Supplementary Figure D: Year-by-Year Annual Sharpe Ratio (DQN vs Long Only)',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    for idx, asset in enumerate(ASSETS):
        ax = axes[idx]
        
        print(f"  Processing yearly bars for {asset}...")
        
        dqn_dates, dqn_returns = load_dqn_ensemble_returns(asset)
        dqn_series = pd.Series(dqn_returns, index=pd.to_datetime(dqn_dates))
        
        long_dates, long_returns = compute_long_only_returns(asset)
        long_series = pd.Series(long_returns, index=pd.to_datetime(long_dates))
        
        def annual_sharpe(x):
            if len(x) == 0 or x.std() == 0:
                return 0
            return x.mean() / x.std() * np.sqrt(252)
        
        dqn_annual = dqn_series.resample('YE').apply(annual_sharpe)
        long_annual = long_series.resample('YE').apply(annual_sharpe)
        
        years = dqn_annual.index.year
        
        x = np.arange(len(years))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, dqn_annual.values, width, 
                      label='DQN Top-5 Ensemble', color=DQN_COLOR, alpha=0.8)
        bars2 = ax.bar(x + width/2, long_annual.values, width,
                      label='Long Only', color=LONG_COLOR, alpha=0.8)
        
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        ax.set_title(f'{asset}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('Annual Sharpe Ratio', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(years, fontsize=9)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', labelsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def main():
    output_dir = Path('drl/dqn/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Generating Supplementary Figures from Ensemble Table 2")
    print("=" * 60)
    print(f"Using port_vol_target = {PORT_VOL_TARGET}")
    print(f"Output directory: {output_dir}")
    print()
    
    print("Generating Figure A: Rolling Sharpe Ratio...")
    fig_a = create_figure_rolling_sharpe()
    fig_a_path = output_dir / 'supp_rolling_sharpe.png'
    fig_a.savefig(fig_a_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig_a)
    print(f"  Saved: {fig_a_path}")
    print()
    
    print("Generating Figure B: Drawdown Curves...")
    fig_b = create_figure_drawdown()
    fig_b_path = output_dir / 'supp_drawdown.png'
    fig_b.savefig(fig_b_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig_b)
    print(f"  Saved: {fig_b_path}")
    print()
    
    print("Generating Figure C: Monthly Returns Heatmap...")
    fig_c = create_figure_monthly_heatmap()
    fig_c_path = output_dir / 'supp_monthly_heatmap.png'
    fig_c.savefig(fig_c_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig_c)
    print(f"  Saved: {fig_c_path}")
    print()
    
    print("Generating Figure D: Year-by-Year Performance...")
    fig_d = create_figure_yearly_bars()
    fig_d_path = output_dir / 'supp_yearly_bars.png'
    fig_d.savefig(fig_d_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig_d)
    print(f"  Saved: {fig_d_path}")
    print()
    
    print("=" * 60)
    print("All supplementary figures generated successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
