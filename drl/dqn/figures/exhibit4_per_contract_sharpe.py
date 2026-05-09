#!/usr/bin/env python3
"""
Exhibit 4: Per-Contract Analysis — Sharpe and Trade Return/Turnover by BP Level

Generates a two-panel figure:
- Panel A (top): Per-contract annualized Sharpe boxplot by asset and BP
- Panel B (bottom): Trade Return/Turnover = sum(returns) / sum(|position_t - position_{t-1}|)

Data source: ensemble_table2_bp positions.csv files (already aggregated across top-5 seeds)
BP levels: 1, 10, 20, 30, 45 (basis points)

Usage:
    python3 drl/dqn/figures/exhibit4_per_contract_sharpe.py

Output:
    drl/dqn/figures/exhibit4_per_contract_sharpe.png (300 DPI)
    drl/dqn/figures/exhibit4_per_contract_sharpe.pdf
"""
import sys
sys.path.insert(0, '.')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuration
ASSETS = ['Commodity', 'Equity Index', 'Fixed Income', 'Forex']
ASSET_SLUGS = {
    'Commodity': 'Commodity',
    'Equity Index': 'Equity_Index',
    'Fixed Income': 'Fixed_Income',
    'Forex': 'Forex',
}
BP_LEVELS = [1, 10, 20, 30, 45]  # Basis points
BP_COLORS = {
    1: '#1f77b4',    # Blue
    10: '#ff7f0e',   # Orange
    20: '#2ca02c',   # Green
    30: '#d62728',   # Red
    45: '#9467bd',   # Purple
}

# Paths
REPO_ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = REPO_ROOT / 'drl' / 'dqn' / 'reports' / 'ensemble_table2_bp'
OUTPUT_PATH = REPO_ROOT / 'drl' / 'dqn' / 'figures' / 'exhibit4_per_contract_sharpe.png'
DPI = 300


def compute_annualized_sharpe(returns: np.ndarray) -> float:
    """Compute annualized Sharpe ratio from daily returns."""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    mean_daily = np.mean(returns)
    std_daily = np.std(returns)
    if std_daily == 0:
        return 0.0
    # Annualized Sharpe = mean_daily * 252 / (std_daily * sqrt(252))
    return mean_daily * 252 / (std_daily * np.sqrt(252))


def compute_trade_return_per_turnover(positions: np.ndarray, returns: np.ndarray) -> float:
    """
    Compute Trade Return / Turnover ratio.
    
    Formula: sum(returns) / sum(|position_t - position_{t-1}|)
    
    This measures the return generated per unit of position turnover.
    """
    if len(returns) == 0 or len(positions) < 2:
        return 0.0
    
    total_return = np.sum(returns)
    
    # Compute turnover as sum of absolute position changes
    position_changes = np.abs(np.diff(positions))
    total_turnover = np.sum(position_changes)
    
    if total_turnover == 0:
        return 0.0
    
    return total_return / total_turnover


def load_positions_csv(asset: str, bp: int) -> pd.DataFrame:
    """Load positions.csv for a given asset and BP level."""
    asset_slug = ASSET_SLUGS[asset]
    csv_path = REPORTS_DIR / asset_slug / f"bp{bp}" / "positions.csv"
    
    if not csv_path.exists():
        print(f"  Warning: File not found: {csv_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def compute_per_contract_metrics(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Compute per-contract metrics from positions DataFrame.
    
    Returns:
        {
            contract: {
                'sharpe': float,
                'trade_return_per_turnover': float,
                'total_return': float,
                'turnover': float,
            }
        }
    """
    metrics = {}
    
    for contract in df['contract'].unique():
        contract_df = df[df['contract'] == contract].sort_values('date')
        
        if len(contract_df) == 0:
            continue
        
        returns = contract_df['return'].values
        positions = contract_df['position'].values
        
        # Compute Sharpe
        sharpe = compute_annualized_sharpe(returns)
        
        # Compute Trade Return / Turnover
        trade_ret_per_turnover = compute_trade_return_per_turnover(positions, returns)
        
        # Also store total return and turnover separately
        total_return = np.sum(returns)
        turnover = np.sum(np.abs(np.diff(positions))) if len(positions) > 1 else 0.0
        
        metrics[contract] = {
            'sharpe': sharpe,
            'trade_return_per_turnover': trade_ret_per_turnover,
            'total_return': total_return,
            'turnover': turnover,
        }
    
    return metrics


def load_all_metrics() -> Dict:
    """
    Load and compute metrics for all assets and BP levels.
    
    Returns:
        {
            asset: {
                bp: {
                    'sharpes': [sharpe1, sharpe2, ...],
                    'trade_ret_per_turnover': [trpt1, trpt2, ...],
                }
            }
        }
    """
    all_metrics = {}
    
    for asset in ASSETS:
        print(f"\nProcessing {asset}...")
        all_metrics[asset] = {}
        
        for bp in BP_LEVELS:
            print(f"  Loading BP={bp}...")
            df = load_positions_csv(asset, bp)
            
            if df.empty:
                print(f"    No data for BP={bp}")
                all_metrics[asset][bp] = {
                    'sharpes': [],
                    'trade_ret_per_turnover': [],
                }
                continue
            
            # Compute per-contract metrics
            contract_metrics = compute_per_contract_metrics(df)
            
            # Extract lists
            sharpes = [m['sharpe'] for m in contract_metrics.values()]
            trpt_values = [m['trade_return_per_turnover'] for m in contract_metrics.values()]
            
            all_metrics[asset][bp] = {
                'sharpes': sharpes,
                'trade_ret_per_turnover': trpt_values,
            }
            
            print(f"    Contracts: {len(sharpes)}, "
                  f"Mean Sharpe: {np.mean(sharpes):.3f}, "
                  f"Mean TR/Turnover: {np.mean(trpt_values):.6f}")
    
    return all_metrics


def create_two_panel_figure(all_metrics: Dict):
    """Create the two-panel boxplot figure."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), dpi=DPI)
    
    fig.suptitle(
        'Exhibit 4: Per-Contract Performance Metrics by Transaction Cost Level',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    # Panel A: Sharpe ratios
    ax_sharpe = axes[0]
    ax_sharpe.set_title('Panel A: Per-Contract Annualized Sharpe Ratio', fontsize=12, fontweight='bold', pad=10)
    
    # Panel B: Trade Return / Turnover
    ax_trpt = axes[1]
    ax_trpt.set_title('Panel B: Trade Return per Unit Turnover', fontsize=12, fontweight='bold', pad=10)
    
    # Prepare data for both panels
    # For each asset, we'll have grouped boxplots by BP level
    asset_positions = []
    asset_labels = []
    current_pos = 1
    
    for asset in ASSETS:
        asset_positions.append(current_pos + len(BP_LEVELS) / 2)
        asset_labels.append(asset.replace(' ', '\n'))
        current_pos += len(BP_LEVELS) + 2  # +2 for spacing between assets
    
    # Plot Panel A: Sharpe
    for idx, asset in enumerate(ASSETS):
        base_pos = 1 + idx * (len(BP_LEVELS) + 2)
        
        for bp_idx, bp in enumerate(BP_LEVELS):
            pos = base_pos + bp_idx
            sharpes = all_metrics[asset][bp]['sharpes']
            
            if sharpes:
                bp_data = ax_sharpe.boxplot(
                    [sharpes],
                    positions=[pos],
                    widths=0.6,
                    patch_artist=True,
                    showfliers=True,
                    flierprops=dict(marker='o', markersize=3, alpha=0.5),
                    medianprops=dict(color='black', linewidth=1.5),
                    boxprops=dict(facecolor=BP_COLORS[bp], alpha=0.7),
                )
    
    # Plot Panel B: Trade Return / Turnover
    for idx, asset in enumerate(ASSETS):
        base_pos = 1 + idx * (len(BP_LEVELS) + 2)
        
        for bp_idx, bp in enumerate(BP_LEVELS):
            pos = base_pos + bp_idx
            trpt_values = all_metrics[asset][bp]['trade_ret_per_turnover']
            
            if trpt_values:
                bp_data = ax_trpt.boxplot(
                    [trpt_values],
                    positions=[pos],
                    widths=0.6,
                    patch_artist=True,
                    showfliers=True,
                    flierprops=dict(marker='o', markersize=3, alpha=0.5),
                    medianprops=dict(color='black', linewidth=1.5),
                    boxprops=dict(facecolor=BP_COLORS[bp], alpha=0.7),
                )
    
    # Styling for Panel A
    ax_sharpe.set_ylabel('Annualized Sharpe Ratio', fontsize=11)
    ax_sharpe.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_sharpe.set_xticks(asset_positions)
    ax_sharpe.set_xticklabels(asset_labels, fontsize=10)
    ax_sharpe.grid(axis='y', alpha=0.3, linestyle=':')
    ax_sharpe.set_xlim(0, current_pos - 1)
    
    # Styling for Panel B
    ax_trpt.set_ylabel('Trade Return / Turnover', fontsize=11)
    ax_trpt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_trpt.set_xticks(asset_positions)
    ax_trpt.set_xticklabels(asset_labels, fontsize=10)
    ax_trpt.grid(axis='y', alpha=0.3, linestyle=':')
    ax_trpt.set_xlim(0, current_pos - 1)
    
    # Add legend for BP levels
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, facecolor=BP_COLORS[bp], alpha=0.7, label=f'BP={bp}')
        for bp in BP_LEVELS
    ]
    ax_sharpe.legend(
        handles=legend_elements,
        loc='upper right',
        title='Transaction Cost',
        fontsize=9,
        title_fontsize=9,
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def create_alternative_two_panel(all_metrics: Dict):
    """
    Create alternative two-panel layout with side-by-side BP boxplots per asset.
    This shows the distribution of metrics across contracts for each BP level.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=DPI)
    
    fig.suptitle(
        'Exhibit 4: Per-Contract Performance Metrics by Transaction Cost Level',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    # Create subplots for each asset
    for idx, asset in enumerate(ASSETS):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Prepare data for boxplots
        sharpe_data = []
        trpt_data = []
        colors = []
        labels = []
        
        for bp in BP_LEVELS:
            sharpes = all_metrics[asset][bp]['sharpes']
            trpt_values = all_metrics[asset][bp]['trade_ret_per_turnover']
            
            if sharpes:
                sharpe_data.append(sharpes)
                trpt_data.append(trpt_values)
                colors.append(BP_COLORS[bp])
                labels.append(f'BP={bp}')
        
        if not sharpe_data:
            ax.set_title(asset, fontsize=11, fontweight='bold')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Create grouped boxplots for Sharpe ratios
        positions = np.arange(1, len(labels) + 1)
        bp_sharpe = ax.boxplot(
            sharpe_data,
            positions=positions - 0.25,
            widths=0.4,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker='o', markersize=3, alpha=0.4),
            medianprops=dict(color='black', linewidth=1.5),
        )
        
        # Color the boxes
        for patch, color in zip(bp_sharpe['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Create second y-axis for Trade Return/Turnover
        ax2 = ax.twinx()
        bp_trpt = ax2.boxplot(
            trpt_data,
            positions=positions + 0.25,
            widths=0.4,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker='s', markersize=3, alpha=0.4),
            medianprops=dict(color='darkred', linewidth=1.5),
        )
        
        # Color the boxes (lighter shade)
        for patch, color in zip(bp_trpt['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.3)
        
        # Styling
        ax.set_title(asset, fontsize=11, fontweight='bold')
        ax.set_ylabel('Annualized Sharpe', fontsize=9, color='black')
        ax2.set_ylabel('Trade Return / Turnover', fontsize=9, color='darkred')
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8, rotation=0)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.4)
        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.4)
        
        # Color the y-axis labels
        ax.tick_params(axis='y', labelcolor='black')
        ax2.tick_params(axis='y', labelcolor='darkred')
    
    # Add overall legend
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, facecolor=BP_COLORS[bp], alpha=0.7, label=f'BP={bp} (Sharpe)')
        for bp in BP_LEVELS
    ]
    legend_elements.append(plt.Rectangle((0,0), 1, 1, facecolor='gray', alpha=0.3, label='Trade Ret/Turnover'))
    
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=3,
        fontsize=9,
        title='Transaction Cost Levels',
        title_fontsize=9,
        bbox_to_anchor=(0.5, 0.02),
    )
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    return fig


def create_vertical_two_panel(all_metrics: Dict):
    """
    Create two-panel layout with one panel per metric type.
    Each panel shows all assets with grouped BP levels.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), dpi=DPI)
    
    fig.suptitle(
        'Exhibit 4: Per-Contract Performance Analysis by Transaction Cost',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    # Panel A: Sharpe Ratios
    ax1 = axes[0]
    ax1.set_title('Panel A: Distribution of Per-Contract Annualized Sharpe Ratios', 
                  fontsize=12, fontweight='bold', pad=10)
    
    # Panel B: Trade Return / Turnover
    ax2 = axes[1]
    ax2.set_title('Panel B: Distribution of Trade Return per Turnover Ratios', 
                  fontsize=12, fontweight='bold', pad=10)
    
    # For each asset, create grouped boxplots
    spacing = 1.5
    group_width = len(BP_LEVELS) * 0.8
    
    for idx, asset in enumerate(ASSETS):
        base_pos = idx * (group_width + spacing) + 1
        positions = [base_pos + i * 0.8 for i in range(len(BP_LEVELS))]
        
        # Get data for each BP level
        sharpe_data = []
        trpt_data = []
        valid_bps = []
        valid_colors = []
        
        for bp_idx, bp in enumerate(BP_LEVELS):
            sharpes = all_metrics[asset][bp]['sharpes']
            trpt_values = all_metrics[asset][bp]['trade_ret_per_turnover']
            
            if sharpes:
                sharpe_data.append(sharpes)
                trpt_data.append(trpt_values)
                valid_bps.append(bp)
                valid_colors.append(BP_COLORS[bp])
        
        if sharpe_data:
            # Plot Sharpe boxplots
            bp1 = ax1.boxplot(
                sharpe_data,
                positions=positions[:len(sharpe_data)],
                widths=0.6,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(marker='o', markersize=3, alpha=0.4),
                medianprops=dict(color='black', linewidth=1.5),
            )
            
            for patch, color in zip(bp1['boxes'], valid_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Plot TR/Turnover boxplots
            bp2 = ax2.boxplot(
                trpt_data,
                positions=positions[:len(trpt_data)],
                widths=0.6,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(marker='o', markersize=3, alpha=0.4),
                medianprops=dict(color='black', linewidth=1.5),
            )
            
            for patch, color in zip(bp2['boxes'], valid_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
    
    # Styling Panel A
    ax1.set_ylabel('Annualized Sharpe Ratio', fontsize=11)
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.grid(axis='y', alpha=0.3, linestyle=':')
    
    # Styling Panel B
    ax2.set_ylabel('Trade Return / Turnover', fontsize=11)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.grid(axis='y', alpha=0.3, linestyle=':')
    
    # Set x-axis labels
    asset_centers = []
    for idx in range(len(ASSETS)):
        base_pos = idx * (group_width + spacing) + 1
        center = base_pos + (len(BP_LEVELS) - 1) * 0.8 / 2
        asset_centers.append(center)
    
    ax1.set_xticks(asset_centers)
    ax1.set_xticklabels([a.replace(' ', '\n') for a in ASSETS], fontsize=10)
    ax2.set_xticks(asset_centers)
    ax2.set_xticklabels([a.replace(' ', '\n') for a in ASSETS], fontsize=10)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, facecolor=BP_COLORS[bp], alpha=0.7, label=f'BP={bp}')
        for bp in BP_LEVELS
    ]
    ax1.legend(
        handles=legend_elements,
        loc='upper right',
        title='Transaction Cost',
        fontsize=9,
        title_fontsize=9,
        framealpha=0.9,
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def print_summary_statistics(all_metrics: Dict):
    """Print summary statistics for all metrics."""
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    
    for asset in ASSETS:
        print(f"\n{asset}:")
        print("-" * 40)
        
        for bp in BP_LEVELS:
            sharpes = all_metrics[asset][bp]['sharpes']
            trpt_values = all_metrics[asset][bp]['trade_ret_per_turnover']
            
            if sharpes:
                print(f"  BP={bp:2d}: n={len(sharpes):2d} | "
                      f"Sharpe: {np.mean(sharpes):6.3f} ± {np.std(sharpes):5.3f} | "
                      f"TR/Turn: {np.mean(trpt_values):9.6f} ± {np.std(trpt_values):9.6f}")
            else:
                print(f"  BP={bp:2d}: No data")


def main():
    print("=" * 80)
    print("Exhibit 4: Per-Contract Analysis — Sharpe and Trade Return/Turnover")
    print("=" * 80)
    print(f"Assets: {ASSETS}")
    print(f"BP Levels: {BP_LEVELS}")
    print(f"Reports Directory: {REPORTS_DIR}")
    print(f"Output Path: {OUTPUT_PATH}")
    print("=" * 80)
    
    # Load and compute all metrics
    all_metrics = load_all_metrics()
    
    # Print summary statistics
    print_summary_statistics(all_metrics)
    
    # Create figure (using vertical two-panel layout)
    print("\nGenerating two-panel figure...")
    fig = create_vertical_two_panel(all_metrics)
    
    # Save outputs
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Save PNG
    fig.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"\nPNG saved to: {OUTPUT_PATH}")
    
    # Save PDF
    pdf_path = OUTPUT_PATH.with_suffix('.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"PDF saved to: {pdf_path}")
    
    plt.close(fig)
    print("\nDone!")


if __name__ == '__main__':
    main()
