#!/usr/bin/env python3
"""
Extract Avg P/L and %+ve metrics from ensemble table2_metrics.json files
and create behavior analysis visualization.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Base path for BP level directories
base_path = Path("/home/congge2026/.openclaw/workspace/IEOR4733_Project/drl/dqn/reports/ensemble_table2_bp")
output_data_path = Path("/home/congge2026/.openclaw/workspace/IEOR4733_Project/drl/dqn/figures/data")
output_fig_path = Path("/home/congge2026/.openclaw/workspace/IEOR4733_Project/drl/dqn/figures")

# BP levels to process
bp_levels = [1, 10, 20, 30, 45]

# Extract metrics from all BP levels
all_data = []

for bp in bp_levels:
    json_path = base_path / f"bp{bp}" / "table2_metrics.json"
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract data for each asset and "All"
    assets = ["Commodity", "Equity Index", "Fixed Income", "Forex", "All"]
    
    for asset in assets:
        # Handle naming inconsistency (bp30 uses underscores)
        asset_key = asset
        if asset not in data and asset == "Equity Index":
            asset_key = "Equity_Index"
        elif asset not in data and asset == "Fixed Income":
            asset_key = "Fixed_Income"
        
        if asset_key in data:
            metrics = data[asset_key]["metrics"]
            all_data.append({
                'BP_Level': bp,
                'Asset': asset,
                'Ave_P_L': metrics['Ave P/L'],
                'Pct_Positive': metrics['% +ve'] * 100  # Convert to percentage
            })

# Create DataFrame
df = pd.DataFrame(all_data)

# Save to CSV
csv_path = output_data_path / "bp_behavior_metrics.csv"
df.to_csv(csv_path, index=False)
print(f"Saved CSV to: {csv_path}")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Pivot data for easier plotting
pivot_pl = df.pivot(index='BP_Level', columns='Asset', values='Ave_P_L')
pivot_pct = df.pivot(index='BP_Level', columns='Asset', values='Pct_Positive')

# Plot 1: Average P/L by BP Level
assets_to_plot = ['All', 'Commodity', 'Equity Index', 'Fixed Income', 'Forex']
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3A7D44']

x = np.arange(len(bp_levels))
width = 0.15

for i, asset in enumerate(assets_to_plot):
    if asset in pivot_pl.columns:
        offset = (i - 2) * width
        ax1.bar(x + offset, pivot_pl[asset], width, label=asset, color=colors[i], alpha=0.8)

ax1.set_xlabel('BP Level (Transaction Fee)', fontsize=12)
ax1.set_ylabel('Average Profit/Loss Ratio', fontsize=12)
ax1.set_title('Average P/L Ratio by BP Level', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f'BP{bp}' for bp in bp_levels])
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Break-even')

# Plot 2: % Positive Trades by BP Level
for i, asset in enumerate(assets_to_plot):
    if asset in pivot_pct.columns:
        offset = (i - 2) * width
        ax2.bar(x + offset, pivot_pct[asset], width, label=asset, color=colors[i], alpha=0.8)

ax2.set_xlabel('BP Level (Transaction Fee)', fontsize=12)
ax2.set_ylabel('% Positive Trades', fontsize=12)
ax2.set_title('Percentage of Positive Trades by BP Level', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f'BP{bp}' for bp in bp_levels])
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(axis='y', alpha=0.3)
ax2.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='Random (50%)')

plt.tight_layout()

# Save figures
png_path = output_fig_path / "bp_behavior_analysis.png"
pdf_path = output_fig_path / "bp_behavior_analysis.pdf"

plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, bbox_inches='tight')

print(f"Saved PNG to: {png_path}")
print(f"Saved PDF to: {pdf_path}")

# Print summary table
print("\n" + "="*60)
print("SUMMARY: Behavior Analysis Metrics by BP Level")
print("="*60)
summary_df = df[df['Asset'] == 'All'][['BP_Level', 'Ave_P_L', 'Pct_Positive']]
summary_df.columns = ['BP Level', 'Avg P/L', '% +ve Trades']
print(summary_df.to_string(index=False))
print("="*60)

plt.close()
