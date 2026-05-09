#!/usr/bin/env python3
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = REPO_ROOT / "drl" / "dqn" / "models"
OUTPUT_DIR = REPO_ROOT / "drl" / "dqn" / "figures"
DATA_DIR = OUTPUT_DIR / "data"

ASSETS = ["Commodity", "Equity_Index", "Fixed_Income", "Forex"]
ASSET_DISPLAY_NAMES = {
    "Commodity": "Commodity",
    "Equity_Index": "Equity Index", 
    "Fixed_Income": "Fixed Income",
    "Forex": "Forex"
}

BP_LEVELS = [1, 10, 20, 30, 45]
SELECTED_SEEDS = [42, 45, 48]
ALL_SEEDS = list(range(42, 52))


def extract_seed_from_dirname(dirname):
    match = re.search(r'_s(\d+)$', dirname)
    if match:
        return int(match.group(1))
    return None


def extract_bp_from_dirname(dirname):
    match = re.search(r'^bp(\d+)_', dirname)
    if match:
        return int(match.group(1))
    return None


def get_validation_metric(model_dir):
    validation_file = model_dir / "validation_metrics.csv"
    if validation_file.exists():
        try:
            df = pd.read_csv(validation_file)
            if 'validation_reward_mean' in df.columns and len(df) > 0:
                return float(df['validation_reward_mean'].iloc[-1])
            if 'best_validation_reward' in df.columns and len(df) > 0:
                return float(df['best_validation_reward'].iloc[-1])
        except Exception:
            pass
    
    manifest_file = model_dir / "manifest.json"
    if manifest_file.exists():
        try:
            with open(manifest_file) as f:
                manifest = json.load(f)
            if 'best_val_reward' in manifest:
                return float(manifest['best_val_reward'])
        except Exception:
            pass
    
    return None


def collect_seed_data_for_asset(asset_slug):
    asset_dir = MODEL_ROOT / asset_slug / "r1"
    if not asset_dir.exists():
        print(f"  Warning: {asset_dir} does not exist")
        return {}
    
    bp_data = {bp: {} for bp in BP_LEVELS}
    
    for model_dir in asset_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        bp = extract_bp_from_dirname(model_dir.name)
        seed = extract_seed_from_dirname(model_dir.name)
        
        if bp is None or seed is None:
            continue
        
        if bp not in BP_LEVELS:
            continue
        
        metric = get_validation_metric(model_dir)
        if metric is not None and not np.isnan(metric):
            bp_data[bp][seed] = metric
    
    return bp_data


def rank_seeds_within_bp(bp_data):
    rankings = {}
    
    for bp, seed_metrics in bp_data.items():
        if not seed_metrics:
            rankings[bp] = []
            continue
        
        sorted_seeds = sorted(seed_metrics.items(), key=lambda x: x[1], reverse=True)
        ranked = [(seed, metric, rank+1) for rank, (seed, metric) in enumerate(sorted_seeds)]
        rankings[bp] = ranked
    
    return rankings


def create_ranking_table(all_asset_rankings):
    records = []
    
    for seed in SELECTED_SEEDS:
        row = {'Seed': seed}
        for bp in BP_LEVELS:
            ranks = []
            for asset in ASSETS:
                if bp in all_asset_rankings.get(asset, {}):
                    rankings = all_asset_rankings[asset][bp]
                    for s, metric, rank in rankings:
                        if s == seed:
                            ranks.append(rank)
                            break
            
            if ranks:
                row[f'BP{bp}'] = np.mean(ranks)
            else:
                row[f'BP{bp}'] = np.nan
        
        records.append(row)
    
    df = pd.DataFrame(records)
    return df


def create_detailed_ranking_table(all_asset_rankings):
    records = []
    
    for asset in ASSETS:
        asset_display = ASSET_DISPLAY_NAMES[asset]
        for seed in SELECTED_SEEDS:
            row = {'Asset': asset_display, 'Seed': seed}
            for bp in BP_LEVELS:
                if bp in all_asset_rankings.get(asset, {}):
                    rankings = all_asset_rankings[asset][bp]
                    for s, metric, rank in rankings:
                        if s == seed:
                            row[f'BP{bp}'] = rank
                            row[f'BP{bp}_Sharpe'] = metric
                            break
                    else:
                        row[f'BP{bp}'] = np.nan
                        row[f'BP{bp}_Sharpe'] = np.nan
                else:
                    row[f'BP{bp}'] = np.nan
                    row[f'BP{bp}_Sharpe'] = np.nan
            records.append(row)
    
    df = pd.DataFrame(records)
    return df


def create_visualization(df, output_path):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    bp_cols = [f'BP{bp}' for bp in BP_LEVELS]
    data_matrix = df[bp_cols].values
    
    im = ax.imshow(data_matrix, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=10)
    
    ax.set_xticks(np.arange(len(bp_cols)))
    ax.set_yticks(np.arange(len(df)))
    ax.set_xticklabels([f'{bp} bps' for bp in BP_LEVELS])
    ax.set_yticklabels([f'Seed {int(seed)}' for seed in df['Seed']])
    
    for i in range(len(df)):
        for j in range(len(bp_cols)):
            value = data_matrix[i, j]
            if not np.isnan(value):
                ax.text(j, i, f'{value:.1f}',
                       ha="center", va="center", color="black", fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Rank (1=best)', rotation=270, labelpad=20)
    
    ax.set_title('Seed Ranking Consistency Across Transaction Cost Levels\n(Lower rank = better Sharpe ratio)', 
                 fontsize=12, fontweight='bold', pad=20)
    ax.set_xlabel('Transaction Cost (BP)', fontsize=11)
    ax.set_ylabel('Seed', fontsize=11)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved heatmap: {output_path}")


def create_table_visualization(df, output_path):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    ax.axis('tight')
    ax.axis('off')
    
    bp_cols = [f'BP{bp}' for bp in BP_LEVELS]
    
    table_data = []
    for _, row in df.iterrows():
        table_row = [
            row['Asset'],
            f"Seed {int(row['Seed'])}",
        ]
        for bp in BP_LEVELS:
            rank = row[f'BP{bp}']
            metric = row[f'BP{bp}_Sharpe']
            if not pd.isna(rank):
                table_row.append(f"{int(rank)} ({metric:.2f})")
            else:
                table_row.append("N/A")
        table_data.append(table_row)
    
    columns = ['Asset', 'Seed'] + [f'{bp} bps' for bp in BP_LEVELS]
    
    table = ax.table(cellText=table_data, colLabels=columns,
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.12] + [0.15]*len(BP_LEVELS))
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data) + 1):
        for j in range(2, len(columns)):
            cell_text = table_data[i-1][j]
            if cell_text != "N/A":
                rank = int(cell_text.split()[0])
                if rank <= 3:
                    table[(i, j)].set_facecolor('#C8E6C9')
                elif rank <= 6:
                    table[(i, j)].set_facecolor('#FFF9C4')
                else:
                    table[(i, j)].set_facecolor('#FFCDD2')
    
    plt.title('Seed Rankings Across Transaction Cost Levels by Asset\n(Rank shown with Sharpe ratio in parentheses)', 
              fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved table: {output_path}")


def main():
    print("=" * 70)
    print("Seed Cross-BP Ranking Analysis")
    print("=" * 70)
    print()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    all_asset_rankings = {}
    
    for asset in ASSETS:
        print(f"Processing {ASSET_DISPLAY_NAMES[asset]}...")
        bp_data = collect_seed_data_for_asset(asset)
        
        if not bp_data:
            print(f"  No data found for {asset}")
            continue
        
        for bp in BP_LEVELS:
            n_seeds = len(bp_data.get(bp, {}))
            print(f"  BP {bp}: {n_seeds} seeds")
        
        rankings = rank_seeds_within_bp(bp_data)
        all_asset_rankings[asset] = rankings
        
        for bp in BP_LEVELS:
            if bp in rankings and rankings[bp]:
                print(f"  BP {bp} top 3: {rankings[bp][:3]}")
    
    print()
    
    print("Creating summary ranking table (averaged across assets)...")
    summary_df = create_ranking_table(all_asset_rankings)
    
    summary_csv_path = DATA_DIR / "seed_ranking_analysis.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"  Saved: {summary_csv_path}")
    print()
    print("Summary Table (Averaged Ranks Across Assets):")
    print(summary_df.to_string(index=False))
    print()
    
    print("Creating detailed ranking table (by asset)...")
    detailed_df = create_detailed_ranking_table(all_asset_rankings)
    
    detailed_csv_path = DATA_DIR / "seed_ranking_analysis_detailed.csv"
    detailed_df.to_csv(detailed_csv_path, index=False)
    print(f"  Saved: {detailed_csv_path}")
    print()
    
    print("Creating visualizations...")
    
    heatmap_path = OUTPUT_DIR / "seed_cross_bp_ranking.png"
    create_visualization(summary_df, heatmap_path)
    
    heatmap_pdf_path = OUTPUT_DIR / "seed_cross_bp_ranking.pdf"
    create_visualization(summary_df, heatmap_pdf_path)
    
    table_path = OUTPUT_DIR / "seed_cross_bp_ranking_table.png"
    create_table_visualization(detailed_df, table_path)
    
    table_pdf_path = OUTPUT_DIR / "seed_cross_bp_ranking_table.pdf"
    create_table_visualization(detailed_df, table_pdf_path)
    
    print()
    print("=" * 70)
    print("Analysis complete!")
    print(f"Output files:")
    print(f"  - {summary_csv_path}")
    print(f"  - {detailed_csv_path}")
    print(f"  - {heatmap_path}")
    print(f"  - {heatmap_pdf_path}")
    print(f"  - {table_path}")
    print(f"  - {table_pdf_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
