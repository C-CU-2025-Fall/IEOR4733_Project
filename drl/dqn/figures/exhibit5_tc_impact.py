#!/usr/bin/env python3
"""
Exhibit 5: Transaction Cost Impact — Sharpe Ratio & Daily Cost vs BP Level

2-panel figure:
  Panel A: Sharpe Ratio vs BP Level (bps), 4 asset lines
  Panel B: Average Daily Cost Per Contract vs BP Level (bps), 4 asset lines

Auto-detects available BP levels from ensemble_table2_bp/{asset}/bp{XX}/metrics.json.
If only partial data exists, plots available points and adds a watermark.

Usage:
    python3 drl/dqn/figures/exhibit5_tc_impact.py

Output:
    drl/dqn/figures/exhibit5_tc_impact.png
"""
import sys
sys.path.insert(0, '.')

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
REPORTS_BP_ROOT = REPO_ROOT / "drl" / "dqn" / "reports" / "ensemble_table2_bp"
OUTPUT_PATH = REPO_ROOT / "drl" / "dqn" / "figures" / "exhibit5_tc_impact.png"

ASSETS = ["Commodity", "Equity Index", "Fixed Income", "Forex"]
ASSET_SLUGS = {
    "Commodity": "Commodity",
    "Equity Index": "Equity_Index",
    "Fixed Income": "Fixed_Income",
    "Forex": "Forex",
}

TC_BP_LEVELS = [0.0001, 0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040, 0.0045]
TC_BP_BPS = [int(bp * 10000) for bp in TC_BP_LEVELS]

ASSET_COLORS = {
    "Commodity": "#1f77b4",
    "Equity Index": "#ff7f0e",
    "Fixed Income": "#2ca02c",
    "Forex": "#d62728",
}

MARKERS = {
    "Commodity": "o",
    "Equity Index": "s",
    "Fixed Income": "^",
    "Forex": "D",
}

DPI = 150
FIGSIZE = (10, 12)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_exhibit5_data() -> dict:
    """Scan ensemble_table2_bp for available BP-level metrics.

    Returns:
        {
            "Commodity": {
                "bp_20": {"sharpe": -0.96, "avg_daily_cost": 12.5},
                ...
            },
            ...
        }
    """
    data = {}

    for asset in ASSETS:
        slug = ASSET_SLUGS[asset]
        asset_dir = REPORTS_BP_ROOT / slug
        asset_data = {}

        if not asset_dir.exists():
            data[asset] = asset_data
            continue

        for bp_dir in sorted(asset_dir.iterdir()):
            if not bp_dir.is_dir():
                continue
            if not bp_dir.name.startswith("bp"):
                continue

            # "bp20" -> 20 bps
            try:
                bp_bps = int(bp_dir.name[2:])
            except ValueError:
                continue

            metrics_path = bp_dir / "metrics.json"
            if not metrics_path.exists():
                continue

            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            sharpe = None
            if isinstance(metrics, dict):
                m = metrics.get("metrics", metrics)
                if isinstance(m, dict):
                    sharpe = m.get("Sharpe")
                    if sharpe is not None:
                        try:
                            sharpe = float(sharpe)
                        except (TypeError, ValueError):
                            sharpe = None

            # Extract average daily cost per contract
            avg_daily_cost = None

            daily_costs_path = bp_dir / "daily_costs.npz"
            if daily_costs_path.exists():
                try:
                    cost_data = np.load(daily_costs_path, allow_pickle=True)
                    if "avg_daily_cost" in cost_data:
                        avg_daily_cost = float(cost_data["avg_daily_cost"])
                    elif "daily_costs" in cost_data:
                        daily_costs = cost_data["daily_costs"]
                        if len(daily_costs) > 0:
                            avg_daily_cost = float(np.mean(daily_costs))
                except Exception:
                    pass

            key = f"bp_{bp_bps}"
            entry = {}
            if sharpe is not None:
                entry["sharpe"] = sharpe
            if avg_daily_cost is not None:
                entry["avg_daily_cost"] = avg_daily_cost

            if entry:
                asset_data[key] = entry

        data[asset] = asset_data

    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def create_figure(data: dict) -> plt.Figure:
    """Create the 2-panel Exhibit 5 figure."""
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=FIGSIZE, dpi=DPI)

    all_bp_keys = set()
    for asset in ASSETS:
        all_bp_keys.update(data.get(asset, {}).keys())

    if not all_bp_keys:
        # No data at all — show empty axes with watermark
        for ax in (ax_a, ax_b):
            ax.set_xlim(0, 50)
            ax.text(
                0.5, 0.5, "No data available yet",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=16, color="gray", alpha=0.7,
            )
        _style_axes(ax_a, "Panel A: Sharpe Ratio vs Transaction Cost", "Sharpe Ratio")
        _style_axes(ax_b, "Panel B: Avg Daily Cost Per Contract vs Transaction Cost", "Avg Daily Cost ($/day)")
        fig.suptitle(
            "Exhibit 5: Transaction Cost Impact on DQN Performance\n(Phase 1 in progress...)",
            fontsize=14, fontweight="bold", y=0.98,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    bp_levels_sorted = sorted(int(k.split("_")[1]) for k in all_bp_keys)
    n_bp = len(bp_levels_sorted)
    is_partial = n_bp < len(TC_BP_BPS)

    # --- Panel A: Sharpe Ratio vs BP Level ---
    for asset in ASSETS:
        asset_data = data.get(asset, {})
        x_vals = []
        y_vals = []
        for bp_bps in bp_levels_sorted:
            key = f"bp_{bp_bps}"
            if key in asset_data and "sharpe" in asset_data[key]:
                x_vals.append(bp_bps)
                y_vals.append(asset_data[key]["sharpe"])

        if x_vals:
            ax_a.plot(
                x_vals, y_vals,
                color=ASSET_COLORS[asset],
                marker=MARKERS[asset],
                markersize=8,
                linewidth=2,
                label=asset,
            )

    _style_axes(ax_a, "Panel A: Sharpe Ratio vs Transaction Cost", "Sharpe Ratio")
    ax_a.legend(loc="best", fontsize=10, framealpha=0.9)

    # --- Panel B: Avg Daily Cost Per Contract vs BP Level ---
    for asset in ASSETS:
        asset_data = data.get(asset, {})
        x_vals = []
        y_vals = []
        for bp_bps in bp_levels_sorted:
            key = f"bp_{bp_bps}"
            if key in asset_data and "avg_daily_cost" in asset_data[key]:
                x_vals.append(bp_bps)
                y_vals.append(asset_data[key]["avg_daily_cost"])

        if x_vals:
            ax_b.plot(
                x_vals, y_vals,
                color=ASSET_COLORS[asset],
                marker=MARKERS[asset],
                markersize=8,
                linewidth=2,
                label=asset,
            )

    _style_axes(ax_b, "Panel B: Avg Daily Cost Per Contract vs Transaction Cost", "Avg Daily Cost ($/day)")
    has_b_data = any(
        any("avg_daily_cost" in entry for entry in data.get(asset, {}).values())
        for asset in ASSETS
    )
    if has_b_data:
        ax_b.legend(loc="best", fontsize=10, framealpha=0.9)

    # Watermark if partial data
    title_suffix = ""
    if is_partial:
        title_suffix = "\n(Phase 1 in progress...)"
        for ax in (ax_a, ax_b):
            ax.text(
                0.5, 0.5, f"Phase 1 in progress — {n_bp} of {len(TC_BP_BPS)} BP levels",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color="gray", alpha=0.3,
                rotation=30,
            )

    fig.suptitle(
        f"Exhibit 5: Transaction Cost Impact on DQN Performance{title_suffix}",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _style_axes(ax, title: str, ylabel: str):
    """Apply consistent styling to an axis."""
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("BP Level (bps)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(False)
    ax.set_xticks(TC_BP_BPS)
    ax.set_xticklabels([str(b) for b in TC_BP_BPS])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Exhibit 5: Transaction Cost Impact on DQN Performance")
    print("=" * 70)
    print(f"Scanning: {REPORTS_BP_ROOT}")

    data = load_exhibit5_data()

    total_points = sum(len(v) for v in data.values())
    print(f"\nData summary:")
    for asset in ASSETS:
        n = len(data.get(asset, {}))
        print(f"  {asset}: {n} BP levels")
        for key, entry in data.get(asset, {}).items():
            sharpe = entry.get("sharpe", "N/A")
            cost = entry.get("avg_daily_cost", "N/A")
            print(f"    {key}: Sharpe={sharpe}, AvgDailyCost={cost}")

    if total_points == 0:
        print("\n⚠ No BP-level data found in ensemble_table2_bp/.")
        print("  Run backtests first: python drl/dqn/reports/generate_ensemble_table2.py --tc-bp 0.0020")
        print("  Generating figure with empty panels (watermark).")

    fig = create_figure(data)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"\nFigure saved to: {OUTPUT_PATH}")

    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    main()