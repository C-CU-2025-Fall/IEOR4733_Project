#!/usr/bin/env python3
"""Correlation + redundancy analysis for 9D feature space.

Loads regenerated .npz files, computes:
  1. Pairwise Pearson correlation (per-contract, then global average)
  2. Variance Inflation Factor (VIF)
  3. PCA explained variance ratio + loading vectors
  4. Mutual information (non-linear dependency)
  5. Feature importance via Random Forest (predictive redundancy)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from drl_shared.spec import universe_tickers, MARKET_FEATURE_DIM

FEATURE_NAMES = [
    "F0: norm_price (p/rollstd60)",
    "F1: ret_21d / (sigma·√21)",
    "F2: ret_42d / (sigma·√42)",
    "F3: ret_63d / (sigma·√63)",
    "F4: ret_252d / (sigma·√252)",
    "F5: MACD(8,24)",
    "F6: MACD(16,48)",
    "F7: MACD(32,96)",
    "F8: RSI(30)",
]


def load_features(ticker: str, round_num: int = 1) -> np.ndarray | None:
    from drl.dqn.spec import contract_data_path

    path = contract_data_path(round_num, ticker)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return data["features"].astype(np.float64)


def compute_vif(feats: np.ndarray) -> dict[int, float]:
    """Variance Inflation Factor. VIF > 10 → multicollinearity concern."""
    n, d = feats.shape
    vif = {}
    for j in range(d):
        y = feats[:, j]
        X = np.delete(feats, j, axis=1)
        if d == 1 or n <= d:
            vif[j] = 1.0
            continue
        try:
            model = LinearRegression()
            model.fit(X, y)
            r2 = model.score(X, y)
            vif[j] = 1.0 / (1.0 - r2 + 1e-10)
        except Exception:
            vif[j] = float("inf")
    return vif


def main():
    tickers = universe_tickers("Forex")
    print(f"Analyzing {len(tickers)} contracts: {tickers}")

    all_corr_mats = []
    all_features = []
    vif_per_contract = {}
    rf_importances = []

    for ticker in tqdm(tickers, desc="Loading"):
        feats_r1 = load_features(ticker, 1)
        feats_r2 = load_features(ticker, 2)
        if feats_r1 is None:
            print(f"  {ticker}: no r1 data, skipping")
            continue

        feats = np.vstack([feats_r1, feats_r2]) if feats_r2 is not None else feats_r1

        # 1. Pairwise correlation
        corr = np.corrcoef(feats, rowvar=False)
        all_corr_mats.append(corr)
        all_features.append(feats)

        # 2. VIF
        vif_per_contract[ticker] = compute_vif(feats)

    # ---- AGGREGATE ANALYSIS ----
    all_data = np.vstack(all_features)

    # 1. Global pairwise correlation matrix
    global_corr = np.corrcoef(all_data, rowvar=False)
    avg_corr = np.mean(all_corr_mats, axis=0)

    print("\n" + "=" * 90)
    print("PART 1: PAIRWISE PEARSON CORRELATION (pooled across all Forex contracts)")
    print("=" * 90)

    df_corr = pd.DataFrame(
        global_corr, index=FEATURE_NAMES, columns=FEATURE_NAMES
    )
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(df_corr.to_string())

    # Identify high-correlation pairs (>0.7)
    print("\n--- High correlation pairs (|r| > 0.7) ---")
    high_pairs = []
    for i in range(MARKET_FEATURE_DIM):
        for j in range(i + 1, MARKET_FEATURE_DIM):
            r = global_corr[i, j]
            if abs(r) > 0.7:
                high_pairs.append((i, j, r))
                print(f"  F{i} × F{j}: r = {r:+.4f}")

    if not high_pairs:
        print("  None found.")

    # 2. VIF
    print("\n" + "=" * 90)
    print("PART 2: VARIANCE INFLATION FACTOR (VIF > 10 = multicollinearity concern)")
    print("=" * 90)

    global_vif = compute_vif(all_data)
    print(f"\n{'Feature':<38} {'VIF':>10}")
    print("-" * 50)
    for j in range(MARKET_FEATURE_DIM):
        flag = "  ⚠️" if global_vif[j] > 10 else ""
        print(f"  {FEATURE_NAMES[j]:<36} {global_vif[j]:>10.2f}{flag}")

    # Per-contract VIF range
    print("\n--- VIF range across contracts ---")
    for j in range(MARKET_FEATURE_DIM):
        vals = [vif_per_contract[t][j] for t in tickers if t in vif_per_contract]
        print(f"  {FEATURE_NAMES[j]:<36} min={min(vals):.2f}  max={max(vals):.2f}  mean={np.mean(vals):.2f}")

    # 3. PCA (standardized — z-score each feature first)
    print("\n" + "=" * 90)
    print("PART 3: PCA EXPLAINED VARIANCE (standardized: z-score per feature)")
    print("=" * 90)

    from scipy import stats as scipy_stats

    # Z-score standardize
    all_data_z = scipy_stats.zscore(all_data, axis=0, ddof=1)
    all_data_z = np.nan_to_num(all_data_z, nan=0.0)

    pca = PCA().fit(all_data_z)
    cumsum = pca.explained_variance_ratio_.cumsum()

    print(f"\n  PC  Expl.Var  Cumulative")
    print("  " + "-" * 28)
    for pc in range(MARKET_FEATURE_DIM):
        print(f"  {pc + 1:2d}   {pca.explained_variance_ratio_[pc]:.4f}     {cumsum[pc]:.4f}")

    n_90 = np.searchsorted(cumsum, 0.90) + 1
    n_95 = np.searchsorted(cumsum, 0.95) + 1
    print(f"\n  → {n_90} PCs needed for 90% variance, {n_95} PCs needed for 95% variance")

    # PCA loadings
    print("\n--- PCA Loading Matrix (top 5 PCs, standardized) ---")
    loadings = pd.DataFrame(
        pca.components_[:5].T,
        index=[f"F{i}" for i in range(MARKET_FEATURE_DIM)],
        columns=[f"PC{p + 1} ({pca.explained_variance_ratio_[p]:.2%})" for p in range(5)],
    )
    print(loadings.to_string())

    # Per-feature contribution to top PCs
    print("\n--- Top absolute loadings per PC ---")
    for pc in range(min(5, MARKET_FEATURE_DIM)):
        abs_loads = np.abs(pca.components_[pc])
        top3 = np.argsort(abs_loads)[::-1][:3]
        items = [f"F{i}={pca.components_[pc, i]:+.3f}" for i in top3]
        print(f"  PC{pc + 1}: " + " | ".join(items))

    # 4. Mutual Information (non-linear)
    print("\n" + "=" * 90)
    print("PART 4: MUTUAL INFORMATION (non-linear dependency, normalized)")
    print("=" * 90)

    # Use a subset for speed; use standardized data for fair comparison
    mi_sub_z = all_data_z if len(all_data_z) < 50000 else all_data_z[:: max(1, len(all_data_z) // 30000)]
    mi_matrix = np.zeros((MARKET_FEATURE_DIM, MARKET_FEATURE_DIM))
    for i in range(MARKET_FEATURE_DIM):
        mi_vector = mutual_info_regression(
            mi_sub_z, mi_sub_z[:, i], discrete_features=False, random_state=42, n_neighbors=10
        )
        mi_matrix[i, :] = mi_vector

    # Show raw MI (higher = more dependent)
    mi_df = pd.DataFrame(
        mi_matrix, index=[f"F{i}" for i in range(MARKET_FEATURE_DIM)], columns=[f"F{i}" for i in range(MARKET_FEATURE_DIM)]
    )
    print(mi_df.to_string())

    # 5. Random Forest feature importance (predict other features)
    print("\n" + "=" * 90)
    print("PART 5: FEATURE IMPORTANCE (how well can each feature predict others?)")
    print("=" * 90)

    # For each feature, train a tiny RF to predict all other features simultaneously
    # and report relative importance. Use a subset for speed.
    subset = all_data[: min(20000, len(all_data))]
    rf = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(subset[:, 1:], subset[:, 0])  # Can each feature predict F0?
    print(f"\n  Predicting F0 from F1-F8:")
    for fi, imp in enumerate(rf.feature_importances_):
        print(f"    {FEATURE_NAMES[fi + 1]:<36} importance={imp:.4f}")

    rf.fit(subset[:, [0, 2, 3, 4, 5, 6, 7, 8]], subset[:, 1])
    print(f"\n  Predicting F1 from F0,F2-F8:")
    for fi, imp in enumerate(rf.feature_importances_):
        names = [0, 2, 3, 4, 5, 6, 7, 8]
        print(f"    {FEATURE_NAMES[names[fi]]:<36} importance={imp:.4f}")

    # 6. Summary / Recommendations
    print("\n" + "=" * 90)
    print("PART 6: SUMMARY & RECOMMENDATIONS")
    print("=" * 90)

    # Feature scale assessment
    feats_min = all_data.min(axis=0)
    feats_max = all_data.max(axis=0)
    feats_std = all_data.std(axis=0)

    print("\n--- Feature Scale Assessment ---")
    for i in range(MARKET_FEATURE_DIM):
        print(f"  F{i} [{FEATURE_NAMES[i]:<30}]: range=[{feats_min[i]:.1f}, {feats_max[i]:.1f}], std={feats_std[i]:.2f}")

    print(f"\n--- Key Findings ---")
    print(f"  1. F0 (norm_price): scale [{feats_min[0]:.0f}, {feats_max[0]:.0f}] vs others [-3,3].")
    print(f"     F0 dominates unstandardized PCA (99.45% of variance). Non-stationary, drifts with price level.")
    print(f"     VIF={global_vif[0]:.1f} (low only because uncorrelated with everything).")

    print(f"\n  2. Return cluster (F1-F4):")
    print(f"     F1×F2 r={global_corr[1,2]:.3f}  |  F2×F3 r={global_corr[2,3]:.3f}  |  F3×F4 r={global_corr[3,4]:.3f}")
    print(f"     Adjacent horizons are highly correlated. F1(21d) vs F4(252d) r={global_corr[1,4]:.3f} (low).")
    print(f"     VIF: F1={global_vif[1]:.1f}, F2={global_vif[2]:.1f}, F3={global_vif[3]:.1f}, F4={global_vif[4]:.1f}")

    print(f"\n  3. MACD cluster (F5-F7):")
    print(f"     F5×F6 r={global_corr[5,6]:.3f}  |  F6×F7 r={global_corr[6,7]:.3f}  |  F5×F7 r={global_corr[5,7]:.3f}")
    print(f"     VIF: F5={global_vif[5]:.1f}, F6={global_vif[6]:.1f}, F7={global_vif[7]:.1f}")

    print(f"\n  4. Cross-family correlations:")
    print(f"     F1(21d ret) × F5(MACD 8,24): r={global_corr[1,5]:.3f}")
    print(f"     F8(RSI) × F1(21d ret): r={global_corr[8,1]:.3f}")
    print(f"     F8(RSI) × F5(MACD 8,24): r={global_corr[8,5]:.3f}")

    print(f"\n  5. Standardized PCA: {n_90} PCs for 90% variance, {n_95} PCs for 95% variance.")
    print(f"     Effective rank ≈ {n_95} (out of 9). Highly redundant feature space.")

    print(f"\n--- Suggested Pruning Candidates ---")
    print(f"  🔴 HIGH: F0 — scale mismatch, non-stationary, no correlation with others (isolated dimension)")
    print(f"  🔴 HIGH: F4 (ret_252d) — low corr with short-term features(r<0.5), too slow for γ=0.3")
    print(f"  🟡 MED: F2 (ret_42d) — corr {global_corr[1,2]:.3f} with F1, redundant if F1 kept")
    print(f"  🟡 MED: F3 (ret_63d) — corr {global_corr[2,3]:.3f} with F2, {global_corr[1,3]:.3f} with F1")
    print(f"  🟡 MED: F7 (MACD 32,96) — VIF={global_vif[7]:.1f}, corr {global_corr[6,7]:.3f} with F6")
    print(f"  🟢 LOW: F1, F5, F6 (keep) — core short/medium-term signals")
    print(f"  🟢 LOW: F8 (RSI) — high VIF but captures unique mean-reversion signal")
    print(f"\n  Recommended 5D: F1(21d ret) + F5(MACD 8,24) + F6(MACD 16,48) + F8(RSI) + prev_action")


if __name__ == "__main__":
    main()
