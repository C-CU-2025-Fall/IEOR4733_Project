#!/usr/bin/env python3
"""Evaluate 3 normalized F0 alternatives against the 8 non-F0 features."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from drl_shared.spec import universe_tickers, MARKET_FEATURE_DIM

FEATURE_NAMES = [
    "F1: ret_21d", "F2: ret_42d", "F3: ret_63d", "F4: ret_252d",
    "F5: MACD(8,24)", "F6: MACD(16,48)", "F7: MACD(32,96)", "F8: RSI(30)",
]

ALT_NAMES = [
    "A: zscore_price(p,60)",
    "B: range_position(p,60)",
    "C: ret_1d / sigma",
]


def load_all(tickers):
    from drl.dqn.spec import contract_data_path
    all_prices, all_sigma, all_returns, all_feats = [], [], [], []
    for t in tickers:
        for r in (1, 2):
            p = contract_data_path(r, t)
            if not p.exists():
                continue
            d = np.load(p, allow_pickle=True)
            all_prices.append(d["prices"])
            all_sigma.append(d["sigma"])
            all_returns.append(d["returns"])
            all_feats.append(d["features"])
    prices = np.concatenate(all_prices)
    sigma = np.concatenate(all_sigma)
    returns = np.concatenate(all_returns)
    feats = np.concatenate(all_feats)  # (n, 9), F1-F8 = cols 1-8
    # Drop F0
    feats = feats[:, 1:]
    return prices, sigma, returns, feats


def compute_alts(prices, sigma, returns):
    n = len(prices)
    # A: z-score (p_t - mean_60) / std_60
    roll_mean = pd.Series(prices).rolling(60, min_periods=5).mean().to_numpy(dtype=float)
    roll_std = pd.Series(prices).rolling(60, min_periods=5).std().to_numpy(dtype=float)
    alt_a = (prices - roll_mean) / (roll_std + 1e-10)

    # B: range position [0,1] → [-1,1]
    roll_min = pd.Series(prices).rolling(60, min_periods=5).min().to_numpy(dtype=float)
    roll_max = pd.Series(prices).rolling(60, min_periods=5).max().to_numpy(dtype=float)
    pos = (prices - roll_min) / (roll_max - roll_min + 1e-10)
    alt_b = 2.0 * pos - 1.0  # [-1, 1]

    # C: 1-day vol-adjusted return: r_t / sigma_t
    alt_c = returns / (sigma + 1e-10)

    # Clip extremes
    for arr in (alt_a, alt_b, alt_c):
        arr[:] = np.nan_to_num(arr, nan=0.0, posinf=3.0, neginf=-3.0)
        arr[:] = np.clip(arr, -5, 5)

    return np.column_stack([alt_a, alt_b, alt_c])


def report(name, arr, feats8):
    print(f"\n{'─' * 70}")
    print(f"  {name}")
    print(f"  Scale: mean={arr.mean():.3f}, std={arr.std():.3f}, min={arr.min():.2f}, max={arr.max():.2f}")

    # Correlation with each F1-F8
    corrs = [np.corrcoef(arr, feats8[:, i])[0, 1] for i in range(8)]
    print(f"  Corr with F1-F8:")
    for i, fn in enumerate(FEATURE_NAMES):
        flag = " ⚠️ high" if abs(corrs[i]) > 0.7 else ""
        print(f"    {fn:<22s}  r = {corrs[i]:+.4f}{flag}")
    print(f"  Mean |r| with others = {np.mean(np.abs(corrs)):.4f}")
    print(f"  Max  |r| with others = {np.max(np.abs(corrs)):.4f}")

    # VIF if we add this as a 9th feature
    X = np.column_stack([feats8, arr])
    d = X.shape[1]
    vif = {}
    for j in range(d):
        y = X[:, j]
        X_j = np.delete(X, j, axis=1)
        from sklearn.linear_model import LinearRegression
        r2 = LinearRegression().fit(X_j, y).score(X_j, y)
        vif[j] = 1.0 / (1.0 - r2 + 1e-10)
    print(f"  VIF if added: {vif[d-1]:.2f} (ideal <5)")


def main():
    tickers = universe_tickers("Forex")
    print(f"Loading {len(tickers)} contracts (r1+r2)...")
    prices, sigma, returns, feats8 = load_all(tickers)
    print(f"Total rows: {len(prices):,}")

    alts = compute_alts(prices, sigma, returns)

    # Original F0 stats for comparison
    from drl.dqn.spec import contract_data_path
    d = np.load(contract_data_path(1, "AN"), allow_pickle=True)
    f0_orig = d["features"][:, 0]
    print(f"\n─── Baseline: Original F0 (p/rollstd60) ───")
    print(f"  Scale: mean={f0_orig.mean():.1f}, std={f0_orig.std():.1f}, "
          f"min={f0_orig.min():.1f}, max={f0_orig.max():.1f}")
    orig_corrs = [np.corrcoef(f0_orig, d["features"][:, 1:][:, i])[0, 1] for i in range(8)]
    print(f"  Max |r| with F1-F8: {np.max(np.abs(orig_corrs)):.4f}")

    # Test each alternative
    for i in range(3):
        # Thinner sample on full data for corr/VIF
        sub = alts[::5, i]
        sub_feats = feats8[::5, :]
        report(ALT_NAMES[i], sub, sub_feats)

    # PCA: how many independent dimensions do we get adding each?
    print(f"\n{'─' * 70}")
    print(f"  PCA: 8D baseline vs 9D with each candidate")
    from sklearn.decomposition import PCA

    for i in range(3):
        X = np.column_stack([feats8[::10, :], alts[::10, i]])
        pca = PCA().fit(X)
        cumsum = pca.explained_variance_ratio_.cumsum()
        n90 = np.searchsorted(cumsum, 0.90) + 1
        n95 = np.searchsorted(cumsum, 0.95) + 1
        ev_new = pca.explained_variance_ratio_[0]
        print(f"  {chr(65+i)}: PC1={ev_new:.2%}, {n95} PCs for 95% var")


if __name__ == "__main__":
    main()
