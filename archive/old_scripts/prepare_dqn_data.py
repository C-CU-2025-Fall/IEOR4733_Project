#!/usr/bin/env python3
"""
Prepare DQN training data from latest baseline configuration.

Uses config.py SOURCE_OVERRIDES to load consistent data.
Saves to data/dqn_train/{ticker}.npz for efficient training.

Training period: 2005-2010 (paper-aligned)
Test period: 2011-2019
"""
import os, sys, numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import SOURCE_OVERRIDES, ASSET_CLASSES, EWMA_SPAN
from data_loader import load_clc_full

TRAIN_END = '2010-12-31'
TRAIN_START = '2005-01-01'
DATALOADER_START = '2005-01-01'  # Override default 2009 start date
OUTPUT_DIR = os.path.join(ROOT, 'data', 'dqn_train')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_features(prices, returns, sigma):
    """Build 8-dim feature matrix (same as strategy_dqn.py)."""
    n = len(prices)
    feats = np.zeros((n, 8), dtype=np.float32)
    
    # 1. normalized price
    p_mean, p_std = prices.mean(), prices.std() + 1e-10
    feats[:, 0] = (prices - p_mean) / p_std
    
    # 2-5. multi-horizon returns (vol-adjusted)
    for idx, h in enumerate([21, 42, 63, 252]):
        col = np.zeros(n)
        for i in range(h, n):
            col[i] = (prices[i] - prices[i - h]) / (sigma[i] * np.sqrt(h) + 1e-10)
        feats[:, idx + 1] = col
    
    # 6. MACD
    p_series = pd.Series(prices)
    ema8 = p_series.ewm(span=8, adjust=False).mean().values
    ema24 = p_series.ewm(span=24, adjust=False).mean().values
    std63 = p_series.rolling(63, min_periods=1).std().values + 1e-10
    q = (ema8 - ema24) / std63
    std252 = pd.Series(q).rolling(252, min_periods=1).std().values + 1e-10
    feats[:, 5] = q / std252
    
    # 7. RSI normalized
    delta = np.diff(prices, prepend=prices[0])
    gain = pd.Series(np.where(delta > 0, delta, 0)).rolling(30, min_periods=1).mean().values
    loss = pd.Series(np.where(delta < 0, -delta, 0)).rolling(30, min_periods=1).mean().values + 1e-10
    feats[:, 6] = (50 - 50 / (1 + gain / loss)) / 50
    
    # 8. vol ratio
    feats[:, 7] = sigma / (sigma.mean() + 1e-10)
    
    return np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

def prepare_contract(ticker):
    """Prepare training data for one contract."""
    source = SOURCE_OVERRIDES.get(ticker, 'RAD')
    df = load_clc_full(ticker, source=source, start_date=DATALOADER_START)
    
    if df is None or len(df) < 500:
        print(f"  {ticker}: ❌ no data (source={source})")
        return False
    
    # Filter to training period
    train_mask = (df['Date'] >= TRAIN_START) & (df['Date'] <= TRAIN_END)
    df_train = df[train_mask].reset_index(drop=True)
    
    if len(df_train) < 500:
        print(f"  {ticker}: ❌ only {len(df_train)} training days (need 500+)")
        return False
    
    prices = df_train['Close'].values.astype(float)
    returns = np.zeros(len(prices))
    returns[1:] = prices[1:] - prices[:-1]
    sigma = pd.Series(returns).ewm(span=EWMA_SPAN, adjust=False).std().values
    features = build_features(prices, returns, sigma)
    
    # Save
    path = os.path.join(OUTPUT_DIR, f"{ticker}.npz")
    np.savez_compressed(path,
        prices=prices,
        returns=returns,
        sigma=sigma,
        features=features,
        dates=df_train['Date'].values,
        source=source,
    )
    
    print(f"  {ticker}: ✅ {len(prices)} days, source={source}")
    return True

def main():
    print("="*70)
    print("DQN Training Data Preparation")
    print("="*70)
    print(f"Training period: {TRAIN_START} to {TRAIN_END}")
    print(f"Output: {OUTPUT_DIR}/")
    print(f"Using config.py SOURCE_OVERRIDES")
    print("="*70)
    
    # Get all tickers
    tickers = []
    for tks in ASSET_CLASSES.values():
        tickers.extend(tks)
    
    print(f"\nPreparing {len(tickers)} contracts...")
    ok, fail = 0, 0
    
    for tk in tickers:
        if prepare_contract(tk):
            ok += 1
        else:
            fail += 1
    
    print(f"\n{'='*70}")
    print(f"Done: {ok}/{len(tickers)} prepared, {fail} failed")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
