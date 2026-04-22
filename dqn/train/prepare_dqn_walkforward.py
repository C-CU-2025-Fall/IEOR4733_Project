#!/usr/bin/env python3
"""
Prepare DQN training data for walk-forward validation.

Round 1: Train 2005-2010 → Test 2011-2019
Round 2: Train 2006-2015 → Test 2015-2019 (overlap for validation)
"""
import os, sys, numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import SOURCE_OVERRIDES, ASSET_CLASSES, EWMA_SPAN
from data_loader import load_clc_full

OUTPUT_DIR = os.path.join(ROOT, 'data', 'dqn_walkforward')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_features(prices, returns, sigma):
    """Build 8-dim feature matrix with multi-scale MACD."""
    n = len(prices)
    feats = np.zeros((n, 8), dtype=np.float32)
    
    # 1. normalized price
    p_mean, p_std = prices.mean(), prices.std() + 1e-10
    feats[:, 0] = (prices - p_mean) / p_std
    
    # 2-5. multi-horizon returns
    for idx, h in enumerate([21, 42, 63, 252]):
        col = np.zeros(n)
        for i in range(h, n):
            col[i] = (prices[i] - prices[i - h]) / (sigma[i] * np.sqrt(h) + 1e-10)
        feats[:, idx + 1] = col
    
    # 6. Multi-scale MACD (3 scales)
    p_series = pd.Series(prices)
    std63 = p_series.rolling(63, min_periods=1).std().values + 1e-10
    
    for (s, l) in [(8, 24), (16, 48), (32, 96)]:
        ema_s = p_series.ewm(span=s, adjust=False).mean().values
        ema_l = p_series.ewm(span=l, adjust=False).mean().values
        q = (ema_s - ema_l) / std63
        std_q = pd.Series(q).rolling(252, min_periods=1).std().values + 1e-10
        feats[:, 5] += (q / std_q) / 3  # Average across scales
    
    # 7. RSI
    delta = np.diff(prices, prepend=prices[0])
    gain = pd.Series(np.where(delta > 0, delta, 0)).rolling(30).mean().values
    loss = pd.Series(np.where(delta < 0, -delta, 0)).rolling(30).mean().values + 1e-10
    feats[:, 6] = (50 - 50 / (1 + gain / loss)) / 50
    
    # 8. vol ratio
    feats[:, 7] = sigma / (sigma.mean() + 1e-10)
    
    return np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

def prepare_round(ticker, round_name, train_start, train_end, test_start, test_end):
    """Prepare data for one walk-forward round."""
    source = SOURCE_OVERRIDES.get(ticker, 'RAD')
    df = load_clc_full(ticker, source=source, start_date=train_start)
    
    if df is None:
        print(f'  {ticker}: ❌ no data')
        return False
    
    # Training period
    train_mask = (df['Date'] >= train_start) & (df['Date'] <= train_end)
    df_train = df[train_mask].reset_index(drop=True)
    
    if len(df_train) < 500:
        print(f'  {ticker}: ❌ train period too short ({len(df_train)} days)')
        return False
    
    prices = df_train['Close'].values.astype(float)
    returns = np.zeros(len(prices))
    returns[1:] = prices[1:] - prices[:-1]
    sigma = pd.Series(returns).ewm(span=EWMA_SPAN, adjust=False).std().values
    features = build_features(prices, returns, sigma)
    
    # Save training data
    path = os.path.join(OUTPUT_DIR, f"{ticker}_{round_name}_train.npz")
    np.savez_compressed(path,
        prices=prices, returns=returns, sigma=sigma,
        features=features, dates=df_train['Date'].values,
        source=source, round=round_name,
        train_start=train_start, train_end=train_end,
    )
    
    # Test period (for reference, not used in training)
    test_mask = (df['Date'] >= test_start) & (df['Date'] <= test_end)
    df_test = df[test_mask]
    
    print(f'  {ticker}: ✅ train={len(df_train)}d ({train_start}~{train_end}), test={len(df_test)}d ({test_start}~{test_end})')
    return True

def main():
    print("="*70)
    print("Walk-Forward DQN Data Preparation")
    print("="*70)
    
    # Focus on FX first
    tickers = ASSET_CLASSES['Forex']
    print(f"\nAsset Class: Forex ({len(tickers)} contracts)")
    print()
    
    # Round 1: 2005-2009 → 2010-2014
    print("Round 1: Train 2005-2009 (5y) → Test 2010-2014 (5y)")
    print("-"*50)
    for tk in tickers:
        prepare_round(tk, 'r1', '2005-01-01', '2009-12-31', '2010-01-01', '2014-12-31')
    
    print()
    
    # Round 2: 2005-2014 → 2015-2019
    print("Round 2: Train 2005-2014 (10y) → Test 2015-2019 (5y)")
    print("-"*50)
    for tk in tickers:
        prepare_round(tk, 'r2', '2005-01-01', '2014-12-31', '2015-01-01', '2019-12-31')
    
    print()
    print("="*70)
    print(f"Output: {OUTPUT_DIR}/")
    print("="*70)

if __name__ == '__main__':
    main()
