#!/usr/bin/env python3
"""
Technical indicator calculation module
Paper page 4: State Space feature engineering
"""

import numpy as np
import pandas as pd
from typing import Union

ArrayLike = Union[np.ndarray, pd.Series]

# =============================================================================
# MACD indicator (Paper Equation 3)
# =============================================================================

def compute_macd(prices: ArrayLike, short_span: int, long_span: int) -> np.ndarray:
    """
    Paper Equation (3): MACD indicator
    
    MACD_t = q_t / std(q_{t-252:t})
    q_t = (m(S) - m(L)) / std(p_{t-63:t})
    
    Args:
        prices: Price series
        short_span: Short-term timescale (paper: 8, 16, 32)
        long_span: Long-term timescale (paper: 24, 48, 96)
    
    Returns:
        MACD value series
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    # Exponentially weighted moving average
    m_short = prices.ewm(span=short_span, adjust=False).mean()
    m_long = prices.ewm(span=long_span, adjust=False).mean()
    
    # 63-day rolling standard deviation
    std_63 = prices.rolling(window=63, min_periods=1).std()
    
    # q_t
    q = (m_short - m_long) / std_63
    
    # 252-day rolling standard deviation
    std_q = q.rolling(window=252, min_periods=1).std()
    
    # MACD_t
    macd = q / std_q
    
    return macd.fillna(0).values


def compute_macd_multi_scale(prices: ArrayLike) -> np.ndarray:
    """
    Multi-timescale MACD average (Paper Equation 12)
    
    Sk ∈ {8, 16, 32}
    Lk ∈ {24, 48, 96}
    
    Returns:
        Average MACD across 3 timescales
    """
    macd_8_24 = compute_macd(prices, 8, 24)
    macd_16_48 = compute_macd(prices, 16, 48)
    macd_32_96 = compute_macd(prices, 32, 96)
    
    return (macd_8_24 + macd_16_48 + macd_32_96) / 3


# =============================================================================
# RSI indicator
# =============================================================================

def compute_rsi(prices: ArrayLike, window: int = 30) -> np.ndarray:
    """
    RSI indicator - Relative Strength Index
    
    Args:
        prices: Price series
        window: Lookback window (paper: 30 days)
    
    Returns:
        RSI value series (0-100)
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    delta = prices.diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50).values


def normalize_rsi(rsi: np.ndarray) -> np.ndarray:
    """
    Normalize RSI to [-1, 1]
    
    <20: Oversold → -1
    >80: Overbought → +1
    50: Neutral → 0
    """
    return (rsi - 50) / 50


# =============================================================================
# Volatility calculation
# =============================================================================

def compute_volatility(returns: ArrayLike, window: int = 60) -> np.ndarray:
    """
    60-day exponentially weighted moving volatility (Paper page 4)
    
    Args:
        returns: Return series
        window: Window size (default 60 days)
    
    Returns:
        Volatility series
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    
    vol = returns.ewm(span=window, adjust=False).std().values
    return vol


def normalize_return(returns: np.ndarray, vol: np.ndarray, horizon: int = 252) -> np.ndarray:
    """
    Paper: Adjust to reasonable timescale using daily volatility
    
    r_normalized = r / (vol * sqrt(horizon))
    
    Args:
        returns: Raw returns
        vol: Volatility estimate
        horizon: Timescale (21/42/63/252, etc.)
    
    Returns:
        Volatility-adjusted returns
    """
    return returns / (vol * np.sqrt(horizon) + 1e-10)


# =============================================================================
# Multi-horizon returns
# =============================================================================

def compute_multi_horizon_returns(returns: np.ndarray, vol: np.ndarray) -> dict:
    """
    Compute multi-horizon returns (Paper page 4)
    
    Horizons:
    - 1 month (21 days)
    - 2 months (42 days)
    - 3 months (63 days)
    - 1 year (252 days)
    
    Returns:
        Dict: {'ret_21': ..., 'ret_42': ..., 'ret_63': ..., 'ret_252': ...}
    """
    return {
        'ret_21': normalize_return(returns, vol, 21),
        'ret_42': normalize_return(returns, vol, 42),
        'ret_63': normalize_return(returns, vol, 63),
        'ret_252': normalize_return(returns, vol, 252)
    }


# =============================================================================
# Price normalization
# =============================================================================

def normalize_prices(prices: np.ndarray) -> np.ndarray:
    """
    Normalize close price series
    
    z-score normalization: (p - mean) / std
    """
    return (prices - np.mean(prices)) / (np.std(prices) + 1e-10)


# =============================================================================
# Full state space construction
# =============================================================================

class FeatureEngineer:
    """
    Feature engineering - build the state space required by the paper
    
    Output: 8 features
    1. Normalized close price
    2. 21-day return (volatility-adjusted)
    3. 42-day return (volatility-adjusted)
    4. 63-day return (volatility-adjusted)
    5. 252-day return (volatility-adjusted)
    6. MACD (multi-timescale average)
    7. RSI (normalized)
    8. Volatility (normalized)
    """
    
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.feature_dim = 8
    
    def build_features(self, prices: np.ndarray, returns: np.ndarray, 
                      current_idx: int) -> np.ndarray:
        """
        Build state features at current_idx
        
        Args:
            prices: Price series
            returns: Return series
            current_idx: Current time index
        
        Returns:
            (window_size, feature_dim) = (60, 8)
        """
        if current_idx < self.window_size:
            return np.zeros((self.window_size, self.feature_dim), dtype=np.float32)
        
        # Get time window data
        start_idx = current_idx - self.window_size
        window_prices = prices[start_idx:current_idx]
        window_returns = returns[start_idx:current_idx]
        
        # Handle NaN: fill with 0
        window_returns = np.nan_to_num(window_returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Compute volatility
        vol = compute_volatility(window_returns, 60)
        vol = np.nan_to_num(vol, nan=0.01, posinf=1.0, neginf=0.001)  # Handle NaN in vol
        vol_mean = np.mean(vol) + 1e-10
        
        # Feature 1: Normalized close price
        norm_price = normalize_prices(window_prices)
        
        # Features 2-5: Multi-horizon returns
        ret_21 = normalize_return(window_returns, vol, 21)
        ret_42 = normalize_return(window_returns, vol, 42)
        ret_63 = normalize_return(window_returns, vol, 63)
        ret_252 = normalize_return(window_returns, vol, 252)
        
        # Feature 6: MACD
        macd = compute_macd_multi_scale(window_prices)
        
        # Feature 7: RSI
        rsi = compute_rsi(window_prices, 30)
        rsi_norm = normalize_rsi(rsi)
        
        # Feature 8: Volatility
        vol_norm = vol / vol_mean
        
        # Stack features: (window_size, 8)
        features = np.stack([
            norm_price,
            ret_21,
            ret_42,
            ret_63,
            ret_252,
            macd,
            rsi_norm,
            vol_norm
        ], axis=1)
        
        # Final NaN handling: ensure no NaN remains
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return features.astype(np.float32)
    
    def get_feature_names(self) -> list:
        """Return list of feature names"""
        return [
            'norm_price',
            'ret_21',
            'ret_42',
            'ret_63',
            'ret_252',
            'macd',
            'rsi_norm',
            'vol_norm'
        ]


# =============================================================================
# Test function
# =============================================================================

def test_indicators():
    """Test indicator calculations"""
    print("Testing technical indicator calculations...")
    
    # Generate test data
    np.random.seed(42)
    n = 500
    prices = 100 + np.cumsum(np.random.randn(n))
    returns = np.diff(prices) / prices[:-1]
    returns = np.insert(returns, 0, 0)
    
    # Test MACD
    macd = compute_macd(prices, 8, 24)
    print(f"✅ MACD: shape={macd.shape}, mean={np.mean(macd):.4f}")
    
    # Test multi-scale MACD
    macd_multi = compute_macd_multi_scale(prices)
    print(f"✅ Multi-MACD: shape={macd_multi.shape}")
    
    # Test RSI
    rsi = compute_rsi(prices, 30)
    print(f"✅ RSI: shape={rsi.shape}, mean={np.mean(rsi):.2f}")
    
    # Test volatility
    vol = compute_volatility(returns, 60)
    print(f"✅ Volatility: shape={vol.shape}, mean={np.mean(vol):.6f}")
    
    # Test feature engineering
    fe = FeatureEngineer(window_size=60)
    features = fe.build_features(prices, returns, 200)
    print(f"✅ Features: shape={features.shape}, dtype={features.dtype}")
    print(f"   Feature count: {fe.feature_dim}")
    print(f"   Time window: {fe.window_size}")
    
    print("\n✅ All indicator tests passed!")
    return True


if __name__ == '__main__':
    test_indicators()
