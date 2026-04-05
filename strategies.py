"""
strategies.py — Trading signal functions
Reference: [4] Baz et al. 2015 for MACD; [27] Lim et al. 2019 for Sign(R)
"""
import numpy as np
import pandas as pd
from config import SIGN_LOOKBACK, MACD_PAIRS, MACD_VOL_WINDOW, MACD_STD_WINDOW


# =============================================================================
# Strategy 1: Long Only
# =============================================================================
def strategy_long_only(n):
    """Long Only: A_t = 1 for all t. Position ∈ {+1}."""
    return np.ones(n)


# =============================================================================
# Strategy 2: Sign(R) — [27] Moskowitz et al. style
# =============================================================================
def strategy_sign_r(returns, lookback=SIGN_LOOKBACK):
    """
    Sign(R): A_t = sign(cumulative return over past `lookback` days).

    Signal computed on percentage returns.
    Position ∈ {-1, 0, +1}.

    Reference: [27] Eq 2-3
    """
    positions = np.zeros(len(returns))
    for t in range(lookback, len(returns)):
        cum_ret = np.prod(1 + returns[t - lookback:t]) - 1
        positions[t] = np.sign(cum_ret)
    return positions


# =============================================================================
# Strategy 3: MACD — [4] Baz et al. 2015
# =============================================================================
def strategy_macd(prices, pairs=MACD_PAIRS,
                  vol_window=MACD_VOL_WINDOW,
                  std_window=MACD_STD_WINDOW):
    """
    MACD: multi-timeframe MACD with φ transformation.

    For each (S, L) pair [4] Eqs 4-6:
      q_t = (EMA(P, S) - EMA(P, L)) / RollingStd(P, 63)
      Y_t = q_t / RollingStd(q, 252)

    Average across pairs, then apply position sizing [4] Eq 7:
      X_t = φ(Y_t) = Y · exp(-Y²/4) / 0.89

    Position ∈ [-1, +1].
    """
    positions = np.zeros(len(prices))
    macd_sum = np.zeros(len(prices))

    for S, L in pairs:
        ema_fast = pd.Series(prices).ewm(span=S, adjust=False).mean()
        ema_slow = pd.Series(prices).ewm(span=L, adjust=False).mean()
        std_vol = pd.Series(prices).rolling(vol_window, min_periods=vol_window).std()
        q = (ema_fast - ema_slow) / std_vol
        std_q = q.rolling(std_window, min_periods=std_window).std()
        macd_sum += (q / std_q).fillna(0).values

    macd_avg = macd_sum / len(pairs)

    for t in range(std_window, len(prices)):
        m = macd_avg[t]
        if not (np.isnan(m) or np.isinf(m)):
            phi = m * np.exp(-m ** 2 / 4) / 0.89
            positions[t] = np.clip(phi, -1, 1)

    return positions
