"""
strategies.py — Trading signal functions
Reference: [4] Baz et al. 2015 for MACD; [37] Moskowitz et al. 2012 for Sign(R)

Note on return conventions:
- Trading framework uses additive returns: r_t = p_t - p_{t-1} (p0-normalized)
- Sign(R) signal uses PERCENTAGE returns for the lookback (Moskowitz convention)
- MACD signal uses raw prices for EMA computation (Baz convention)
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
# Strategy 2: Sign(R) — [37] Moskowitz et al. 2012; [27] Lim et al. 2019
# =============================================================================
def strategy_sign_r(pct_returns, lookback=SIGN_LOOKBACK):
    """
    Sign(R): A_t = sign(r_{t-252:t})  [Paper Eq 10]
    
    Signal is the sign of cumulative percentage return over past `lookback` days.
    Caller should pass PERCENTAGE returns (pct_change), not additive returns.
    
    Position ∈ {-1, 0, +1}.
    
    Reference: [37] Moskowitz, Ooi, Pedersen 2012; [27] Lim, Zohren, Roberts 2019
    """
    positions = np.zeros(len(pct_returns))
    for t in range(lookback, len(pct_returns)):
        cum_ret = np.sum(pct_returns[t - lookback:t])
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
    
    Paper Eq 3: MACD_t = q_t / std(q_{t-252:t})
    Paper Eq 11-12: A_t = φ(MACD_tilde), where MACD_tilde = avg across pairs
    
    For each (S, L) pair:
      q_t = (EMA(P, S) - EMA(P, L)) / std(P_{t-63:t})
      MACD_t = q_t / std(q_{t-252:t})
    
    Average MACD_t across pairs, then apply:
      φ(x) = x * exp(-x²/4) / 0.89
    
    Position ∈ [-1, +1].
    
    Reference: [4] Baz et al. 2015
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
            positions[t] = m * np.exp(-m ** 2 / 4) / 0.89

    return positions
