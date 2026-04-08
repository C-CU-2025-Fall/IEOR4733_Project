"""
metrics.py — 9 portfolio metrics (single source of truth)

Paper: Zhang, Zohren, Roberts (2019) Section 4.4

Framework: ADDITIVE profits on p0-normalized prices.
  r_t = p_t - p_{t-1}  (additive)
  Wealth = N × W_0 + cumsum(R_eq)

Metrics:
  1. E(R)     — mean(R) × 252
  2. std(R)   — std(R) × √252
  3. DD       — sqrt(mean(min(0,R)²)) × √252
  4. Sharpe   — E(R) / std(R)
  5. Sortino  — E(R) / DD
  6. MDD      — max((peak - wealth) / peak)  [running max method]
  7. Calmar   — realised_annual_return / MDD
  8. % +ve    — fraction of positive return days
  9. Ave P/L  — mean(R>0) / |mean(R<0)|
"""
import numpy as np
from config import TRADING_DAYS

T = TRADING_DAYS
METRIC_NAMES = [
    'E(R)', 'std(R)', 'DD', 'Sharpe', 'Sortino',
    'MDD', 'Calmar', '% +ve', 'Ave P/L',
]


def compute_metrics(R_eq, n_contracts, w0=1.0):
    """Compute all 9 metrics from portfolio daily returns.

    Args:
        R_eq:         1D array of equal-weight portfolio daily returns (additive)
        n_contracts:  number of contracts in portfolio
        w0:           initial wealth per contract

    Returns:
        list of 9 rounded values [E(R), std, DD, Sharpe, Sortino, MDD, Calmar, %+ve, AveP/L]
    """
    n_years = len(R_eq) / T  # auto from data length

    er = np.mean(R_eq) * T
    std = np.std(R_eq) * np.sqrt(T)
    dd = np.sqrt(np.mean(np.minimum(R_eq, 0) ** 2)) * np.sqrt(T)
    sharpe = er / std if std > 0 else 0.0
    sortino = er / dd if dd > 0 else 0.0

    pct_pos = np.sum(R_eq > 0) / len(R_eq)
    pos_r = R_eq[R_eq > 0]
    neg_r = R_eq[R_eq < 0]
    avg_pl = (np.mean(pos_r) / abs(np.mean(neg_r))
              if len(pos_r) > 0 and len(neg_r) > 0 else 0.0)

    # MDD: max drawdown from running max of additive wealth
    cumret = np.cumsum(R_eq)
    wealth = n_contracts * w0 + cumret
    peak = np.maximum.accumulate(wealth)
    mdd = float(np.max((peak - wealth) / peak))

    # Calmar: realised annualised return / MDD
    realised_ann = (wealth[-1] - wealth[0]) / wealth[0] / n_years
    calmar = realised_ann / mdd if mdd > 0 else 0.0

    return [round(v, 3) for v in
            [er, std, dd, sharpe, sortino, mdd, calmar, pct_pos, avg_pl]]
