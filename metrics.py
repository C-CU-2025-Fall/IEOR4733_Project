"""
metrics.py — 9 portfolio metrics (single source of truth)

Paper: Zhang, Zohren, Roberts (2019) Section 4.4

Framework: ADDITIVE profits on p0-normalized prices.
  r_t = p_t - p_{t-1}  (additive)
  Wealth = N × W_0 + cumsum(R_eq)

Metrics (per paper Section 4.4):
  1. E(R)     — mean(R) × 252
  2. std(R)   — std(R) × √252
  3. DD       — std(R[R<0]) × √252  [paper: "annualised std of negative returns"]
  4. Sharpe   — E(R) / std(R)
  5. Sortino  — E(R) / DD
  6. MDD      — max drawdown from additive wealth path
  7. Calmar   — realised annual return / MDD
  8. % +ve    — fraction of positive return days
  9. Ave P/L  — mean(R>0) / |mean(R<0)|

Notes on MDD/Calmar:
  Paper does not specify initial wealth level. We use N × W_0 as the initial
  wealth (sum of per-contract initial capital), which empirically produces
  values closest to the paper's reported MDD and Calmar.
"""
import numpy as np
from config import TRADING_DAYS

T = TRADING_DAYS
W0 = 1.0  # initial wealth per contract

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
    R_eq = np.asarray(R_eq, dtype=float)
    n_years = len(R_eq) / T

    er = R_eq.mean() * T
    std = R_eq.std(ddof=0) * np.sqrt(T)

    # DD: std of negative returns only [paper: "std of trade returns that are negative"]
    neg = R_eq[R_eq < 0]
    dd = neg.std(ddof=0) * np.sqrt(T) if len(neg) > 0 else 0.0

    sharpe = er / std if std > 0 else 0.0
    sortino = er / dd if dd > 0 else 0.0

    pct_pos = (R_eq > 0).mean()

    pos_r = R_eq[R_eq > 0]
    neg_r = R_eq[R_eq < 0]
    avg_pl = (
        pos_r.mean() / abs(neg_r.mean())
        if len(pos_r) > 0 and len(neg_r) > 0 else 0.0
    )

    # MDD: from additive wealth path with init_wealth = N × W_0
    cumret = np.cumsum(R_eq)
    wealth = n_contracts * w0 + cumret
    peak = np.maximum.accumulate(wealth)
    mdd = float(np.max((peak - wealth) / peak))

    # Calmar: realised annual return / MDD
    realised_ann = (wealth[-1] - wealth[0]) / wealth[0] / n_years
    calmar = realised_ann / mdd if mdd > 0 else 0.0

    return [round(v, 3) for v in
            [er, std, dd, sharpe, sortino, mdd, calmar, pct_pos, avg_pl]]
