"""
metrics.py — 9 portfolio metrics (single source of truth)

Paper: Zhang, Zohren, Roberts (2019) Section 4.4
Reference: [27] Lim et al. (Deep Momentum Networks)

Framework: ADDITIVE profits on p0-normalized prices.
  r_t = p_t - p_{t-1}  (additive)
  Wealth = N × W_0 + cumsum(R_eq)

Metrics (per paper Section 4.4, "as suggested in [27]"):
  1. E(R)     — mean(R) × 252
  2. std(R)   — std(R) × √252
  3. DD       — sqrt(mean(min(R,0)²)) × √252  [zero-target downside deviation]
  4. Sharpe   — E(R) / std(R)
  5. Sortino  — E(R) / DD
  6. MDD      — max drawdown from additive wealth path
  7. Calmar   — realised annual return / MDD
  8. % +ve    — fraction of positive return days
  9. Ave P/L  — mean(R>0) / |mean(R<0)|

Notes:
  - DD uses zero-target LPM(2) per standard Sortino framework (MAR=0).
    This is more orthodox than std(R[R<0]) per CFA guidance.
  - Calmar uses realised_ann/MDD. While [27] says "compares expected annual return
    with MDD", using E(R)/MDD with N×W0 init_wealth produces values 10x off paper.
    realised_ann/MDD keeps numerator and denominator on the same wealth scale.
  - MDD uses N×W_0 as init_wealth (empirically closest to paper values).
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

    # DD: zero-target downside deviation (LPM(2) with MAR=0)
    # Per standard Sortino framework and [27] "Downside Deviation"
    shortfall = np.minimum(R_eq, 0.0)
    dd = np.sqrt(np.mean(shortfall ** 2)) * np.sqrt(T)

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
    # Using realised_ann (not E(R)) so numerator scales with init_wealth like MDD.
    # This empirically matches paper values; E(R)/MDD would be 10x off due to
    # init_wealth scaling mismatch.
    realised_ann = (wealth[-1] - wealth[0]) / wealth[0] / n_years
    calmar = realised_ann / mdd if mdd > 0 else 0.0

    return [round(v, 3) for v in
            [er, std, dd, sharpe, sortino, mdd, calmar, pct_pos, avg_pl]]
