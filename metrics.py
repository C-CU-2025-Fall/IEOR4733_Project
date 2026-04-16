"""
metrics.py — Paper-literal additive metrics (single source of truth)

Paper: Zhang, Zohren, Roberts (2019) Section 4.4
Reference: [27] Lim et al. (Deep Momentum Networks)

Framework: ADDITIVE profits on raw prices.
  r_t = p_t - p_{t-1}  (additive)
  R_t = Eq 4 trade return (contract-level, volatility-scaled)
  R_port = (1/N) Σ R_i  (Eq 13, equal-weight portfolio)
  All metrics computed on R_port directly.

Metrics (paper-literal):
  1. E(R)     — mean(R_port) × 252
  2. std(R)   — std(R_port) × √252
  3. DD       — std(R_port[R<0]) × √252  [paper: "annualised std of negative returns"]
  4. Sharpe   — E(R) / std(R)
  5. Sortino  — E(R) / DD
  6. MDD      — max drawdown from additive wealth = W₀ + cumsum(R_port)
  7. Calmar   — E(R) / MDD
  8. % +ve    — fraction of positive R_port days
  9. Ave P/L  — mean(R>0) / |mean(R<0)|

NOTE: Paper's reported Calmar values are internally inconsistent with
E(R)/MDD (e.g., Table 3 Equity Index Long: 0.504/0.127 ≠ 0.466).
We use the paper-literal formula; matching the table exactly is not possible.
"""
import numpy as np

TRADING_DAYS = 252

METRIC_NAMES = [
    'E(R)', 'std(R)', 'DD', 'Sharpe', 'Sortino',
    'MDD', 'Calmar', '% +ve', 'Ave P/L',
]


def compute_metrics(R_port, n_contracts=None, w0=None):
    """Compute all 9 metrics from portfolio daily returns (additive).

    Args:
        R_port:  1D array of equal-weight portfolio daily returns
        n_contracts: number of sleeves/contracts in the portfolio
        w0:      initial wealth for MDD calculation. If omitted, use
                 n_contracts so additive wealth starts at one unit per sleeve.

    Returns:
        list of 9 rounded values [E(R), std, DD, Sharpe, Sortino, MDD, Calmar, %+ve, AveP/L]
    """
    R = np.asarray(R_port, dtype=float)
    R = R[np.isfinite(R)]
    T = TRADING_DAYS
    if w0 is None:
        w0 = float(n_contracts) if n_contracts is not None else 1.0

    # ── Tier A: core return & risk metrics ──
    er = R.mean() * T
    vol = R.std(ddof=0) * np.sqrt(T)
    sharpe = er / vol if vol > 0 else 0.0

    # DD: paper-literal "annualised standard deviation of trade returns that are negative"
    neg = R[R < 0]
    dd = neg.std(ddof=0) * np.sqrt(T) if len(neg) > 1 else 0.0
    sortino = er / dd if dd > 0 else 0.0

    # ── Tier B: distribution metrics ──
    pos = R[R > 0]
    pct_pos = len(pos) / len(R) if len(R) > 0 else 0.0
    avg_pl = (pos.mean() / abs(neg.mean())) if len(pos) > 0 and len(neg) > 0 else 0.0

    # ── MDD: additive wealth path ──
    # wealth = W₀ + cumsum(R_port); MDD = max((peak - wealth) / peak)
    # Use W₀ = N_contracts by default so the additive path starts with
    # one unit of wealth per equal-weight sleeve.
    wealth = w0 + np.cumsum(R)
    peak = np.maximum.accumulate(wealth)
    drawdown = (peak - wealth) / peak
    mdd = float(np.nanmax(drawdown)) if len(drawdown) > 0 else 0.0

    # Calmar: paper-literal E(R) / MDD
    calmar = er / mdd if mdd > 0 else 0.0

    return [round(v, 3) for v in
            [er, vol, dd, sharpe, sortino, mdd, calmar, pct_pos, avg_pl]]
