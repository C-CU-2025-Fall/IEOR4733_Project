"""
metrics.py — Portfolio metrics with reproduction-aligned Calmar by default

Paper: Zhang, Zohren, Roberts (2019) Section 4.4
Reference: [27] Lim et al. (Deep Momentum Networks)

Framework: ADDITIVE profits on raw prices.
  r_t = p_t - p_{t-1}  (additive)
  R_t = Eq 4 trade return (contract-level, volatility-scaled)
  R_port = (1/N) Σ R_i  (Eq 13, equal-weight portfolio)
  All metrics computed on R_port directly.

Metrics:
  1. E(R)     — mean(R_port) × 252
  2. std(R)   — std(R_port) × √252
  3. DD       — std(R_port[R<0]) × √252  [paper: "annualised std of negative returns"]
  4. Sharpe   — E(R) / std(R)
  5. Sortino  — E(R) / DD
  6. MDD      — max drawdown from additive wealth = W₀ + cumsum(R_port)
  7. Calmar   — default: wealth CAGR / MDD on the same additive wealth path
                 optional literal mode: E(R) / MDD
  8. % +ve    — fraction of positive R_port days
  9. Ave P/L  — mean(R>0) / |mean(R<0)|

NOTE: The paper's reported Calmar values are internally inconsistent with
standard E(R)/MDD. For reproduction, the default here is the better-aligned
additive-wealth CAGR numerator; the old literal formula remains available via
calmar_mode='literal'.
"""
import numpy as np

TRADING_DAYS = 252

METRIC_NAMES = [
    'E(R)', 'std(R)', 'DD', 'Sharpe', 'Sortino',
    'MDD', 'Calmar', '% +ve', 'Ave P/L',
]


def _wealth_cagr_from_additive_path(wealth, w0):
    """Annualize additive-wealth growth as a CAGR-like rate."""
    wealth = np.asarray(wealth, dtype=float)
    wealth = wealth[np.isfinite(wealth)]
    if len(wealth) == 0 or w0 <= 0:
        return 0.0
    final_wealth = wealth[-1]
    if final_wealth <= 0:
        return 0.0
    n_periods = len(wealth)
    return (final_wealth / w0) ** (TRADING_DAYS / n_periods) - 1.0


def additive_wealth_path(pnl_series, w0=1.0):
    """Build an additive wealth path from additive PnL increments."""
    pnl = np.asarray(pnl_series, dtype=float)
    pnl = pnl[np.isfinite(pnl)]
    if len(pnl) == 0:
        return np.asarray([], dtype=float)
    return float(w0) + np.cumsum(pnl)

def cagr_from_path(path, periods_per_year=TRADING_DAYS):
    """Annualized CAGR from a positive wealth/NAV path."""
    arr = np.asarray(path, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    start = arr[0]
    end = arr[-1]
    if start <= 0 or end <= 0:
        return 0.0
    return (end / start) ** (periods_per_year / len(arr)) - 1.0


def max_drawdown_from_path(path):
    """Classical peak-to-trough drawdown on any positive wealth/NAV path."""
    arr = np.asarray(path, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    peak = np.maximum.accumulate(arr)
    drawdown = (peak - arr) / peak
    return float(np.nanmax(drawdown))

def compute_metrics(R_port, n_contracts=None, w0=None, N_contracts=None,
                    calmar_mode='wealth_cagr'):
    """Compute all 9 metrics from portfolio daily returns (additive).

    Args:
        R_port:  1D array of equal-weight portfolio daily returns
        n_contracts: number of sleeves/contracts in the portfolio
        w0:      initial wealth for MDD calculation. If omitted, use
                 n_contracts so additive wealth starts at one unit per sleeve.

    Returns:
        calmar_mode: 'wealth_cagr' (default) or 'literal'

    Returns:
        list of 9 rounded values [E(R), std, DD, Sharpe, Sortino, MDD, Calmar, %+ve, AveP/L]
    """
    R = np.asarray(R_port, dtype=float)
    R = R[np.isfinite(R)]
    T = TRADING_DAYS
    if n_contracts is None and N_contracts is not None:
        n_contracts = N_contracts
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
    wealth = additive_wealth_path(R, w0=w0)
    mdd = max_drawdown_from_path(wealth)

    if calmar_mode == 'wealth_cagr':
        annual_return = _wealth_cagr_from_additive_path(wealth, w0)
    elif calmar_mode == 'literal':
        annual_return = er
    else:
        raise ValueError(f"Unknown calmar_mode: {calmar_mode}")

    calmar = annual_return / mdd if mdd > 0 else 0.0

    return [round(v, 3) for v in
            [er, vol, dd, sharpe, sortino, mdd, calmar, pct_pos, avg_pl]]
