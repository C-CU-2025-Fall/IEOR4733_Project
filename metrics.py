"""
metrics.py — Single source of truth for all 9 portfolio metrics.

Paper reference: Zhang, Zohren, Roberts (2020) Section 4.4

Framework: ADDITIVE profits on p0-normalized prices.
  r_t = p_t - p_{t-1}  (additive, not percentage)
  Wealth = N × W_0 + cumsum(R_eq)  (additive accumulation)
  
Metrics (9 total):
  1. E(R)     — Annualised expected return: mean(R) × 252
  2. std(R)   — Annualised std: std(R) × √252
  3. DD       — Annualised downside deviation: sqrt(mean(min(0,R)²)) × √252
  4. Sharpe   — E(R) / std(R)
  5. Sortino  — E(R) / DD
  6. MDD      — Max drawdown on additive wealth: max((peak - W) / peak)
  7. Calmar   — realised_annual_return / MDD
  8. % +ve    — Fraction of positive return days
  9. Ave P/L  — Mean positive return / mean |negative return|

All other modules MUST import from this file. No duplicate implementations.
"""
import numpy as np
from config import TRADING_DAYS

METRIC_NAMES = ['E(R)', 'std(R)', 'DD', 'Sharpe', 'Sortino', 'MDD', 'Calmar', '% +ve', 'Ave P/L']


# =============================================================================
# Individual metric functions
# =============================================================================

def compute_expected_return(port_returns, annual_factor=TRADING_DAYS):
    """1. E(R): Annualised expected return = mean(R) × T."""
    return float(np.mean(port_returns) * annual_factor)


def compute_annualized_std(port_returns, annual_factor=TRADING_DAYS):
    """2. std(R): Annualised standard deviation = std(R) × √T."""
    return float(np.std(port_returns) * np.sqrt(annual_factor))


def compute_downside_deviation(port_returns, annual_factor=TRADING_DAYS):
    """3. DD: Annualised downside deviation.
    
    Formula: sqrt(mean(min(0, R)²)) × √T
    Only negative returns contribute to downside risk.
    """
    downside = np.minimum(port_returns, 0)
    return float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(annual_factor))


def compute_sharpe(port_returns, annual_factor=TRADING_DAYS):
    """4. Sharpe Ratio: E(R) / std(R)."""
    er = compute_expected_return(port_returns, annual_factor)
    std_r = compute_annualized_std(port_returns, annual_factor)
    return float(er / std_r) if std_r > 0 else 0.0


def compute_sortino(port_returns, annual_factor=TRADING_DAYS):
    """5. Sortino Ratio: E(R) / DD."""
    er = compute_expected_return(port_returns, annual_factor)
    dd = compute_downside_deviation(port_returns, annual_factor)
    return float(er / dd) if dd > 0 else 0.0


def compute_max_drawdown(port_returns, n_contracts=1, w0=1.0, n_years=9):
    """6. MDD: Maximum Drawdown on additive wealth.
    
    Additive wealth: W_t = n_contracts × w0 + cumsum(R_eq[0:t])
    Peak: running maximum of W
    MDD: max over all t of (peak_t - W_t) / peak_t
    
    Args:
        port_returns: equal-weight portfolio daily returns (additive)
        n_contracts:  number of contracts in portfolio
        w0:           initial wealth per contract
        n_years:      number of years (for fallback)
    """
    cumret = np.cumsum(port_returns)
    wealth = n_contracts * w0 + cumret
    peak = np.maximum.accumulate(wealth)
    drawdowns = (peak - wealth) / np.maximum(peak, 1e-10)
    return float(np.max(drawdowns))


def compute_calmar(port_returns, n_contracts=1, w0=1.0, n_years=9):
    """7. Calmar Ratio: realised_annual_return / MDD.
    
    realised_annual_return = (W_T - W_0) / W_0 / n_years
    where W_T is final wealth, W_0 is initial wealth.
    
    This is NOT E(R)/MDD — it uses the actual terminal wealth.
    """
    cumret = np.cumsum(port_returns)
    wealth = n_contracts * w0 + cumret
    w_start = wealth[0]
    w_end = wealth[-1]
    realised_ann = (w_end - w_start) / w_start / n_years
    
    mdd = compute_max_drawdown(port_returns, n_contracts, w0, n_years)
    return float(realised_ann / mdd) if mdd > 0 else 0.0


def compute_pct_positive(port_returns):
    """8. % +ve: Fraction of positive return days."""
    return float(np.sum(port_returns > 0) / len(port_returns))


def compute_avg_pl_ratio(port_returns):
    """9. Ave P/L: mean(R | R > 0) / |mean(R | R < 0)|."""
    pos = port_returns[port_returns > 0]
    neg = port_returns[port_returns < 0]
    avg_pos = np.mean(pos) if len(pos) > 0 else 0.0
    avg_neg = abs(np.mean(neg)) if len(neg) > 0 else 1e-10
    return float(avg_pos / avg_neg)


# =============================================================================
# Composite function — compute all 9 metrics at once
# =============================================================================

def compute_all_metrics(port_returns, n_contracts=1, w0=1.0, n_years=9):
    """Compute all 9 metrics from portfolio returns.
    
    Args:
        port_returns:   1D array of equal-weight portfolio daily returns (additive)
        n_contracts:    number of contracts in portfolio
        w0:             initial wealth per contract
        n_years:        number of years in test period
    
    Returns:
        dict with all 9 metric values (rounded to 3 decimals)
    """
    er = compute_expected_return(port_returns)
    std_r = compute_annualized_std(port_returns)
    dd = compute_downside_deviation(port_returns)
    sharpe = er / std_r if std_r > 0 else 0.0
    sortino = er / dd if dd > 0 else 0.0
    mdd = compute_max_drawdown(port_returns, n_contracts, w0, n_years)
    calmar = compute_calmar(port_returns, n_contracts, w0, n_years)
    pct_pos = compute_pct_positive(port_returns)
    avg_pl = compute_avg_pl_ratio(port_returns)
    
    return {name: round(v, 3) for name, v in zip(
        METRIC_NAMES, [er, std_r, dd, sharpe, sortino, mdd, calmar, pct_pos, avg_pl]
    )}
