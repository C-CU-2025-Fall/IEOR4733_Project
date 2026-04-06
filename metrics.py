"""
metrics.py — Single source of truth for all 9 portfolio metrics.

Paper reference: Zhang, Zohren, Roberts (2020) Section 4.4

Metrics:
  1. E(R)     — Annualised expected return
  2. std(R)   — Annualised standard deviation
  3. DD       — Annualised downside deviation (from zero)
  4. Sharpe   — E(R) / std(R)
  5. Sortino  — E(R) / DD
  6. MDD      — Rolling 252-day maximum drawdown on NAV
  7. Calmar   — E(R) / MDD
  8. % +ve    — Fraction of positive return days
  9. Ave P/L  — Mean positive return / mean |negative return|

All other modules MUST import from this file. No duplicate implementations.
"""
import numpy as np
import pandas as pd
from config import TRADING_DAYS


# =============================================================================
# Individual metric functions
# =============================================================================

def calc_expected_return(port_returns, annual_factor=TRADING_DAYS):
    """E(R): annualised mean return."""
    return np.mean(port_returns) * annual_factor


def calc_std(port_returns, annual_factor=TRADING_DAYS):
    """std(R): annualised standard deviation of returns."""
    return np.std(port_returns) * np.sqrt(annual_factor)


def calc_downside_deviation(port_returns, annual_factor=TRADING_DAYS):
    """DD: downside deviation from zero. sqrt(mean(min(0,R)²)) × √T."""
    downside = np.minimum(port_returns, 0)
    return np.sqrt(np.mean(downside ** 2)) * np.sqrt(annual_factor)


def calc_sharpe(port_returns, annual_factor=TRADING_DAYS):
    """Sharpe ratio: E(R) / std(R)."""
    er = calc_expected_return(port_returns, annual_factor)
    std_r = calc_std(port_returns, annual_factor)
    return er / std_r if std_r > 0 else 0.0


def calc_sortino(port_returns, annual_factor=TRADING_DAYS):
    """Sortino ratio: E(R) / DD."""
    er = calc_expected_return(port_returns, annual_factor)
    dd = calc_downside_deviation(port_returns, annual_factor)
    return er / dd if dd > 0 else 0.0


def calc_mdd_rolling(port_returns, window=TRADING_DAYS):
    """MDD: rolling window max drawdown on cumprod(1+R).
    
    For each 252-day window, compute max drawdown, then take the max across all windows.
    This is the "worst year" drawdown.
    """
    mdds = []
    for i in range(len(port_returns) - window + 1):
        w = port_returns[i:i + window]
        wealth = np.cumprod(1 + w)
        peak = np.maximum.accumulate(wealth)
        mdds.append(np.max((peak - wealth) / peak))
    return float(max(mdds)) if mdds else 0.0


def calc_mdd_from_nav(total_nav, window=TRADING_DAYS):
    """MDD: rolling window max drawdown directly on NAV series.
    
    For each 252-day window, compute max drawdown on NAV, take max across windows.
    """
    nav_vals = total_nav.values if isinstance(total_nav, pd.Series) else total_nav
    mdds = []
    for i in range(len(nav_vals) - window + 1):
        w = nav_vals[i:i + window]
        peak = np.maximum.accumulate(w)
        mdds.append(np.max((peak - w) / peak))
    return float(max(mdds)) if mdds else 0.0


def calc_calmar(port_returns, annual_factor=TRADING_DAYS, mdd=None):
    """Calmar ratio: E(R) / MDD."""
    er = calc_expected_return(port_returns, annual_factor)
    if mdd is None:
        mdd = calc_mdd_rolling(port_returns)
    return er / mdd if mdd > 0 else 0.0


def calc_pct_positive(port_returns):
    """% +ve: fraction of positive return days."""
    return float(np.sum(port_returns > 0) / len(port_returns))


def calc_avg_pl(port_returns):
    """Ave P/L: ratio of mean positive return to mean |negative return|."""
    pos = port_returns[port_returns > 0]
    neg = port_returns[port_returns < 0]
    avg_pos = np.mean(pos) if len(pos) > 0 else 0.0
    avg_neg = abs(np.mean(neg)) if len(neg) > 0 else 1e-10
    return float(avg_pos / avg_neg)


# =============================================================================
# Compose all 9 metrics — two entry points
# =============================================================================

METRIC_NAMES = ['E(R)', 'std(R)', 'DD', 'Sharpe', 'Sortino', 'MDD', 'Calmar', '% +ve', 'Ave P/L']


def compute_all_metrics(port_returns, total_nav=None, annual_factor=TRADING_DAYS):
    """Compute all 9 metrics from portfolio returns.
    
    Args:
        port_returns: 1D array of portfolio daily returns
        total_nav:    Optional NAV series. If provided, MDD is computed on NAV directly.
                      If None, MDD uses cumprod(1+R) with rolling 252d window.
        annual_factor: Trading days per year (default 252)
    
    Returns:
        dict with all 9 metric values (rounded to 3 decimals)
    """
    er = calc_expected_return(port_returns, annual_factor)
    std_r = calc_std(port_returns, annual_factor)
    dd = calc_downside_deviation(port_returns, annual_factor)
    sharpe = er / std_r if std_r > 0 else 0.0
    sortino = er / dd if dd > 0 else 0.0
    
    if total_nav is not None:
        mdd = calc_mdd_from_nav(total_nav, window=annual_factor)
    else:
        mdd = calc_mdd_rolling(port_returns, window=annual_factor)
    
    calmar = er / mdd if mdd > 0 else 0.0
    pct_pos = calc_pct_positive(port_returns)
    avg_pl = calc_avg_pl(port_returns)
    
    return {name: round(v, 3) for name, v in zip(
        METRIC_NAMES, [er, std_r, dd, sharpe, sortino, mdd, calmar, pct_pos, avg_pl]
    )}
