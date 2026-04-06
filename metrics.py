"""
metrics.py — Portfolio performance metrics (9 indicators, each as a function)

Note: With additive r_t = p_t - p_{t-1}, the returns R_t are in price units.
After per-contract vol scaling, each contract's daily vol ≈ σ_tgt_daily.
We treat the scaled returns as return-like quantities for metric computation.
"""
import numpy as np
from config import TRADING_DAYS


# =============================================================================
# Individual metric functions — each independently testable
# =============================================================================

def calc_expected_return(port_returns, annual_factor=TRADING_DAYS):
    """E(R): annualised mean return."""
    return np.mean(port_returns) * annual_factor


def calc_std(port_returns, annual_factor=TRADING_DAYS):
    """std(R): annualised standard deviation of returns."""
    return np.std(port_returns) * np.sqrt(annual_factor)


def calc_downside_deviation(port_returns, annual_factor=TRADING_DAYS):
    """DD: downside deviation = sqrt(1/n * sum(min(0, R_i)^2)), annualized."""
    n = len(port_returns)
    return np.sqrt(np.mean(np.minimum(port_returns, 0) ** 2)) * np.sqrt(annual_factor)


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


def calc_mdd(port_returns, window=252):
    """
    Rolling Maximum Drawdown: average of max drawdown over rolling windows.

    MDD = max((Peak_t - Trough_t) / Peak_t) over window, averaged across windows.
    Uses cumprod(1 + R) as wealth curve within each window.
    """
    mdds = []
    for i in range(len(port_returns) - window + 1):
        w = port_returns[i:i + window]
        wealth = np.cumprod(1 + w)
        peak = np.maximum.accumulate(wealth)
        dd = (peak - wealth) / peak
        mdds.append(float(np.max(dd)))
    return float(np.mean(mdds)) if mdds else 0.0


def calc_calmar(port_returns, annual_factor=TRADING_DAYS):
    """Calmar ratio: E(R) / MDD."""
    er = calc_expected_return(port_returns, annual_factor)
    mdd = calc_mdd(port_returns)
    return er / mdd if mdd > 0 else 0.0


def calc_pct_positive(port_returns):
    """% +ve: fraction of positive return days."""
    return np.sum(port_returns > 0) / len(port_returns)


def calc_avg_pl(port_returns):
    """Ave P/L: ratio of average positive return to average |negative return|."""
    pos = port_returns[port_returns > 0]
    neg = port_returns[port_returns < 0]
    avg_pos = np.mean(pos) if len(pos) > 0 else 0.0
    avg_neg = abs(np.mean(neg)) if len(neg) > 0 else 1e-10
    return avg_pos / avg_neg


# =============================================================================
# Compose all 9 metrics
# =============================================================================

METRIC_FUNCTIONS = {
    'E(R)':      calc_expected_return,
    'std(R)':    calc_std,
    'DD':        calc_downside_deviation,
    'Sharpe':    calc_sharpe,
    'Sortino':   calc_sortino,
    'MDD':       calc_mdd,
    'Calmar':    calc_calmar,
    '% +ve':     calc_pct_positive,
    'Ave P/L':   calc_avg_pl,
}


def compute_all_metrics(port_returns, annual_factor=TRADING_DAYS):
    """Compute all 9 metrics from a 1D array of portfolio returns."""
    return {name: round(fn(port_returns, annual_factor)
                        if name not in ('MDD', '% +ve', 'Ave P/L')
                        else fn(port_returns), 3)
            for name, fn in METRIC_FUNCTIONS.items()}
