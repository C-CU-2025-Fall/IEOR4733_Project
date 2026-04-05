"""
vol_scaling.py — Per-contract and portfolio-level volatility scaling
Reference: [27] Lim et al. 2019 Eq 1; Zhang et al. 2019 Eq 4, 13
"""
import numpy as np
import pandas as pd
from config import EWMA_SPAN, PORT_TGT_STD, TRADING_DAYS


def compute_ewma_vol(returns, span=EWMA_SPAN):
    """
    Compute σ_t = EWMA(span) std of daily returns.
    Returns array of same length. NaN filled with first valid value.
    """
    vol = pd.Series(returns).ewm(span=span, adjust=False).std().values
    # Fill NaN/zero with first valid volatility
    first_valid = None
    for v in vol:
        if not np.isnan(v) and v > 0:
            first_valid = v
            break
    if first_valid is None:
        first_valid = 0.01  # fallback
    vol = np.nan_to_num(vol, nan=first_valid, posinf=first_valid, neginf=first_valid)
    vol[vol == 0] = first_valid
    return vol


def scale_per_contract(returns, sigma_tgt_annual, span=EWMA_SPAN):
    """
    Per-contract volatility scaling (Formula 4 inner scaling).

    [27]: "annualised volatility target σ_tgt = 15%"
    σ_t = EWMA(60) std of daily returns (NOT annualised)
    scaling_factor = σ_tgt_annual / (σ_t × √252)
    → = σ_tgt_annual / annualised_σ_t

    Args:
        returns:        daily return series
        sigma_tgt_annual: annualised volatility target (e.g. 0.15)
        span:          EWMA span

    Returns:
        scaling factor array c_t = σ_tgt / σ_t_annualised
    """
    vol_daily = compute_ewma_vol(returns, span)
    vol_annual = vol_daily * np.sqrt(TRADING_DAYS)
    return sigma_tgt_annual / vol_annual


def scale_portfolio(returns, target_std=PORT_TGT_STD):
    """
    Portfolio-level volatility scaling (Eq 13).

    Scale portfolio returns so that annualised std = target_std.

    Args:
        returns:    portfolio daily returns
        target_std: target annualised std (default 0.97 for Table 2)

    Returns:
        scaled returns
    """
    current_std = np.std(returns) * np.sqrt(TRADING_DAYS)
    if current_std > 0:
        return returns * (target_std / current_std)
    return returns
