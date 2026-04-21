"""
vol_scaling.py — Per-contract and portfolio-level volatility scaling
Reference: Zhang et al. 2019 Eq 4, 13; [17] Harvey; [27] Lim; [37] Moskowitz

This module is the single home for:
- per-contract scaling used in Eq. 4 style returns
- portfolio-level bridges used to map Table 3 style returns into Table 2 candidates

Framework: additive profits r_t = p_t - p_{t-1}
σ_t = EWMA(60) std of daily price diffs (same units as r_t)
σ_tgt = 0.10 (in price-diff units, constant across all contracts)
σ_tgt/σ_t is dimensionless → normalises each contract to same daily vol

The specific value of σ_tgt doesn't affect Sharpe, Sortino, % +ve, Ave P/L.
It only scales E(R), std(R), DD, MDD proportionally.
"""
import numpy as np
import pandas as pd
from config import EWMA_SPAN, PORT_TGT_STD, TRADING_DAYS


def compute_ewma_vol(values, span=EWMA_SPAN):
    """
    σ_t = EWMA(span) std of daily values.
    Works with additive price diffs (units: price/day).
    Returns array of same length. NaN/zero filled with first valid value.
    """
    vol = pd.Series(values).ewm(span=span, adjust=False).std().values
    first_valid = None
    for v in vol:
        if not np.isnan(v) and v > 0:
            first_valid = v
            break
    if first_valid is None:
        first_valid = 1e-6
    vol = np.nan_to_num(vol, nan=first_valid, posinf=first_valid, neginf=first_valid)
    vol[vol == 0] = first_valid
    return vol


def scale_per_contract(price_diffs, sigma_tgt_daily, span=EWMA_SPAN, max_leverage=None):
    """
    Per-contract volatility scaling (Formula 4 inner scaling).

    For ADDITIVE price diffs:
      σ_t = EWMA(60) std of (p_t - p_{t-1})  [price units/day]
      scaling = σ_tgt_daily / σ_t             [dimensionless]

    After scaling, each contract's daily vol ≈ σ_tgt_daily.
    σ_tgt_daily is the same for all contracts → normalises to same scale.

    Args:
        price_diffs:     daily price differences r_t = p_t - p_{t-1}
        sigma_tgt_daily: target daily volatility (same units as r_t)
        span:            EWMA span
        max_leverage:    optional cap on |scaling| to prevent extreme leverage

    Returns:
        scaling factor array (same length as input)
    """
    vol = compute_ewma_vol(price_diffs, span)
    scaling = sigma_tgt_daily / vol
    if max_leverage is not None:
        scaling = np.clip(scaling, -max_leverage, max_leverage)
    return scaling


def _clean_series(values):
    arr = np.asarray(values, dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _lagged_multiplier_from_vol(vol, target_daily):
    positive = vol[vol > 0]
    first_valid = positive.iloc[0] if len(positive) else target_daily
    vol = vol.fillna(first_valid)
    vol = vol.replace(0, first_valid)
    return (target_daily / vol).to_numpy(dtype=float)


def get_portfolio_bridge_multipliers(name: str, values, target_std=PORT_TGT_STD):
    """Return the pointwise bridge multipliers k_t for a named bridge.

    The returned array satisfies:
        bridged_values[t] = k_t * values[t]
    """
    arr = _clean_series(values)
    if name == 'constant_posthoc':
        current_std = np.std(arr) * np.sqrt(TRADING_DAYS)
        k = (target_std / current_std) if current_std > 0 else 1.0
        return np.full(len(arr), k, dtype=float)

    target_daily = target_std / np.sqrt(TRADING_DAYS)
    ser = pd.Series(arr)
    if name == 'ewma60_lagged':
        vol = ser.ewm(span=60, adjust=False).std().shift(1)
        return _lagged_multiplier_from_vol(vol, target_daily)
    if name == 'rolling252_lagged':
        vol = ser.rolling(window=252, min_periods=20).std().shift(1)
        return _lagged_multiplier_from_vol(vol, target_daily)
    raise ValueError(f'Unknown portfolio bridge: {name}')


def apply_portfolio_bridge(values, name: str, target_std=PORT_TGT_STD):
    """Apply a named Table 2 bridge and return the bridged series."""
    arr = _clean_series(values)
    k = get_portfolio_bridge_multipliers(name, arr, target_std)
    return arr * k


def constant_posthoc_scaler(target_std=PORT_TGT_STD):
    """Return a scaler that rescales the whole series by one constant."""
    def scale(values):
        return apply_portfolio_bridge(values, 'constant_posthoc', target_std)
    return scale


def ewma_portfolio_scaler(target_std=PORT_TGT_STD, span=EWMA_SPAN):
    """Lagged EWMA portfolio volatility targeter."""
    def scale(values):
        if span != 60:
            raise ValueError('Only ewma60_lagged is supported in the named bridge registry')
        return apply_portfolio_bridge(values, 'ewma60_lagged', target_std)
    return scale


def rolling_portfolio_scaler(target_std=PORT_TGT_STD, window=252, min_periods=20):
    """Lagged rolling-window portfolio volatility targeter."""
    def scale(values):
        if window != 252 or min_periods != 20:
            raise ValueError('Only rolling252_lagged is supported in the named bridge registry')
        return apply_portfolio_bridge(values, 'rolling252_lagged', target_std)
    return scale


def get_portfolio_bridge(name: str, target_std=PORT_TGT_STD):
    """Build a named Table 2 bridge scaler."""
    if name == 'constant_posthoc':
        return constant_posthoc_scaler(target_std)
    if name == 'ewma60_lagged':
        return ewma_portfolio_scaler(target_std, span=60)
    if name == 'rolling252_lagged':
        return rolling_portfolio_scaler(target_std, window=252)
    raise ValueError(f'Unknown portfolio bridge: {name}')


PORTFOLIO_BRIDGES = {
    'constant_posthoc': constant_posthoc_scaler(),
    'ewma60_lagged': ewma_portfolio_scaler(span=60),
    'rolling252_lagged': rolling_portfolio_scaler(window=252),
}
