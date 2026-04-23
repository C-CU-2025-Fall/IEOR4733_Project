#!/usr/bin/env python3
"""
baseline_run.py — Table 2 & Table 3 baseline reproduction (single entry point)

Paper: Zhang, Zohren, Roberts (2019) "Deep Reinforcement Learning for Trading"

Usage:
    python baseline_run.py                  # Table 3 (all asset classes)
    python baseline_run.py --table 2        # Table 2 (with portfolio vol scaling)
    python baseline_run.py --table both     # Both tables
    python baseline_run.py --asset Forex    # Single asset class
    python baseline_run.py --sigma 0.064    # Custom σ_tgt
    python baseline_run.py --test-start 2015-01-01 --test-end 2019-12-31  # Custom period
"""
import argparse
from functools import lru_cache
import os
import numpy as np
import pandas as pd
from data_loader import load_clc_full
from strategies import strategy_sign_r, strategy_macd
from vol_scaling import get_portfolio_bridge, get_portfolio_bridge_multipliers
from metrics import (
    cagr_from_path,
    compute_metrics,
    max_drawdown_from_path,
)
from config import (
    ASSET_CLASSES, BP, TRADING_DAYS, SIGN_LOOKBACK,
    PAPER_TABLE2, PAPER_TABLE3, METRIC_NAMES, EXCLUDED_CONTRACTS,
    SOURCE_OVERRIDES,
)

# Core 5 metrics for summary table
CORE_METRICS = ['E(R)', 'std(R)', 'Sharpe', '% +ve', 'Ave P/L']
CORE_METRIC_IDX = [METRIC_NAMES.index(n) for n in CORE_METRICS]

# ─── Parameters ───────────────────────────────────────────────────
DEFAULT_SIGMA_TGT = 0.0580   # Unified default target vol for baseline + DRL stack
EWMA_SPAN = 60              # EWMA span for σ_t [Paper Section 3.2]
T = TRADING_DAYS            # 252
W0 = 1.0                    # Initial wealth per contract
# ─── Data Loading ─────────────────────────────────────────────────


@lru_cache(maxsize=None)
def _prepare_contract_cached(ticker, test_start, test_end, source):
    df = load_clc_full(ticker, source=source, anchor_date=test_start)
    if df is None:
        return None
    prices = df['Close'].values.astype(float)
    if len(prices) < 500:
        return None

    rt = np.zeros(len(prices))
    rt[1:] = prices[1:] - prices[:-1]
    sigma = pd.Series(rt).ewm(span=EWMA_SPAN, adjust=False).std().values

    mask_s = df['Date'] >= test_start
    mask_e = df['Date'] <= test_end
    if not mask_s.any() or not mask_e.any():
        return None
    t0 = mask_s.idxmax()
    t1 = len(df) - 1 - mask_e[::-1].values.argmax()
    start = max(t0, SIGN_LOOKBACK)
    # `t1` is the last in-range index, so include it explicitly.
    dates = df['Date'].iloc[start:t1 + 1].values
    return {
        'tk': ticker,
        'rt': rt,
        'sigma': sigma,
        'prices': prices,
        'start': start,
        't1': t1,
        'dates': dates,
        'source': source,
        'macd_pos': strategy_macd(prices),
    }


def load_contracts(ac_name, test_start='2011-01-01', test_end='2019-12-31',
                   excluded_contracts=None, source_overrides=None):
    """Load and prepare all contracts for an asset class."""
    tickers = ASSET_CLASSES.get(ac_name, [])
    if excluded_contracts is None:
        excluded_contracts = EXCLUDED_CONTRACTS
    if source_overrides is None:
        source_overrides = SOURCE_OVERRIDES
    raw = []
    for tk in tickers:
        if tk in excluded_contracts:
            continue
        source = source_overrides.get(tk, 'RAD')
        prepared = _prepare_contract_cached(tk, test_start, test_end, source)
        if prepared is not None:
            raw.append(prepared)
    return raw


def parse_exclusion_arg(exclude_arg):
    """Parse a comma-separated exclusion override from the CLI."""
    if not exclude_arg:
        return list(EXCLUDED_CONTRACTS)
    tokens = [tk.strip().upper() for tk in exclude_arg.split(',')]
    tokens = [tk for tk in tokens if tk]
    return sorted(set(tokens))


def build_reporting_portfolio_risk_price_sigma0(raw_data, sigma_tgt, strat='Long',
                                               bridge_multiplier_series=None):
    """Build the fixed reporting-world sleeve-capital wealth path.

    Each contract sleeve starts with capital:
        C_i,0 = p_i,0 * sigma_tgt / sigma_i,0

    Sleeve wealth then accumulates normalized Eq. 4 rewards:
        w_i,t = 1 + cumsum(R_i,t / C_i,0)

    The reporting portfolio is the equal-weight average of sleeve wealth paths.
    """
    sleeve_paths = []
    for rd in raw_data:
        detail = compute_contract_returns(rd, strat, sigma_tgt, detail=True)
        start, t1 = rd['start'], rd['t1']
        Rt = detail['Rt'][start:t1 + 1]
        prices = detail['prices'][start:t1 + 1]
        sigma = detail['sigma'][start:t1 + 1]
        if len(Rt) == 0 or len(prices) == 0 or len(sigma) == 0:
            continue
        if bridge_multiplier_series is not None:
            dates = pd.Index(rd['dates'][:len(Rt)])
            k = bridge_multiplier_series.reindex(dates).fillna(1.0).to_numpy(dtype=float)
            Rt = Rt * k
        p0 = float(prices[0])
        sigma0 = float(sigma[0])
        if not np.isfinite(p0) or not np.isfinite(sigma0) or sigma0 <= 0:
            continue
        capital0 = p0 * sigma_tgt / sigma0
        if not np.isfinite(capital0) or capital0 <= 0:
            continue
        sleeve_paths.append(1.0 + np.cumsum(Rt / capital0))

    if not sleeve_paths:
        return None

    min_len = min(len(path) for path in sleeve_paths)
    if min_len == 0:
        return None

    sleeves = np.column_stack([path[:min_len] for path in sleeve_paths])
    portfolio = sleeves.mean(axis=1)

    sleeve_simple = np.full_like(sleeves, np.nan, dtype=float)
    if min_len > 1:
        sleeve_simple[1:, :] = sleeves[1:, :] / sleeves[:-1, :] - 1.0

    portfolio_simple = np.full(min_len, np.nan, dtype=float)
    portfolio_log = np.full(min_len, np.nan, dtype=float)
    if min_len > 1:
        portfolio_simple[1:] = portfolio[1:] / portfolio[:-1] - 1.0
        portfolio_log[1:] = np.log(portfolio[1:] / portfolio[:-1])

    return {
        'portfolio_path': portfolio,
        'portfolio_simple_returns': portfolio_simple,
        'portfolio_log_returns': portfolio_log,
        'sleeve_paths': sleeves,
        'sleeve_simple_returns': sleeve_simple,
        'length': min_len,
    }


def compute_portfolio_return_series(raw_data, strat, sigma_tgt, aggregation_mode='variable_n'):
    """Eq 13 portfolio returns as a dated Series."""
    series = []
    for rd in raw_data:
        Rt = compute_contract_returns(rd, strat, sigma_tgt)
        start, t1, dates = rd['start'], rd['t1'], rd['dates']
        slc = Rt[start:t1 + 1]
        series.append(pd.Series(slc[:len(dates)], index=dates[:len(slc)]))
    df_all = pd.DataFrame(series)
    if aggregation_mode == 'dropna':
        return df_all.T.dropna().mean(axis=1)
    if aggregation_mode == 'variable_n':
        return df_all.T.mean(axis=1)
    raise ValueError(f'Unknown aggregation_mode: {aggregation_mode}')


def apply_portfolio_vol_scaling(R_eq, target_std):
    """Backward-compatible constant post-hoc Table 2 bridge."""
    return get_portfolio_bridge('constant_posthoc', target_std)(np.asarray(R_eq, dtype=float))


def apply_portfolio_bridge_to_reporting(reporting, port_bridge=None, port_vol_target=None):
    """Apply the same Table 2 bridge to the reporting-world path.

    The current reporting world is constructed from normalized additive sleeve PnL
    paths. To keep MDD/Calmar in the same world as the Table 2 bridge, we derive
    additive reporting increments from the portfolio path, apply the same bridge,
    and rebuild the reporting wealth path from the bridged increments.
    """
    if reporting is None or port_bridge is None or port_vol_target is None:
        return reporting

    path = np.asarray(reporting['portfolio_path'], dtype=float)
    simple = np.asarray(reporting.get('portfolio_simple_returns'), dtype=float)
    if len(path) == 0 or len(simple) == 0:
        return reporting

    # Apply the Table 2 bridge to reporting simple returns, then rebuild the
    # reporting path multiplicatively from the same starting wealth.
    bridged_simple = np.array(simple, copy=True)
    if len(simple) > 1:
        valid_simple = np.nan_to_num(simple[1:], nan=0.0, posinf=0.0, neginf=0.0)
        bridged_simple[1:] = get_portfolio_bridge(port_bridge, port_vol_target)(valid_simple)

    bridged_path = np.empty_like(path)
    bridged_path[0] = path[0]
    for t in range(1, len(bridged_path)):
        prev = bridged_path[t - 1]
        step = bridged_simple[t]
        next_val = prev * (1.0 + step)
        # Keep the reporting path strictly positive for drawdown / CAGR logic.
        bridged_path[t] = next_val if np.isfinite(next_val) and next_val > 1e-12 else 1e-12

    portfolio_log = np.full(len(bridged_path), np.nan, dtype=float)
    if len(bridged_path) > 1:
        with np.errstate(divide='ignore', invalid='ignore'):
            portfolio_log[1:] = np.log(bridged_path[1:] / bridged_path[:-1])

    out = dict(reporting)
    out['portfolio_path'] = bridged_path
    out['portfolio_simple_returns'] = bridged_simple
    out['portfolio_log_returns'] = portfolio_log
    out['portfolio_additive_returns'] = np.full(len(bridged_path), np.nan, dtype=float)
    return out


def apply_bridge_to_reporting_sleeves_simple(reporting, bridge_multiplier_series=None):
    """Apply bridge multipliers to sleeve simple returns, then rebuild paths.

    This preserves per-sleeve / position information while keeping the reporting
    lane in a multiplicative wealth world. It avoids directly scaling additive
    sleeve PnL into negative-NAV paths.
    """
    if reporting is None or bridge_multiplier_series is None:
        return reporting

    sleeves = np.asarray(reporting.get('sleeve_paths'), dtype=float)
    sleeve_simple = np.asarray(reporting.get('sleeve_simple_returns'), dtype=float)
    if sleeves.ndim != 2 or sleeve_simple.ndim != 2 or sleeves.shape != sleeve_simple.shape:
        return reporting

    min_len, n_sleeves = sleeves.shape
    if min_len == 0 or n_sleeves == 0:
        return reporting

    k = np.asarray(bridge_multiplier_series, dtype=float)
    if len(k) < min_len:
        padded = np.ones(min_len, dtype=float)
        padded[:len(k)] = k
        k = padded
    else:
        k = k[:min_len]

    bridged_simple = np.array(sleeve_simple, copy=True)
    if min_len > 1:
        valid = np.nan_to_num(bridged_simple[1:, :], nan=0.0, posinf=0.0, neginf=0.0)
        bridged_simple[1:, :] = valid * k[1:, None]

    bridged_sleeves = np.empty_like(sleeves)
    bridged_sleeves[0, :] = sleeves[0, :]
    for j in range(n_sleeves):
        for t in range(1, min_len):
            prev = bridged_sleeves[t - 1, j]
            step = bridged_simple[t, j]
            next_val = prev * (1.0 + step)
            bridged_sleeves[t, j] = next_val if np.isfinite(next_val) and next_val > 1e-12 else 1e-12

    portfolio = bridged_sleeves.mean(axis=1)
    portfolio_simple = np.full(min_len, np.nan, dtype=float)
    portfolio_log = np.full(min_len, np.nan, dtype=float)
    if min_len > 1:
        with np.errstate(divide='ignore', invalid='ignore'):
            portfolio_simple[1:] = portfolio[1:] / portfolio[:-1] - 1.0
            portfolio_log[1:] = np.log(portfolio[1:] / portfolio[:-1])

    out = dict(reporting)
    out['sleeve_paths'] = bridged_sleeves
    out['sleeve_simple_returns'] = bridged_simple
    out['portfolio_path'] = portfolio
    out['portfolio_simple_returns'] = portfolio_simple
    out['portfolio_log_returns'] = portfolio_log
    return out


def compute_reporting_mdd_calmar_risk_price_sigma0(raw_data, sigma_tgt, strat='Long',
                                                   port_bridge=None, port_vol_target=None,
                                                   report_bridge_mode='split_world',
                                                   round_output=True):
    """
    Reporting-world bridge:

    - keep Eq. 4 trade rewards R_i,t
    - define sleeve initial capital as p_i,0 * sigma_tgt / sigma_i,0
    - accumulate normalized sleeve wealth:
          w_i,t = 1 + cumsum(R_i,t / C_i,0)
    - equal-weight sleeve wealth paths into a portfolio path

    This keeps the trade lane untouched while giving MDD/Calmar a portfolio
    wealth object that uses both initial price and initial risk scale.
    """
    bridge_multiplier_series = None
    if report_bridge_mode == 'same_as_port_additive' and port_bridge is not None and port_vol_target is not None:
        port_series = compute_portfolio_return_series(raw_data, strat, sigma_tgt, aggregation_mode='variable_n')
        k = get_portfolio_bridge_multipliers(port_bridge, port_series.values, port_vol_target)
        bridge_multiplier_series = pd.Series(k, index=port_series.index)

    reporting = build_reporting_portfolio_risk_price_sigma0(
        raw_data,
        sigma_tgt=sigma_tgt,
        strat=strat,
        bridge_multiplier_series=bridge_multiplier_series,
    )
    if report_bridge_mode == 'same_as_port_simple':
        reporting = apply_portfolio_bridge_to_reporting(
            reporting,
            port_bridge=port_bridge,
            port_vol_target=port_vol_target,
        )
    if report_bridge_mode == 'same_as_port_contract' and port_bridge is not None and port_vol_target is not None:
        port_series = compute_portfolio_return_series(raw_data, strat, sigma_tgt, aggregation_mode='variable_n')
        k = get_portfolio_bridge_multipliers(port_bridge, port_series.values, port_vol_target)
        reporting = apply_bridge_to_reporting_sleeves_simple(
            reporting,
            bridge_multiplier_series=k,
        )
    if reporting is None:
        return None
    port = reporting['portfolio_path']
    mdd = max_drawdown_from_path(port)
    cagr = cagr_from_path(port)
    calmar = cagr / mdd if mdd > 0 else 0.0
    if round_output:
        return round(mdd, 3), round(calmar, 3), reporting['length']
    return float(mdd), float(calmar), reporting['length']


# ─── Eq 4: Trade Return ──────────────────────────────────────────
def compute_contract_returns(rd, strat, sigma_tgt, detail=False):
    """Compute daily R_t for one contract using Paper Eq 4:

    R_t = A_{t-1} × (σ_tgt / σ_{t-1}) × r_t
        − bp × p_{t-1} × |(σ_tgt/σ_{t-1})×A_{t-1} − (σ_tgt/σ_{t-2})×A_{t-2}|

    Args:
        detail: if True, return dict with Rt + position/scale/TC details
    Returns:
        Rt array (default), or dict with full diagnostic info (detail=True)
    """
    rt, sigma, prices = rd['rt'], rd['sigma'], rd['prices']
    n = len(rt)

    # Position signal A_t
    if strat == 'Long':
        pos = np.ones(n)
    elif strat == 'Sign(R)':
        pos = strategy_sign_r(rt, SIGN_LOOKBACK)
    else:
        pos = rd['macd_pos']

    Rt = np.zeros(n)
    scaled_pos = np.zeros(n)   # A_{t-1} × σ_tgt/σ_{t-1}
    gross_pnl = np.zeros(n)    # sp × r_t  (before TC)
    tc_cost = np.zeros(n)      # bp × p_{t-1} × |Δscaled_pos|

    for t in range(1, n):
        if sigma[t - 1] > 0 and (t < 2 or sigma[t - 2] > 0):
            a_prev = pos[t - 1] if strat != 'Long' else 1.0
            a_prev2 = pos[t - 2] if strat != 'Long' else 1.0
            sp = a_prev * sigma_tgt / sigma[t - 1]
            spp = a_prev2 * sigma_tgt / sigma[t - 2] if t >= 2 else 0.0
            scaled_pos[t] = sp
            gross_pnl[t] = sp * rt[t]
            tc_cost[t] = BP * prices[t - 1] * abs(sp - spp)
            Rt[t] = gross_pnl[t] - tc_cost[t]

    if detail:
        return {'Rt': Rt, 'A_t': pos, 'scaled_pos': scaled_pos,
                'gross_pnl': gross_pnl, 'tc_cost': tc_cost,
                'sigma': sigma, 'prices': prices, 'rt': rt}
    return Rt


def compute_contract_returns_from_positions_loop(rd, positions, sigma_tgt, detail=False):
    """Compute Eq.4 returns from an explicit position array aligned to the full contract history."""
    rt, sigma, prices = rd['rt'], rd['sigma'], rd['prices']
    n = len(rt)
    pos = np.asarray(positions, dtype=float)
    if len(pos) != n:
        raise ValueError(f"Explicit positions length mismatch for {rd['tk']}: {len(pos)} vs {n}")

    Rt = np.zeros(n)
    scaled_pos = np.zeros(n)
    gross_pnl = np.zeros(n)
    tc_cost = np.zeros(n)

    for t in range(1, n):
        if sigma[t - 1] > 0 and (t < 2 or sigma[t - 2] > 0):
            a_prev = pos[t - 1]
            a_prev2 = pos[t - 2] if t >= 2 else 0.0
            sp = a_prev * sigma_tgt / sigma[t - 1]
            spp = a_prev2 * sigma_tgt / sigma[t - 2] if t >= 2 else 0.0
            scaled_pos[t] = sp
            gross_pnl[t] = sp * rt[t]
            tc_cost[t] = BP * prices[t - 1] * abs(sp - spp)
            Rt[t] = gross_pnl[t] - tc_cost[t]

    if detail:
        return {
            'Rt': Rt,
            'A_t': pos,
            'scaled_pos': scaled_pos,
            'gross_pnl': gross_pnl,
            'tc_cost': tc_cost,
            'sigma': sigma,
            'prices': prices,
            'rt': rt,
        }
    return Rt


def compute_contract_returns_from_positions(rd, positions, sigma_tgt, detail=False):
    """Vectorized Eq.4 returns from an explicit position array.

    This is the production path for model adapters. ``compute_contract_returns_from_positions_loop``
    remains as the reference implementation and is covered by parity tests.
    """
    rt, sigma, prices = rd['rt'], rd['sigma'], rd['prices']
    n = len(rt)
    pos = np.asarray(positions, dtype=float)
    if len(pos) != n:
        raise ValueError(f"Explicit positions length mismatch for {rd['tk']}: {len(pos)} vs {n}")

    Rt = np.zeros(n)
    scaled_pos = np.zeros(n)
    gross_pnl = np.zeros(n)
    tc_cost = np.zeros(n)
    if n <= 1:
        if detail:
            return {
                'Rt': Rt,
                'A_t': pos,
                'scaled_pos': scaled_pos,
                'gross_pnl': gross_pnl,
                'tc_cost': tc_cost,
                'sigma': sigma,
                'prices': prices,
                'rt': rt,
            }
        return Rt

    idx = np.arange(1, n)
    sig_prev = sigma[idx - 1]
    valid = sig_prev > 0
    if len(idx) > 1:
        valid[1:] &= sigma[idx[1:] - 2] > 0

    valid_idx = idx[valid]
    if len(valid_idx):
        sp = pos[valid_idx - 1] * sigma_tgt / sigma[valid_idx - 1]
        spp = np.zeros(len(valid_idx))
        has_prev = valid_idx >= 2
        spp[has_prev] = pos[valid_idx[has_prev] - 2] * sigma_tgt / sigma[valid_idx[has_prev] - 2]
        scaled_pos[valid_idx] = sp
        gross_pnl[valid_idx] = sp * rt[valid_idx]
        tc_cost[valid_idx] = BP * prices[valid_idx - 1] * np.abs(sp - spp)
        Rt[valid_idx] = gross_pnl[valid_idx] - tc_cost[valid_idx]

    if detail:
        return {
            'Rt': Rt,
            'A_t': pos,
            'scaled_pos': scaled_pos,
            'gross_pnl': gross_pnl,
            'tc_cost': tc_cost,
            'sigma': sigma,
            'prices': prices,
            'rt': rt,
        }
    return Rt


# ─── Eq 13: Portfolio Return ─────────────────────────────────────
def compute_portfolio_returns(raw_data, strat, sigma_tgt,
                              aggregation_mode='variable_n'):
    """Eq 13: R_port = (1/N) × Σ R_i  (equal-weight average)."""
    series = []
    for rd in raw_data:
        Rt = compute_contract_returns(rd, strat, sigma_tgt)
        start, t1, dates = rd['start'], rd['t1'], rd['dates']
        slc = Rt[start:t1 + 1]
        series.append(pd.Series(slc[:len(dates)], index=dates[:len(slc)]))
    df_all = pd.DataFrame(series)
    if aggregation_mode == 'dropna':
        port = df_all.T.dropna().mean(axis=1)
    elif aggregation_mode == 'variable_n':
        # Average only over contracts with data on each date. This preserves
        # dates across exchanges with different holiday calendars.
        port = df_all.T.mean(axis=1)
    else:
        raise ValueError(f'Unknown aggregation_mode: {aggregation_mode}')
    return port.values


def compute_portfolio_returns_from_position_provider(
    raw_data,
    sigma_tgt,
    position_provider,
    aggregation_mode='variable_n',
):
    """Eq 13 portfolio returns from explicit per-contract position arrays.

    position_provider must return a full-history position array for each contract payload.
    """
    series = []
    for rd in raw_data:
        pos = np.asarray(position_provider(rd), dtype=float)
        Rt = compute_contract_returns_from_positions(rd, pos, sigma_tgt)
        start, t1, dates = rd['start'], rd['t1'], rd['dates']
        slc = Rt[start:t1 + 1]
        series.append(pd.Series(slc[:len(dates)], index=dates[:len(slc)]))
    df_all = pd.DataFrame(series)
    if aggregation_mode == 'dropna':
        port = df_all.T.dropna().mean(axis=1)
    elif aggregation_mode == 'variable_n':
        port = df_all.T.mean(axis=1)
    else:
        raise ValueError(f'Unknown aggregation_mode: {aggregation_mode}')
    return port.values


def compute_strategy_metrics(
    raw_data,
    strat,
    sigma_tgt,
    aggregation_mode='variable_n',
    port_vol_target=None,
    port_bridge='constant_posthoc',
    position_provider=None,
):
    """Compute all 9 metrics for one strategy using the unified baseline stack."""
    built_in = {'Long', 'Sign(R)', 'MACD'}
    if position_provider is None:
        if strat not in built_in:
            raise ValueError(f"Unknown baseline strategy without position provider: {strat}")
        R = compute_portfolio_returns(raw_data, strat, sigma_tgt, aggregation_mode=aggregation_mode)
    else:
        R = compute_portfolio_returns_from_position_provider(
            raw_data,
            sigma_tgt=sigma_tgt,
            position_provider=position_provider,
            aggregation_mode=aggregation_mode,
        )

    if port_vol_target is not None:
        R = get_portfolio_bridge(port_bridge, port_vol_target)(R)

    N = len(raw_data)
    all_names = ['E(R)', 'std(R)', 'DD', 'Sharpe', 'Sortino',
                 'MDD', 'Calmar', '% +ve', 'Ave P/L']
    m_all_raw = compute_metrics(R, n_contracts=N, round_output=False)

    return dict(zip(all_names, m_all_raw))


def evaluate_asset_strategy_table3(
    ac_name,
    strat,
    sigma_tgt,
    test_start='2011-01-01',
    test_end='2019-12-31',
    excluded_contracts=None,
    source_overrides=None,
    aggregation_mode='variable_n',
    position_provider=None,
):
    """Evaluate one strategy on Table 3 with the unified baseline backtest stack."""
    if excluded_contracts is None:
        excluded_contracts = EXCLUDED_CONTRACTS
    if source_overrides is None:
        source_overrides = SOURCE_OVERRIDES
    raw = load_contracts(
        ac_name,
        test_start,
        test_end,
        excluded_contracts=excluded_contracts,
        source_overrides=source_overrides,
    )
    metrics_raw = compute_strategy_metrics(
        raw_data=raw,
        strat=strat,
        sigma_tgt=sigma_tgt,
        aggregation_mode=aggregation_mode,
        position_provider=position_provider,
    )
    metrics_round = {k: round(v, 3) for k, v in metrics_raw.items()}
    paper = PAPER_TABLE3.get(ac_name, {}).get(strat, None)
    errs = {}
    if paper is not None:
        for k in METRIC_NAMES:
            target = paper.get(k, 0.0)
            errs[k] = pct_err_raw(metrics_raw[k], target) if target != 0 else 0.0
    return metrics_round, metrics_raw, paper, errs

# ─── Output ───────────────────────────────────────────────────────
def fmt(vals):
    return "  ".join(f"{v:>+7.3f}" for v in vals)


def pct_err_raw(ours, paper):
    if paper == 0:
        return 0.0
    return abs((ours - paper) / abs(paper)) * 100.0


def run_table(raw_data, ac_name, sigma_tgt, paper_table, table_label,
              port_vol_target=None, metric_names=None,
              aggregation_mode='variable_n',
              port_bridge='constant_posthoc',
              strategy_entries=None,
              test_start='2011-01-01',
              test_end='2019-12-31'):
    """Run one table (Table 2 or 3) for one asset class."""
    N = len(raw_data)
    if N == 0:
        return 0, 0, 0

    if metric_names is None:
        metric_names = METRIC_NAMES

    # Get indices for the metrics we want to display
    all_names = ['E(R)', 'std(R)', 'DD', 'Sharpe', 'Sortino',
                 'MDD', 'Calmar', '% +ve', 'Ave P/L']
    metric_idx = [all_names.index(n) for n in metric_names]
    n_metrics = len(metric_names)

    port_str = f" | port_vol→{port_vol_target} | bridge={port_bridge}" if port_vol_target else ""
    print(f"\n{'=' * 90}")
    print(f"  {table_label} — {ac_name} ({N} contracts)")
    print(f"  σ_tgt={sigma_tgt} | EWMA({EWMA_SPAN}) | bp={BP}{port_str}")
    print(f"  Metrics: {', '.join(metric_names)}")
    print(f"{'=' * 90}")

    if strategy_entries is None:
        strategy_entries = [('Long', None)]

    total_n10, total_n15, total = 0, 0, 0
    for strat, position_provider in strategy_entries:
        metric_map = compute_strategy_metrics(
            raw_data=raw_data,
            strat=strat,
            sigma_tgt=sigma_tgt,
            aggregation_mode=aggregation_mode,
            port_vol_target=port_vol_target,
            port_bridge=port_bridge,
            position_provider=position_provider,
        )
        m_raw = [metric_map[n] for n in metric_names]
        m = [round(v, 3) for v in m_raw]
        pv_dict = paper_table[ac_name][strat]
        pv = [pv_dict[k] for k in metric_names]
        errs = [pct_err_raw(m_raw[i], pv[i])
                for i in range(n_metrics)]
        n10 = sum(1 for e in errs if e < 10)
        n15 = sum(1 for e in errs if e < 15)
        total_n10 += n10
        total_n15 += n15
        total += n_metrics

        print(f"\n  {strat:8s} (≤10%:{n10}/{n_metrics}  ≤15%:{n15}/{n_metrics})")
        print(f"  Ours  : {fmt(m)}")
        print(f"  Paper : {fmt(pv)}")
        print(f"  %Err  : {'  '.join(f'{e:>6.1f}%' for e in errs)}")

    return total_n10, total_n15, total


# ─── Main ─────────────────────────────────────────────────────────
def run_diagnostics(raw_data, ac_name, sigma_tgt, test_start='2011-01-01', test_end='2019-12-31'):
    """Output full portfolio diagnostic with per-contract positions.
    Saves a CSV with daily scaled_pos, gross_pnl, tc_cost for each contract
    plus portfolio-level R_port and cumsum.
    """
    all_details = {}
    for rd in raw_data:
        tk = rd['tk']
        det = compute_contract_returns(rd, 'Long', sigma_tgt, detail=True)
        start, t1, dates = rd['start'], rd['t1'], rd['dates']
        slc = slice(start, t1 + 1)
        df = pd.DataFrame({
            f'{tk}_price': det['prices'][slc],
            f'{tk}_rt': det['rt'][slc],
            f'{tk}_sigma': det['sigma'][slc],
            f'{tk}_scaled_pos': det['scaled_pos'][slc],
            f'{tk}_gross_pnl': det['gross_pnl'][slc],
            f'{tk}_tc': det['tc_cost'][slc],
            f'{tk}_Rt': det['Rt'][slc],
        }, index=dates[:t1 - start + 1])
        all_details[tk] = df

    # Merge all contracts on date index
    merged = pd.concat(all_details.values(), axis=1)

    # Portfolio-level columns
    rt_cols = [c for c in merged.columns if c.endswith('_Rt')]
    pos_cols = [c for c in merged.columns if c.endswith('_scaled_pos')]
    pnl_cols = [c for c in merged.columns if c.endswith('_gross_pnl')]
    tc_cols = [c for c in merged.columns if c.endswith('_tc')]

    merged['port_Rt'] = merged[rt_cols].mean(axis=1)
    merged['port_gross_pnl'] = merged[pnl_cols].mean(axis=1)
    merged['port_tc'] = merged[tc_cols].mean(axis=1)
    merged['port_cumsum'] = merged['port_Rt'].cumsum()
    merged['N_contracts'] = merged[rt_cols].notna().sum(axis=1)
    # Per-contract mean stats
    merged['mean_scaled_pos'] = merged[pos_cols].mean(axis=1)
    merged['port_daily_std'] = merged['port_Rt'].rolling(60).std()

    outdir = f'diagnostics/{ac_name.replace(" ", "_")}'
    os.makedirs(outdir, exist_ok=True)
    fname = f'{outdir}/portfolio_diagnostic_{sigma_tgt}.csv'
    merged.to_csv(fname)
    print(f'  → Saved {fname} ({len(merged)} rows × {len(merged.columns)} cols)')

    # Summary
    Rt = merged['port_Rt'].dropna().values
    if len(Rt) > 100:
        er = np.mean(Rt) * T
        std = np.std(Rt) * np.sqrt(T)
        print(f'  Summary: E(R)={er:+.3f}  std={std:.3f}  Sharpe={er/std:.3f}')
        print(f'  Cumsum range: [{merged["port_cumsum"].min():.2f}, {merged["port_cumsum"].max():.2f}]')
        print(f'  Mean scaled_pos: {merged["mean_scaled_pos"].mean():.4f}')
        print(f'  Mean TC/contract: {merged["port_tc"].mean():.6f}')
        print(f'  TC as % of gross: {abs(merged["port_tc"].mean()) / abs(merged["port_gross_pnl"].mean()) * 100:.2f}%')
    return fname


def main():
    parser = argparse.ArgumentParser(description='Baseline reproduction')
    parser.add_argument('--table', choices=['2', '3', 'both'], default='3',
                        help='Which table to run (default: 3)')
    parser.add_argument('--asset', default=None,
                        help='Single asset class (e.g. "Equity Index")')
    parser.add_argument('--sigma', type=float, default=DEFAULT_SIGMA_TGT,
                        help=f'σ_tgt per contract (default: {DEFAULT_SIGMA_TGT})')
    parser.add_argument('--test-start', default='2011-01-01')
    parser.add_argument('--test-end', default='2019-12-31')
    parser.add_argument('--port-vol-target', type=float, default=0.97,
                        help='Portfolio vol target for Table 2 (default: 0.97)')
    parser.add_argument('--port-bridge',
                        choices=['constant_posthoc', 'ewma60_lagged', 'rolling252_lagged'],
                        default='constant_posthoc',
                        help='Portfolio-level Table 2 bridge (default: constant_posthoc)')
    parser.add_argument('--all-metrics', action='store_true',
                        help='Show all 9 metrics (default: 5 core metrics)')
    parser.add_argument('--diagnostic', action='store_true',
                        help='Save full portfolio diagnostic CSV with positions')
    parser.add_argument('--aggregation', choices=['variable_n', 'dropna'],
                        default='variable_n',
                        help='Portfolio aggregation mode (default: variable_n)')
    parser.add_argument('--exclude-contracts', default=None,
                        help='Comma-separated exclusion override, e.g. FB,ZA,ZO')
    args = parser.parse_args()

    excluded_contracts = parse_exclusion_arg(args.exclude_contracts)

    asset_classes = [args.asset] if args.asset else [
        'Commodity', 'Equity Index', 'Fixed Income', 'Forex', 'All'
    ]

    # Select metric set
    if args.all_metrics:
        metric_names = list(METRIC_NAMES)  # all 9
    else:
        metric_names = CORE_METRICS  # 5 core
    print(f"Using {'ALL 9' if args.all_metrics else 'CORE 5'} metrics: {metric_names}")

    tables = []
    if args.table in ('3', 'both'):
        tables.append(('Table 3', PAPER_TABLE3, None))
    if args.table in ('2', 'both'):
        tables.append(('Table 2', PAPER_TABLE2, args.port_vol_target))

    grand_n10, grand_n15, grand_total = 0, 0, 0

    for table_label, paper_table, port_vol in tables:
        for ac in asset_classes:
            if ac == 'All':
                # All = combine all asset classes (excluding EXCLUDED_CONTRACTS)
                raw = []
                for a in ['Commodity', 'Equity Index', 'Fixed Income', 'Forex']:
                    raw.extend(load_contracts(
                        a,
                        args.test_start,
                        args.test_end,
                        excluded_contracts=excluded_contracts,
                    ))
            else:
                raw = load_contracts(
                    ac,
                    args.test_start,
                    args.test_end,
                    excluded_contracts=excluded_contracts,
                )
            n10, n15, tot = run_table(raw, ac, args.sigma, paper_table,
                                      table_label, port_vol,
                                      metric_names=metric_names,
                                      aggregation_mode=args.aggregation,
                                      port_bridge=args.port_bridge,
                                      test_start=args.test_start,
                                      test_end=args.test_end)
            if args.diagnostic and port_vol is None:  # only Table 3
                run_diagnostics(raw, ac, args.sigma, args.test_start, args.test_end)
            grand_n10 += n10
            grand_n15 += n15
            grand_total += tot

    if grand_total > 0:
        print(f"\n{'=' * 60}")
        print(f"  GRAND TOTAL: ≤10%: {grand_n10}/{grand_total}"
              f" | ≤15%: {grand_n15}/{grand_total}")
        print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
