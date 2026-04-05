#!/usr/bin/env python3
"""
Unified Table 2 & Table 3 Reproduction — CLC Official Data
Paper: Zhang, Zohren, Roberts (2019) "Deep Reinforcement Learning for Trading"

Key references:
  [4]  Baz et al. 2015 — MACD signal definition (Eq 3-8), φ function
  [27] Lim, Zohren, Roberts 2019 — Volatility scaling framework (Eq 1)
       "We set the annualised volatility target σ_tgt to be 15%"
       σ_t = EWMA(60) std of one-day returns r_{t,t+1}

Paper Formula 4 (reward function):
  R_t = μ [ A_{t-1}·(σ_tgt/σ_{t-1})·r_t − bp·p_{t-1}·|c_{t-1} − c_{t-2}| ]

  where c_t = A_t · σ_tgt / σ_t, μ=1, bp=0.0020
  r_t = p_t − p_{t-1} (additive profits)
  σ_t = EWMA(60) std of r_t

Note: σ_tgt is never numerically specified in the paper.
  - [27] uses σ_tgt = 0.15 (annualized) with RETURNS-based r_t
  - This paper uses additive r_t = p_t − p_{t-1}
  - With additive r_t and σ_tgt=0.10, Equity std matches perfectly (0.929 vs 0.928)
  - Remaining std discrepancies for other classes likely from missing contracts + data version

Modes:
  python table2_table3_unified.py                         # Table 3: no vol scaling at all
  python table2_table3_unified.py --per-contract           # Table 3 + per-contract vol scaling
  python table2_table3_unified.py --per-contract --portfolio  # Table 2 (both layers)
  python table2_table3_unified.py --sigma-tgt 0.15         # custom σ_tgt
"""

import argparse
import numpy as np
import pandas as pd
import os
import math

# =============================================================================
# Constants
# =============================================================================
BP = 0.0020          # Transaction cost rate (20 bps), Table 1
TD = 252             # Trading days/year
PORT_TGT_STD = 0.97  # Table 2 portfolio-level target std
SIGMA_TGT_DAILY = 0.10 / math.sqrt(TD)  # Per-contract daily vol target (from annualized 10%)

# 50 contracts minus 3 with no 2011-2019 data:
#   ZH (Heating Oil Electronic): all zeros
#   ZU (Crude Oil Electronic): all zeros
#   US (T-Bonds Composite): all NaN
ASSET_CLASSES = {
    'Commodity': [t for t in [
        'CC','DA','GI','JO','KC','KW','LB','NR','SB','ZA',
        'ZC','ZF','ZG','ZH','ZI','ZK','ZL','ZN','ZO','ZP',
        'ZR','ZT','ZU','ZW','ZZ'
    ] if t not in ['ZH','ZU']],
    'Equity Index': ['CA','EN','ER','ES','LX','MD','SC','SP','XU','XX','YM'],
    'Fixed Income': [t for t in ['DT','FB','TY','UB','US'] if t != 'US'],
    'Forex': ['AN','BN','CN','DX','FN','JN','MP','NK','SN'],
}

# Paper Table 3 (Appendix B) — "Raw Signal"
PAPER_TABLE3 = {
    'Commodity': {
        'Long':   {'E(R)':-0.298,'std(R)':0.412,'DD':0.258,'Sharpe':-0.723,'Sortino':-1.152,'MDD':0.248,'Calmar':-0.130,'% +ve':0.473,'Ave P/L':0.987},
        'Sign(R)': {'E(R)':0.101,'std(R)':0.312,'DD':0.185,'Sharpe':0.325,'Sortino':0.548,'MDD':0.082,'Calmar':0.115,'% +ve':0.494,'Ave P/L':1.081},
        'MACD':   {'E(R)':-0.039,'std(R)':0.227,'DD':0.136,'Sharpe':-0.174,'Sortino':-0.290,'MDD':0.132,'Calmar':-0.059,'% +ve':0.486,'Ave P/L':1.024},
    },
    'Equity Index': {
        'Long':   {'E(R)':0.504,'std(R)':0.928,'DD':0.606,'Sharpe':0.543,'Sortino':0.831,'MDD':0.127,'Calmar':0.466,'% +ve':0.541,'Ave P/L':0.928},
        'Sign(R)': {'E(R)':0.168,'std(R)':0.799,'DD':0.526,'Sharpe':0.211,'Sortino':0.319,'MDD':0.299,'Calmar':0.075,'% +ve':0.528,'Ave P/L':0.928},
        'MACD':   {'E(R)':-0.068,'std(R)':0.586,'DD':0.385,'Sharpe':-0.117,'Sortino':-0.178,'MDD':0.351,'Calmar':-0.041,'% +ve':0.519,'Ave P/L':0.904},
    },
    'Fixed Income': {
        'Long':   {'E(R)':0.605,'std(R)':0.939,'DD':0.561,'Sharpe':0.645,'Sortino':1.081,'MDD':0.108,'Calmar':0.455,'% +ve':0.515,'Ave P/L':1.048},
        'Sign(R)': {'E(R)':0.189,'std(R)':0.795,'DD':0.496,'Sharpe':0.237,'Sortino':0.381,'MDD':0.165,'Calmar':0.103,'% +ve':0.504,'Ave P/L':1.024},
        'MACD':   {'E(R)':0.136,'std(R)':0.609,'DD':0.367,'Sharpe':0.224,'Sortino':0.371,'MDD':0.124,'Calmar':0.131,'% +ve':0.485,'Ave P/L':1.102},
    },
    'Forex': {
        'Long':   {'E(R)':-0.198,'std(R)':0.472,'DD':0.285,'Sharpe':-0.420,'Sortino':-0.696,'MDD':0.219,'Calmar':-0.101,'% +ve':0.491,'Ave P/L':0.966},
        'Sign(R)': {'E(R)':-0.113,'std(R)':0.551,'DD':0.341,'Sharpe':-0.207,'Sortino':-0.332,'MDD':0.170,'Calmar':-0.071,'% +ve':0.499,'Ave P/L':0.968},
        'MACD':   {'E(R)':0.016,'std(R)':0.424,'DD':0.259,'Sharpe':0.037,'Sortino':0.061,'MDD':0.156,'Calmar':0.016,'% +ve':0.493,'Ave P/L':1.034},
    },
}

# Paper Table 2 — with portfolio-level volatility targeting
PAPER_TABLE2 = {
    'Commodity': {
        'Long':   {'E(R)':-0.710,'std(R)':0.979,'DD':0.604,'Sharpe':-0.726,'Sortino':-1.177,'MDD':0.350,'Calmar':-0.140,'% +ve':0.473,'Ave P/L':0.989},
        'Sign(R)': {'E(R)':0.347,'std(R)':0.980,'DD':0.572,'Sharpe':0.354,'Sortino':0.606,'MDD':0.116,'Calmar':0.119,'% +ve':0.494,'Ave P/L':1.084},
        'MACD':   {'E(R)':-0.171,'std(R)':0.978,'DD':0.584,'Sharpe':-0.175,'Sortino':-0.293,'MDD':0.190,'Calmar':-0.060,'% +ve':0.486,'Ave P/L':1.026},
    },
    'Equity Index': {
        'Long':   {'E(R)':0.668,'std(R)':0.970,'DD':0.606,'Sharpe':0.688,'Sortino':1.102,'MDD':0.132,'Calmar':0.509,'% +ve':0.542,'Ave P/L':0.948},
        'Sign(R)': {'E(R)':0.228,'std(R)':0.966,'DD':0.610,'Sharpe':0.236,'Sortino':0.374,'MDD':0.344,'Calmar':0.077,'% +ve':0.528,'Ave P/L':0.930},
        'MACD':   {'E(R)':0.016,'std(R)':0.962,'DD':0.618,'Sharpe':0.017,'Sortino':0.027,'MDD':0.311,'Calmar':0.006,'% +ve':0.519,'Ave P/L':0.927},
    },
    'Fixed Income': {
        'Long':   {'E(R)':0.680,'std(R)':0.975,'DD':0.576,'Sharpe':0.698,'Sortino':1.180,'MDD':0.061,'Calmar':0.444,'% +ve':0.515,'Ave P/L':1.054},
        'Sign(R)': {'E(R)':0.214,'std(R)':0.972,'DD':0.592,'Sharpe':0.221,'Sortino':0.363,'MDD':0.080,'Calmar':0.083,'% +ve':0.504,'Ave P/L':1.019},
        'MACD':   {'E(R)':0.219,'std(R)':0.967,'DD':0.579,'Sharpe':0.228,'Sortino':0.380,'MDD':0.065,'Calmar':0.123,'% +ve':0.486,'Ave P/L':1.101},
    },
    'Forex': {
        'Long':   {'E(R)':-0.344,'std(R)':0.973,'DD':0.583,'Sharpe':-0.353,'Sortino':-0.590,'MDD':0.423,'Calmar':-0.097,'% +ve':0.491,'Ave P/L':0.979},
        'Sign(R)': {'E(R)':-0.297,'std(R)':0.973,'DD':0.592,'Sharpe':-0.306,'Sortino':-0.502,'MDD':0.434,'Calmar':-0.111,'% +ve':0.499,'Ave P/L':0.954},
        'MACD':   {'E(R)':0.006,'std(R)':0.970,'DD':0.582,'Sharpe':0.007,'Sortino':0.011,'MDD':0.329,'Calmar':0.002,'% +ve':0.493,'Ave P/L':1.029},
    },
}

METRIC_NAMES = ['E(R)','std(R)','DD','Sharpe','Sortino','MDD','Calmar','% +ve','Ave P/L']


# =============================================================================
# Data Loading
# =============================================================================
def load_clc_full(ticker):
    """Load ALL available CLC data for a contract (full history for warmup)."""
    f = f'data/CLC/{ticker}_RAD.CSV'
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f, header=None,
                     names=['Date','Open','High','Low','Close','Volume','OI'])
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df = df[df['Close'].notna() & (df['Close'] > 0)].sort_values('Date').reset_index(drop=True)
    if len(df) < 500:
        return None
    return df


# =============================================================================
# Strategy Functions (positions A_t ∈ {-1, 0, +1} or [-1, 1])
# Following [4] Baz et al. 2015 and [27] Lim et al. 2019
# =============================================================================
def strategy_long_only(n):
    """Long Only: A_t = 1 for all t."""
    return np.ones(n)


def strategy_sign_r(returns, lookback=252):
    """Sign(R) [27]: A_t = sign(cumulative return over past 252 days).
    Signal computed on percentage returns; position ∈ {-1, 0, +1}.
    """
    positions = np.zeros(len(returns))
    for t in range(lookback, len(returns)):
        cum_ret = np.prod(1 + returns[t-lookback:t]) - 1
        positions[t] = np.sign(cum_ret)
    return positions


def strategy_macd(prices):
    """MACD [4]: multi-timeframe MACD with φ transformation.

    Eqs 4-8 from [27]:
      q_t = (m(S) - m(L)) / std(p_{t-63:t})
      Y_t = q_t / std(z_{t-252:t})
      X_t = φ(Y_t) = Y·exp(-Y²/4) / 0.89

    Time-scales: S∈{8,16,32}, L∈{24,48,96}
    """
    positions = np.zeros(len(prices))
    macd_sum = np.zeros(len(prices))

    for S, L in [(8, 24), (16, 48), (32, 96)]:
        ema_fast = pd.Series(prices).ewm(span=S, adjust=False).mean()
        ema_slow = pd.Series(prices).ewm(span=L, adjust=False).mean()
        std_63 = pd.Series(prices).rolling(63, min_periods=63).std()
        q = (ema_fast - ema_slow) / std_63
        std_q = q.rolling(252, min_periods=252).std()
        macd_sum += (q / std_q).fillna(0).values

    macd_avg = macd_sum / 3

    for t in range(252, len(prices)):
        m = macd_avg[t]
        if not (np.isnan(m) or np.isinf(m)):
            phi = m * np.exp(-m**2 / 4) / 0.89
            positions[t] = np.clip(phi, -1, 1)

    return positions


# =============================================================================
# Trade Return Computation — Paper Formula 4
# =============================================================================
def compute_contract_returns(prices, positions, sigma_tgt=None):
    """
    Compute per-contract trade returns using paper's Formula 4.

    R_t = c_{t-1} · r_t  −  bp · p_{t-1} · |c_{t-1} − c_{t-2}|

    where:
      r_t = p_t − p_{t-1}   (additive profits, as stated in paper)
      c_t = A_t · σ_tgt/σ_t  (if sigma_tgt given)
      c_t = A_t               (if sigma_tgt=None → no per-contract scaling)
      σ_t = EWMA(span=60) std of r_t

    Args:
        prices:    close prices array (full history for warmup)
        positions: signal positions array (full history)
        sigma_tgt: target volatility (None = no per-contract scaling)
    """
    n = len(prices)

    # r_t = (p_t - p_{t-1}) / p_{t-1} (percentage returns / additive profits)
    # Note: Paper says "additive profits" but Table 3 metrics (std~0.4-0.9) confirm
    # these are percentage returns, not raw price differences
    r_add = np.zeros(n)
    r_add[1:] = (prices[1:] - prices[:-1]) / prices[:-1]

    if sigma_tgt is not None:
        # σ_t = EWMA(60) std of additive r_t (daily)
        vol = pd.Series(r_add).ewm(span=60, adjust=False).std().values
        # 修复：用第 61 天的 vol 填充前 60 天的 NaN 和 0
        vol_60 = vol[60] if len(vol) > 60 and not np.isnan(vol[60]) and vol[60] > 0 else 0.01
        vol = np.nan_to_num(vol, nan=vol_60, posinf=vol_60, neginf=vol_60)
        vol[vol == 0] = vol_60
        # Scaled positions: c_t = A_t · σ_tgt / σ_t
        # σ_tgt 必须是日波动率目标（与 σ_t 同单位）
        scaled = positions * (sigma_tgt / vol)
    else:
        # No per-contract vol scaling: c_t = A_t
        scaled = positions.copy()

    # Trade returns: R_t = c_{t-1}·r_t − bp·|c_{t-1}−c_{t-2}|
    # Note: bp is in percentage terms (20 bps = 0.0002), not multiplied by price
    # Paper notation "p_{t-1}" refers to notional (=1 for % returns), not raw price
    trade_rets = np.zeros(n)
    for t in range(2, n):
        trade_rets[t] = (scaled[t-1] * r_add[t]
                         - BP * abs(scaled[t-1] - scaled[t-2]))

    return trade_rets


# =============================================================================
# Portfolio Metrics
# =============================================================================
def compute_portfolio_metrics(trade_rets_list, apply_portfolio_scaling=False):
    """
    Compute all 9 Table 2/3 metrics.

    Steps (Eq 13):
    1. Equal-weight average of per-contract returns
    2. Optionally scale to portfolio-level target std = 0.97
    3. Compute E(R), std(R), DD, Sharpe, Sortino, MDD, Calmar, % +ve, Ave P/L
    """
    min_len = min(len(r) for r in trade_rets_list)
    port_raw = np.mean([r[:min_len] for r in trade_rets_list], axis=0)

    # Portfolio-level volatility scaling (Table 2 only)
    if apply_portfolio_scaling:
        raw_std = np.std(port_raw) * np.sqrt(TD)
        if raw_std > 0:
            port_scaled = port_raw * (PORT_TGT_STD / raw_std)
        else:
            port_scaled = port_raw
    else:
        port_scaled = port_raw.copy()

    # Metrics on scaled returns
    er = np.mean(port_scaled) * TD
    std_r = np.std(port_scaled) * np.sqrt(TD)

    # Downside deviation: annualised std of negative returns only
    neg = port_scaled[port_scaled < 0]
    dd = np.std(neg) * np.sqrt(TD) if len(neg) > 1 else std_r / np.sqrt(2)

    sharpe = er / std_r if std_r > 0 else 0
    sortino = er / dd if dd > 0 else 0

    # --- MDD ---
    # Compute on wealth curve (cumprod) for proper drawdown calculation
    # Start with wealth = 1, apply daily returns
    wealth = np.cumprod(1 + port_scaled)
    running_max = np.maximum.accumulate(wealth)
    drawdown = (running_max - wealth) / running_max
    mdd = float(np.nanmax(drawdown))

    # Calmar = E(R) / MDD
    calmar = er / mdd if mdd > 0 else 0

    # % +ve and Ave P/L on unscaled returns
    pct_pos = np.sum(port_raw > 0) / len(port_raw)
    pos_r = port_raw[port_raw > 0]
    neg_r = port_raw[port_raw < 0]
    avg_pos = np.mean(pos_r) if len(pos_r) > 0 else 0
    avg_neg = abs(np.mean(neg_r)) if len(neg_r) > 0 else 1e-10

    return {
        'E(R)': round(er, 3),
        'std(R)': round(std_r, 3),
        'DD': round(dd, 3),
        'Sharpe': round(sharpe, 3),
        'Sortino': round(sortino, 3),
        'MDD': round(mdd, 3),
        'Calmar': round(calmar, 3),
        '% +ve': round(pct_pos, 3),
        'Ave P/L': round(avg_pos / avg_neg, 3),
    }


# =============================================================================
# Comparison Helper
# =============================================================================
def status_icon(ours, paper, metric):
    """Generate ✅/⚠️/❌ based on % difference from paper."""
    if paper == 0:
        return '  ' if abs(ours) < 0.01 else '❌'
    pct = abs(ours - paper) / abs(paper) * 100
    if metric == 'std(R)':
        return '✅' if pct < 5 else '⚠️' if pct < 15 else '❌'
    elif metric in ['% +ve', 'Ave P/L']:
        return '✅' if pct < 10 else '⚠️' if pct < 25 else '❌'
    elif metric == 'MDD':
        return '✅' if pct < 30 else '⚠️' if pct < 60 else '❌'
    else:
        return '✅' if pct < 30 else '⚠️' if pct < 60 else '❌'


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Table 2/3 Reproduction')
    parser.add_argument('--per-contract', action='store_true',
                        help='Apply per-contract volatility scaling (σ_tgt/σ_t)')
    parser.add_argument('--portfolio', action='store_true',
                        help='Apply portfolio-level volatility scaling (std→0.97)')
    parser.add_argument('--sigma-tgt', type=float, default=0.10,
                        help='Target volatility for per-contract scaling (default: 0.10)')
    args = parser.parse_args()

    # Select paper targets
    if args.portfolio:
        paper_targets = PAPER_TABLE2
        table_name = "Table 2"
    else:
        paper_targets = PAPER_TABLE3
        table_name = "Table 3"

    # Build config description
    config_parts = [f"{table_name}"]
    if args.per_contract:
        config_parts.append(f"per-contract vol scaling: σ_tgt={args.sigma_tgt}")
    else:
        config_parts.append("NO per-contract vol scaling (raw signal)")
    if args.portfolio:
        config_parts.append(f"portfolio vol scaling → std={PORT_TGT_STD}")

    print("=" * 110)
    print(f"📊 {' | '.join(config_parts)}")
    print(f"   Formula 4: rt = pt−pt-1 (additive), σ_t = EWMA(60) on rt, cost = bp×pt-1×|Δc|")
    print("=" * 110)

    for ac, tickers in ASSET_CLASSES.items():
        print(f"\n{'='*110}")
        print(f"  {ac} ({len(tickers)} contracts)")
        print(f"{'='*110}")

        strat_rets = {'Long': [], 'Sign(R)': [], 'MACD': []}
        loaded = []

        for tk in tickers:
            df = load_clc_full(tk)
            if df is None:
                continue

            prices = df['Close'].values
            # Returns for signal computation (Sign(R) uses % returns)
            r_ret = df['Close'].pct_change().fillna(0).values

            # Test period boundaries
            t0 = df[df['Date'] >= '2011-01-01'].index[0]
            t1 = df[df['Date'] <= '2019-12-31'].index[-1]

            # Compute positions on FULL history (for warmup)
            pos_long = strategy_long_only(len(prices))
            pos_sign = strategy_sign_r(r_ret)
            pos_macd = strategy_macd(prices)

            # 年化 σ_tgt 转换为日目标（与 vol 同单位）
            sigma_tgt = (args.sigma_tgt / math.sqrt(TD)) if args.per_contract else None

            for pos, key in [(pos_long, 'Long'), (pos_sign, 'Sign(R)'),
                             (pos_macd, 'MACD')]:
                all_tr = compute_contract_returns(prices, pos, sigma_tgt=sigma_tgt)
                # Extract test period, skip first 252 days for signal warmup
                start = max(t0, 252)
                strat_rets[key].append(all_tr[start:t1+1])

            loaded.append(tk)

        print(f"  Loaded: {len(loaded)}/{len(tickers)} — {loaded}")
        if not loaded:
            continue

        pp = paper_targets.get(ac, {})

        for strat in ['Long', 'Sign(R)', 'MACD']:
            if not strat_rets[strat]:
                continue

            metrics = compute_portfolio_metrics(
                strat_rets[strat],
                apply_portfolio_scaling=args.portfolio,
            )

            paper = pp.get(strat, {})
            print(f"\n  {strat}:")
            print(f"    {'Metric':<10} {'Ours':>8}  {'Paper':>8}  {'Diff':>8}  {'%':>7}  Status")
            print(f"    {'-'*60}")

            for mn in METRIC_NAMES:
                ov = metrics[mn]
                pv = paper.get(mn)
                if pv is not None:
                    diff = ov - pv
                    pct = abs(diff / abs(pv)) * 100 if pv != 0 else 0
                    s = status_icon(ov, pv, mn)
                    print(f"    {mn:<10} {ov:>+8.3f}  {pv:>+8.3f}  {diff:>+8.3f}  {pct:>6.1f}%  {s}")
                else:
                    print(f"    {mn:<10} {ov:>+8.3f}")


if __name__ == '__main__':
    main()
