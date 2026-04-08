#!/usr/bin/env python3
"""
Table 3 Verification — Percentage Returns Framework

Paper: Zhang, Zohren, Roberts (2019) "Deep Reinforcement Learning for Trading"
Table 3: Baseline strategies WITHOUT portfolio-level vol scaling

Framework:
  - r_t = (p_t - p_{t-1}) / p_{t-1}  (percentage returns)
  - c_t = A_t × (σ_tgt / σ_t)         (vol-targeting position)
  - σ_tgt = 0.10 annualized → 0.10/√252 daily
  - R_t = c_{t-1} × r_t − bp × |c_{t-1} − c_{t-2}|
"""

import numpy as np
import pandas as pd
import os

# =============================================================================
# Parameters
# =============================================================================
BP = 0.0020               # Transaction cost (20 bps)
SIGMA_TGT_ANNUAL = 0.10   # Annual vol target (10%)
SIGMA_TGT_DAILY = SIGMA_TGT_ANNUAL / np.sqrt(252)
EWMA_SPAN = 60
TEST_START = '2011-01-01'
TEST_END = '2019-12-31'
WARMUP = 252              # Skip first 252 days

# Paper Table 3 targets (Appendix B)
PAPER_TABLE3 = {
    'Equity Index': {
        'Long':   {'E(R)':0.504,'std(R)':0.928,'DD':0.606,'Sharpe':0.543,'Sortino':0.831,'MDD':0.127,'Calmar':0.466,'% +ve':0.541,'Ave P/L':0.928},
        'Sign(R)': {'E(R)':0.168,'std(R)':0.799,'DD':0.526,'Sharpe':0.211,'Sortino':0.319,'MDD':0.299,'Calmar':0.075,'% +ve':0.528,'Ave P/L':0.928},
        'MACD':   {'E(R)':-0.068,'std(R)':0.586,'DD':0.385,'Sharpe':-0.117,'Sortino':-0.178,'MDD':0.351,'Calmar':-0.041,'% +ve':0.519,'Ave P/L':0.904},
    },
    'Forex': {
        'Long':   {'E(R)':-0.198,'std(R)':0.472,'DD':0.285,'Sharpe':-0.420,'Sortino':-0.696,'MDD':0.219,'Calmar':-0.101,'% +ve':0.491,'Ave P/L':0.966},
        'Sign(R)': {'E(R)':-0.113,'std(R)':0.551,'DD':0.341,'Sharpe':-0.207,'Sortino':-0.332,'MDD':0.170,'Calmar':-0.071,'% +ve':0.499,'Ave P/L':0.968},
        'MACD':   {'E(R)':0.016,'std(R)':0.424,'DD':0.259,'Sharpe':0.037,'Sortino':0.061,'MDD':0.156,'Calmar':0.016,'% +ve':0.493,'Ave P/L':1.034},
    },
}

# =============================================================================
# Data Loading
# =============================================================================
def load_clc(ticker):
    """Load CLC ratio-adjusted data"""
    f = f'data/CLC/{ticker}_RAD.CSV'
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f, header=None, names=['Date','Open','High','Low','Close','Volume','OI'])
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df = df[(df['Date'] >= TEST_START) & (df['Date'] <= TEST_END)]
    df = df[df['Close'].notna() & (df['Close'] > 0)].sort_values('Date').reset_index(drop=True)
    return df

# =============================================================================
# Strategies
# =============================================================================
def strategy_long_only(n):
    """Always long"""
    return np.ones(n)

def strategy_sign_r(returns):
    """Sign of past 252-day cumulative return"""
    n = len(returns)
    positions = np.zeros(n)
    for t in range(252, n):
        cum_ret = np.prod(1 + returns[t-252:t]) - 1
        positions[t] = np.sign(cum_ret)
    return positions

def strategy_macd(prices):
    """MACD signal with 3 time-scales"""
    n = len(prices)
    positions = np.zeros(n)
    
    for t in range(252, n):
        macd_signals = []
        for (S, L) in [(8,24), (16,48), (32,96)]:
            # EMA
            ema_short = prices[t-S:t].mean()
            ema_long = prices[t-L:t].mean()
            macd = (ema_short - ema_long) / prices[t-63:t].std()
            # phi function
            phi = macd * np.exp(-macd**2 / 4) / 0.89
            macd_signals.append(phi)
        positions[t] = np.mean(macd_signals)
    
    return np.clip(positions, -1, 1)

# =============================================================================
# Trade Returns (Percentage Framework)
# =============================================================================
def compute_trade_returns(prices, positions):
    """
    Compute trade returns using percentage framework.
    
    R_t = c_{t-1} × r_t − bp × |c_{t-1} − c_{t-2}|
    
    where:
      r_t = (p_t - p_{t-1}) / p_{t-1}  (percentage returns)
      c_t = A_t × (σ_tgt / σ_t)         (vol-targeting position)
    """
    n = len(prices)
    
    # 1. Percentage returns
    pct_returns = np.zeros(n)
    pct_returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
    
    # 2. Volatility scaling
    vol = pd.Series(pct_returns).ewm(span=EWMA_SPAN, adjust=False).std().values
    vol = np.nan_to_num(vol, nan=SIGMA_TGT_DAILY, posinf=SIGMA_TGT_DAILY, neginf=SIGMA_TGT_DAILY)
    vol = np.clip(vol, 1e-6, None)  # Avoid div by zero
    
    scaling = SIGMA_TGT_DAILY / vol
    # scaling = np.clip(scaling, 0, 5.0)  # No leverage cap (paper doesn't specify)
    
    # 3. Scaled positions
    c = positions * scaling
    
    # 4. Trade returns
    R = np.zeros(n)
    for t in range(2, n):
        R[t] = c[t-1] * pct_returns[t] - BP * abs(c[t-1] - c[t-2])
    
    return R[WARMUP:]  # Skip warmup

# =============================================================================
# Metrics
# =============================================================================
def calc_metrics(returns):
    """Calculate all 9 Table 3 metrics"""
    er = returns.mean() * 252
    std = returns.std() * np.sqrt(252)
    dd = returns[returns < 0].std() * np.sqrt(252) if len(returns[returns < 0]) > 1 else std / np.sqrt(2)
    sharpe = er / std if std > 0 else 0
    sortino = er / dd if dd > 0 else 0
    
    # MDD
    wealth = np.cumprod(1 + returns)
    mdd = (np.maximum.accumulate(wealth) - wealth) / np.maximum.accumulate(wealth)
    mdd = np.max(mdd)
    
    calmar = er / mdd if mdd > 0 else 0
    pct_pos = (returns > 0).mean()
    avg_pl = returns[returns > 0].mean() / abs(returns[returns < 0].mean()) if returns[returns < 0].mean() != 0 else 0
    
    return {
        'E(R)': round(er, 3),
        'std(R)': round(std, 3),
        'DD': round(dd, 3),
        'Sharpe': round(sharpe, 3),
        'Sortino': round(sortino, 3),
        'MDD': round(mdd, 3),
        'Calmar': round(calmar, 3),
        '% +ve': round(pct_pos, 3),
        'Ave P/L': round(avg_pl, 3),
    }

# =============================================================================
# Portfolio Construction
# =============================================================================
def build_portfolio(contract_returns_list):
    """Equal-weight portfolio, aligned by minimum length"""
    min_len = min(len(r) for r in contract_returns_list)
    return np.mean([r[:min_len] for r in contract_returns_list], axis=0)

# =============================================================================
# Main
# =============================================================================
def test_asset_class(asset_class, tickers, paper_targets):
    """Test one asset class"""
    print(f"\n{'='*80}")
    print(f"  {asset_class}")
    print(f"{'='*80}")
    
    strat_returns = {'Long': [], 'Sign(R)': [], 'MACD': []}
    loaded = []
    
    for tk in tickers:
        df = load_clc(tk)
        if df is None:
            continue
        
        prices = df['Close'].values
        n = len(prices)
        
        # Strategies
        pos_long = strategy_long_only(n)
        pos_sign = strategy_sign_r((prices[1:] - prices[:-1]) / prices[:-1])
        pos_sign = np.insert(pos_sign, 0, 0)  # Align length
        pos_macd = strategy_macd(prices)
        
        # Trade returns
        for pos, key in [(pos_long, 'Long'), (pos_sign, 'Sign(R)'), (pos_macd, 'MACD')]:
            R = compute_trade_returns(prices, pos)
            strat_returns[key].append(R)
        
        loaded.append(tk)
    
    print(f"  Loaded: {len(loaded)}/{len(tickers)} contracts")
    
    # Portfolio & Metrics
    for strat in ['Long', 'Sign(R)', 'MACD']:
        if not strat_returns[strat]:
            continue
        
        port_ret = build_portfolio(strat_returns[strat])
        metrics = calc_metrics(port_ret)
        paper = paper_targets.get(strat, {})
        
        print(f"\n  {strat}:")
        print(f"    {'Metric':<10} {'Ours':>8} {'Paper':>8} {'Diff%':>8} Status")
        print(f"    {'-'*45}")
        
        for mn in ['E(R)', 'std(R)', 'DD', 'Sharpe', 'Sortino', 'MDD', 'Calmar', '% +ve', 'Ave P/L']:
            ours = metrics[mn]
            pv = paper.get(mn)
            if pv is not None:
                diff_pct = abs(ours - pv) / abs(pv) * 100 if pv != 0 else 0
                status = '✅' if diff_pct < 30 else '⚠️' if diff_pct < 60 else '❌'
                print(f"    {mn:<10} {ours:>8.3f} {pv:>8.3f} {diff_pct:>7.1f}% {status}")

# =============================================================================
# Run
# =============================================================================
if __name__ == '__main__':
    print("="*80)
    print("  Table 3 Verification — Percentage Returns Framework")
    print(f"  σ_tgt = {SIGMA_TGT_ANNUAL*100:.0f}% annualized, bp = {BP*10000:.0f} bps")
    print("="*80)
    
    # Equity Index (best data quality)
    EQ_TICKERS = ['CA', 'EN', 'ER', 'ES', 'MD', 'SC', 'SP', 'XU', 'XX', 'YM']
    test_asset_class('Equity Index', EQ_TICKERS, PAPER_TABLE3['Equity Index'])
    
    # Forex (also good quality)
    FX_TICKERS = ['AN', 'BN', 'CN', 'DX', 'FN', 'JN', 'MP', 'NK', 'SN']
    test_asset_class('Forex', FX_TICKERS, PAPER_TABLE3['Forex'])
