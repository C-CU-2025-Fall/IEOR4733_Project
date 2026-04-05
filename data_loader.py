"""
data_loader.py — CLC data loading utilities
"""
import os
import numpy as np
import pandas as pd


def load_clc_full(ticker, data_dir='data/CLC', start_date='2009-01-01'):
    """
    Load CLC ratio-adjusted data from start_date onwards.
    Default 2009-01-01 gives ~504 trading days warmup before 2011 test,
    enough for MACD std_window=252 + longest EMA span=96.

    CSV format (no header): Date,Open,High,Low,Close,Volume,OpenInterest
    Date format: MM/DD/YYYY
    """
    fpath = os.path.join(data_dir, f'{ticker}_RAD.CSV')
    if not os.path.exists(fpath):
        return None
    df = pd.read_csv(fpath, header=None,
                     names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df = df[df['Close'].notna() & (df['Close'] > 0)].sort_values('Date').reset_index(drop=True)
    # Only keep data from start_date
    if start_date:
        df = df[df['Date'] >= start_date].reset_index(drop=True)
    if len(df) < 500:
        return None
    return df


def get_price_diffs(prices):
    """
    Additive profits: r_t = p_t - p_{t-1}
    Paper Section 3.2: "additive profits of holding a single contract"
    Used for: trade return computation (Formula 4)
    """
    diff = np.zeros(len(prices))
    diff[1:] = prices[1:] - prices[:-1]
    return diff


def get_pct_returns(prices):
    """
    Percentage returns: r_t = p_t / p_{t-1} - 1
    Used for: signal computation (Sign(R) cumulative return)
    """
    ret = np.zeros(len(prices))
    ret[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
    return ret


def extract_test_period(df, test_start='2011-01-01', test_end='2019-12-31'):
    """
    Get index boundaries and dates for the test period.
    Returns (start_idx, end_idx, date_series) or (None, None, None).
    """
    mask_start = df['Date'] >= test_start
    mask_end = df['Date'] <= test_end
    if not mask_start.any() or not mask_end.any():
        return None, None, None
    t0 = mask_start.idxmax()
    t1 = len(df) - 1 - mask_end[::-1].values.argmax()
    return t0, t1, df['Date']
