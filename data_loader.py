"""
data_loader.py — CLC data loading utilities
"""
import os
import pandas as pd
import numpy as np


def load_clc_full(ticker, data_dir='data/CLC'):
    """
    Load ALL available CLC ratio-adjusted data for a contract.
    Returns full history for indicator warmup.

    CSV format (no header): Date,Open,High,Low,Close,Volume,OpenInterest
    Date format: MM/DD/YYYY
    """
    fpath = os.path.join(data_dir, f'{ticker}_RAD.CSV')
    if not os.path.exists(fpath):
        return None

    df = pd.read_csv(fpath, header=None,
                     names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

    # Filter out invalid close prices (zero or NaN)
    df = df[df['Close'].notna() & (df['Close'] > 0)].sort_values('Date').reset_index(drop=True)

    if len(df) < 500:
        return None

    return df


def extract_test_period(df, test_start='2011-01-01', test_end='2019-12-31'):
    """
    Get index boundaries for the test period within a full DataFrame.
    Returns (start_idx, end_idx) using integer indices.
    """
    mask_start = df['Date'] >= test_start
    mask_end = df['Date'] <= test_end

    if not mask_start.any() or not mask_end.any():
        return None, None

    return mask_start.idxmax(), mask_end[::-1].idxmax()


def get_returns(prices):
    """Compute percentage returns: r_t = p_t/p_{t-1} - 1"""
    ret = np.zeros(len(prices))
    ret[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
    return ret
