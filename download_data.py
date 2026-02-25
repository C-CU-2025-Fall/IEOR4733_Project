#!/usr/bin/env python3
"""
Data Download Script for IEOR4733 Project
Paper: "Deep Reinforcement Learning for Trading" by Zhang, Zohren, and Roberts (2019)

This script downloads all required data with proper rate limiting to be website-friendly.
"""

import os
import time
import random
from datetime import datetime
import pandas as pd
import numpy as np

# Rate limiting settings (to avoid getting banned)
MIN_DELAY = 1.0  # Minimum delay between requests (seconds)
MAX_DELAY = 3.0  # Maximum delay between requests (seconds)
BATCH_SIZE = 50  # Number of tickers per batch
BATCH_DELAY = 10  # Delay between batches (seconds)

# Date range (adjusted from paper for realistic course project scope)
START_DATE = '2010-01-01'
END_DATE = '2023-12-31'

# Output directory
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)


def random_delay(min_sec=MIN_DELAY, max_sec=MAX_DELAY):
    """Add random delay between requests to be website-friendly."""
    delay = random.uniform(min_sec, max_sec)
    print(f"  ⏳ Waiting {delay:.1f}s (rate limiting)...")
    time.sleep(delay)


def get_sp500_tickers():
    """
    Get current S&P 500 constituent tickers from Wikipedia.
    Paper reference: "We use the S&P 500 index constituents as our investment universe"
    """
    print("\n📋 Fetching S&P 500 constituent list...")
    
    try:
        # Use requests with proper headers to avoid 403 Forbidden
        import requests
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML with pandas
        tables = pd.read_html(response.text)
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()
        
        # Clean tickers (remove dots, convert to standard format)
        tickers = [t.replace('.', '-') for t in tickers]
        
        print(f"  ✅ Found {len(tickers)} S&P 500 constituents")
        return tickers
        
    except Exception as e:
        print(f"  ❌ Error fetching S&P 500 list: {e}")
        # Fallback to a predefined list of major stocks
        print("  📝 Using fallback list of major S&P 500 stocks...")
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
            'V', 'JPM', 'XOM', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'LLY',
            'PEP', 'KO', 'COST', 'AVGO', 'MCD', 'WMT', 'CSCO', 'TMO', 'ACN', 'ABT',
            'DHR', 'CMCSA', 'NKE', 'ADBE', 'NFLX', 'VZ', 'QCOM', 'INTC', 'MDT', 'TXN',
            'WFC', 'NEE', 'BMY', 'PM', 'HON', 'RTX', 'LIN', 'ORCL', 'UPS', 'AMGN'
        ]


def download_stock_prices(tickers, start_date, end_date, output_file):
    """
    Download historical stock prices using yfinance with rate limiting.
    Paper reference: "We use daily closing prices"
    """
    import yfinance as yf
    
    print(f"\n📈 Downloading stock prices for {len(tickers)} tickers...")
    print(f"   Date range: {start_date} to {end_date}")
    
    all_data = []
    failed_tickers = []
    
    # Process in batches to be website-friendly
    total_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_num in range(total_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(tickers))
        batch_tickers = tickers[start_idx:end_idx]
        
        print(f"\n  📦 Batch {batch_num + 1}/{total_batches}: Processing tickers {start_idx + 1}-{end_idx}")
        
        for i, ticker in enumerate(batch_tickers):
            try:
                print(f"    [{start_idx + i + 1}/{len(tickers)}] Downloading {ticker}...", end='')
                
                # Download data
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date, auto_adjust=True)
                
                if df.empty:
                    print(" ❌ No data")
                    failed_tickers.append(ticker)
                else:
                    # Keep relevant columns
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    df['Ticker'] = ticker
                    df = df.reset_index()
                    all_data.append(df)
                    print(f" ✅ {len(df)} rows")
                
                # Rate limiting between individual requests
                if i < len(batch_tickers) - 1:
                    random_delay(0.5, 1.5)
                    
            except Exception as e:
                print(f" ❌ Error: {e}")
                failed_tickers.append(ticker)
        
        # Longer delay between batches
        if batch_num < total_batches - 1:
            print(f"\n  ⏳ Batch complete. Waiting {BATCH_DELAY}s before next batch...")
            time.sleep(BATCH_DELAY)
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"\n  ✅ Saved {len(combined_df)} rows for {len(all_data)} tickers to {output_file}")
    else:
        print("\n  ❌ No data downloaded!")
        return None
    
    if failed_tickers:
        print(f"\n  ⚠️ Failed to download {len(failed_tickers)} tickers: {failed_tickers[:10]}...")
    
    return combined_df


def download_index_data(start_date, end_date, output_file):
    """
    Download S&P 500 index and VIX data.
    Paper reference: S&P 500 index used as benchmark
    """
    import yfinance as yf
    
    print("\n📊 Downloading index data...")
    
    indices = {
        '^GSPC': 'SP500',      # S&P 500 Index
        '^VIX': 'VIX',          # Volatility Index
        'SPY': 'SPY_ETF'        # S&P 500 ETF (alternative)
    }
    
    all_data = []
    
    for ticker, name in indices.items():
        try:
            print(f"  📥 Downloading {name} ({ticker})...", end='')
            
            index = yf.Ticker(ticker)
            df = index.history(start=start_date, end=end_date, auto_adjust=True)
            
            if not df.empty:
                df = df[['Close']].copy()
                df.columns = [name]
                df.index.name = 'Date'
                all_data.append(df)
                print(f" ✅ {len(df)} rows")
            else:
                print(" ❌ No data")
                
        except Exception as e:
            print(f" ❌ Error: {e}")
        
        # Rate limiting
        random_delay()
    
    if all_data:
        combined_df = pd.concat(all_data, axis=1)
        combined_df = combined_df.reset_index()
        combined_df.to_csv(output_file, index=False)
        print(f"\n  ✅ Saved index data to {output_file}")
        return combined_df
    else:
        print("\n  ❌ No index data downloaded!")
        return None


def download_risk_free_rate(start_date, end_date, output_file):
    """
    Download 3-Month Treasury Bill rate from FRED.
    Paper reference: "The risk-free rate is the 3-month T-bill rate"
    
    Uses pandas_datareader as fallback (no API key required).
    """
    print("\n💰 Downloading risk-free rate (3-Month T-Bill)...")
    
    try:
        # Try pandas_datareader first (no API key needed)
        import pandas_datareader.data as web
        
        print("  📥 Using pandas_datareader (FRED)...", end='')
        random_delay(0.5, 1.0)  # Brief delay before request
        
        rf_data = web.DataReader('DTB3', 'fred', start=start_date, end=end_date)
        
        if not rf_data.empty:
            # Forward fill missing values (common for daily T-bill data)
            rf_data = rf_data.fillna(method='ffill')
            rf_data = rf_data.reset_index()
            rf_data.columns = ['Date', 'RiskFreeRate']
            rf_data.to_csv(output_file, index=False)
            print(f" ✅ {len(rf_data)} rows")
            print(f"\n  ✅ Saved risk-free rate data to {output_file}")
            return rf_data
        else:
            print(" ❌ No data")
            
    except Exception as e:
        print(f" ❌ pandas_datareader failed: {e}")
        
        # Fallback: try fredapi if available
        try:
            print("  📥 Trying fredapi as fallback...", end='')
            from fredapi import Fred
            
            # Note: You need to get a free API key from https://fred.stlouisfed.org/docs/api/api_key.html
            # and set it as environment variable: export FRED_API_KEY=your_key
            api_key = os.environ.get('FRED_API_KEY')
            if not api_key:
                print(" ❌ No FRED API key found. Set FRED_API_KEY environment variable.")
                print("     Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
                return None
                
            fred = Fred(api_key=api_key)
            rf_series = fred.get_series('DTB3', observation_start=start_date, observation_end=end_date)
            
            rf_data = rf_series.to_frame('RiskFreeRate')
            rf_data.index.name = 'Date'
            rf_data = rf_data.fillna(method='ffill')
            rf_data = rf_data.reset_index()
            rf_data.to_csv(output_file, index=False)
            print(f" ✅ {len(rf_data)} rows")
            return rf_data
            
        except Exception as e2:
            print(f" ❌ fredapi also failed: {e2}")
            return None


def verify_data():
    """Verify downloaded data completeness."""
    print("\n" + "="*60)
    print("📊 DATA VERIFICATION")
    print("="*60)
    
    files = {
        'Stock Prices': f'{DATA_DIR}/sp500_prices.csv',
        'Index Data': f'{DATA_DIR}/index_data.csv',
        'Risk-Free Rate': f'{DATA_DIR}/risk_free_rate.csv'
    }
    
    all_ok = True
    
    for name, filepath in files.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            print(f"\n✅ {name}:")
            print(f"   File: {filepath}")
            print(f"   Rows: {len(df):,}")
            print(f"   Columns: {list(df.columns)}")
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                print(f"   Date Range: {df['Date'].min()} to {df['Date'].max()}")
        else:
            print(f"\n❌ {name}: File not found - {filepath}")
            all_ok = False
    
    return all_ok


def main():
    """Main function to download all data."""
    print("="*60)
    print("🚀 IEOR4733 Data Download Script")
    print("="*60)
    print(f"📅 Date Range: {START_DATE} to {END_DATE}")
    print(f"📁 Output Directory: {DATA_DIR}/")
    print(f"⏱️ Rate Limiting: {MIN_DELAY}-{MAX_DELAY}s between requests")
    print("="*60)
    
    # Step 1: Get S&P 500 tickers
    tickers = get_sp500_tickers()
    
    # Step 2: Download stock prices
    stock_file = f'{DATA_DIR}/sp500_prices.csv'
    if not os.path.exists(stock_file):
        download_stock_prices(tickers, START_DATE, END_DATE, stock_file)
    else:
        print(f"\n⏭️ Stock prices already exist: {stock_file}")
    
    # Step 3: Download index data
    index_file = f'{DATA_DIR}/index_data.csv'
    if not os.path.exists(index_file):
        download_index_data(START_DATE, END_DATE, index_file)
    else:
        print(f"\n⏭️ Index data already exists: {index_file}")
    
    # Step 4: Download risk-free rate
    rf_file = f'{DATA_DIR}/risk_free_rate.csv'
    if not os.path.exists(rf_file):
        download_risk_free_rate(START_DATE, END_DATE, rf_file)
    else:
        print(f"\n⏭️ Risk-free rate already exists: {rf_file}")
    
    # Step 5: Verify data
    verify_data()
    
    print("\n" + "="*60)
    print("✅ Data download complete!")
    print("="*60)


if __name__ == "__main__":
    main()