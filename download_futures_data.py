#!/usr/bin/env python3
"""
Download Futures Data for IEOR4733 Project
Paper: "Deep Reinforcement Learning for Trading" by Zhang, Zohren, and Roberts (2019)

Downloads 45 available futures contracts from Yahoo Finance with rate limiting.
Updated to use futures_coverage_results.json format.
"""

import os
import time
import random
import json
import pandas as pd
import yfinance as yf

# Rate limiting settings (be very friendly to Yahoo!)
MIN_DELAY = 2.0
MAX_DELAY = 4.0
BATCH_SIZE = 10
BATCH_DELAY = 15  # Longer delay between batches

# Date range - 尝试获取更早数据
# 论文：2005-2019
# 目标：2005-2019（如果 Yahoo Finance 有数据）
START_DATE = '2005-01-01'
END_DATE = '2019-12-31'

# Output directory
DATA_DIR = 'data/futures'
os.makedirs(DATA_DIR, exist_ok=True)

# Additional contracts found during quality check (not in original JSON)
ADDITIONAL_CONTRACTS = [
    {"ticker": "KE=F", "name": "Kansas Wheat", "asset_class": "Commodities - Agriculture", "rows": 2261},
    {"ticker": "DC=F", "name": "Class III Milk", "asset_class": "Commodities - Agriculture", "rows": 2243},
]

def random_delay(min_sec=None, max_sec=None):
    """Add random delay to avoid getting banned."""
    min_sec = min_sec or MIN_DELAY
    max_sec = max_sec or MAX_DELAY
    delay = random.uniform(min_sec, max_sec)
    print(f"  ⏳ Waiting {delay:.1f}s...")
    time.sleep(delay)

def get_available_tickers():
    """Extract list of available Yahoo tickers from coverage results."""
    # Load coverage results
    with open('futures_coverage_results.json', 'r') as f:
        coverage_results = json.load(f)
    
    tickers = []
    for asset_class, data in coverage_results['by_asset_class'].items():
        for item in data['available']:
            tickers.append({
                'ticker': item['ticker'],
                'name': item.get('name', item['ticker']),
                'asset_class': asset_class,
                'rows': item.get('rows', 0)
            })
    
    # Add additional contracts
    for item in ADDITIONAL_CONTRACTS:
        tickers.append(item)
    
    return tickers

def download_futures_data(force_redownload=False):
    """Download all available futures data with rate limiting."""
    tickers = get_available_tickers()
    
    print("="*70)
    print("📥 Downloading Futures Data")
    print("="*70)
    print(f"Total contracts to download: {len(tickers)}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Output directory: {DATA_DIR}/")
    if force_redownload:
        print("⚠️  Force re-download mode (overwriting existing files)")
    print("="*70)
    
    # Check for already downloaded files (resume capability)
    already_downloaded = set()
    if not force_redownload:
    if os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            if f.endswith('.csv'):
                ticker = f.replace('.csv', '')
                already_downloaded.add(ticker)
    
    if already_downloaded:
        print(f"\n✅ Already downloaded: {len(already_downloaded)} contracts")
    
    to_download = [t for t in tickers if t['ticker'] not in already_downloaded]
    
    if not to_download:
        print("\n✅ All contracts already downloaded!")
        return
    
    print(f"\n📥 To download: {len(to_download)} contracts")
    
    # Download in batches
    total_batches = (len(to_download) + BATCH_SIZE - 1) // BATCH_SIZE
    failed = []
    
    for batch_num in range(total_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(to_download))
        batch = to_download[start_idx:end_idx]
        
        print(f"\n📦 Batch {batch_num + 1}/{total_batches}")
        
        for i, ticker_info in enumerate(batch):
            ticker = ticker_info['ticker']
            name = ticker_info.get('name', ticker)
            asset_class = ticker_info['asset_class']
            
            print(f"\n  [{start_idx + i + 1}/{len(to_download)}] {ticker} ({name}) - {asset_class}")
            
            try:
                # Download data
                df = yf.download(ticker, start=START_DATE, end=END_DATE, 
                               progress=False, auto_adjust=True)
                
                if df is not None and not df.empty:
                    # Save to CSV
                    output_file = os.path.join(DATA_DIR, f"{ticker}.csv")
                    df.to_csv(output_file)
                    print(f"    ✅ Downloaded {len(df)} rows → {output_file}")
                else:
                    print(f"    ❌ No data available")
                    failed.append(ticker)
                    
            except Exception as e:
                print(f"    ❌ Error: {str(e)[:50]}")
                failed.append(ticker)
            
            # Rate limiting between downloads
            if start_idx + i + 1 < len(to_download):
                random_delay()
        
        # Longer delay between batches
        if batch_num + 1 < total_batches:
            print(f"\n  ⏳ Batch complete. Waiting {BATCH_DELAY}s before next batch...")
            time.sleep(BATCH_DELAY)
    
    # Summary
    print("\n" + "="*70)
    print("📊 DOWNLOAD SUMMARY")
    print("="*70)
    
    downloaded_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    print(f"\n✅ Successfully downloaded: {len(downloaded_files)} contracts")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)} contracts")
        print(f"   {failed}")
    
    # Save manifest
    manifest = {
        'downloaded': [t['ticker'] for t in tickers if t['ticker'] in [f.replace('.csv', '') for f in downloaded_files]],
        'failed': failed,
        'date_range': f"{START_DATE} to {END_DATE}",
        'total_contracts': len(tickers)
    }
    
    with open(os.path.join(DATA_DIR, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\n💾 Manifest saved to {DATA_DIR}/manifest.json")

if __name__ == "__main__":
    import sys
    force = '--force' in sys.argv or '-f' in sys.argv
    download_futures_data(force_redownload=force)
