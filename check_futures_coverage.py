#!/usr/bin/env python3
"""
Check Yahoo Finance Futures Coverage
Compares paper's exact futures contracts (from Appendix A) with Yahoo Finance availability.

Paper uses Pinnacle Data tickers - we need to map them to Yahoo Finance format.
"""

import time
import random
import json
import yfinance as yf

# Rate limiting settings
MIN_DELAY = 2.0
MAX_DELAY = 4.0

def random_delay():
    delay = random.uniform(MIN_DELAY, MAX_DELAY)
    print(f"  ⏳ Waiting {delay:.1f}s...")
    time.sleep(delay)

# Paper's exact 50 futures contracts from Appendix A
# Format: "Pinnacle Ticker": {"name": "Contract Name", "yahoo": "Yahoo Finance Ticker or None"}
PAPER_CONTRACTS = {
    # Commodities (25 contracts)
    "CC": {"name": "COCOA", "yahoo": "CC=F"},
    "DA": {"name": "MILK III", "yahoo": None},  # No Yahoo equivalent
    "GI": {"name": "GOLDMAN SAKS C.I.", "yahoo": None},  # No Yahoo equivalent
    "JO": {"name": "ORANGE JUICE", "yahoo": "OJ=F"},
    "KC": {"name": "COFFEE", "yahoo": "KC=F"},
    "KW": {"name": "WHEAT KC", "yahoo": None},  # KC Wheat - different from regular wheat
    "LB": {"name": "LUMBER", "yahoo": "LBS=F"},  # Try Lumber futures
    "NR": {"name": "ROUGH RICE", "yahoo": "ZR=F"},  # Rough rice
    "SB": {"name": "SUGAR #11", "yahoo": "SB=F"},
    "ZA": {"name": "PALLADIUM", "yahoo": "PA=F"},
    "ZC": {"name": "CORN", "yahoo": "ZC=F"},
    "ZF": {"name": "FEEDER CATTLE", "yahoo": "GF=F"},  # Feeder cattle
    "ZG": {"name": "GOLD", "yahoo": "GC=F"},
    "ZH": {"name": "HEATING OIL", "yahoo": "HO=F"},
    "ZI": {"name": "SILVER", "yahoo": "SI=F"},
    "ZK": {"name": "COPPER", "yahoo": "HG=F"},
    "ZL": {"name": "SOYBEAN OIL", "yahoo": "ZL=F"},
    "ZN": {"name": "NATURAL GAS", "yahoo": "NG=F"},  # Note: ZN is Natural Gas in paper, not 10Y Treasury
    "ZO": {"name": "OATS", "yahoo": "ZO=F"},
    "ZP": {"name": "PLATINUM", "yahoo": "PL=F"},
    "ZR": {"name": "ROUGH RICE", "yahoo": "ZR=F"},  # Duplicate?
    "ZT": {"name": "LIVE CATTLE", "yahoo": "LE=F"},  # Live cattle
    "ZU": {"name": "CRUDE OIL", "yahoo": "CL=F"},
    "ZW": {"name": "WHEAT", "yahoo": "ZW=F"},
    "ZZ": {"name": "LEAN HOGS", "yahoo": "HE=F"},  # Lean hogs
    
    # Equity Indexes (11 contracts)
    "CA": {"name": "CAC40 INDEX", "yahoo": None},  # French index - may not be on Yahoo
    "EN": {"name": "NASDAQ MINI", "yahoo": "NQ=F"},
    "ER": {"name": "RUSSELL 2000 MINI", "yahoo": "RTY=F"},
    "ES": {"name": "S&P 500 MINI", "yahoo": "ES=F"},
    "LX": {"name": "FTSE 100 INDEX", "yahoo": None},  # UK index
    "MD": {"name": "S&P 400 MINI", "yahoo": None},  # MidCap - not on Yahoo
    "SC": {"name": "S&P 500 COMPOSITE", "yahoo": "ES=F"},  # Same as ES
    "SP": {"name": "S&P 500 DAY SESSION", "yahoo": "ES=F"},  # Same as ES
    "XU": {"name": "DOW JONES EUROSTOXX50", "yahoo": None},  # European index
    "XX": {"name": "DOW JONES STOXX 50", "yahoo": None},  # European index
    "YM": {"name": "MINI DOW JONES", "yahoo": "YM=F"},
    
    # Fixed Income (5 contracts)
    "DT": {"name": "EURO BOND BUND", "yahoo": None},  # German bund - not on Yahoo
    "FB": {"name": "T-NOTE 5-year", "yahoo": "ZF=F"},
    "TY": {"name": "T-NOTE 10-year", "yahoo": "ZN=F"},
    "UB": {"name": "EURO BOBL", "yahoo": None},  # German bond - not on Yahoo
    "US": {"name": "T-BONDS", "yahoo": "ZB=F"},
    
    # Forex (9 contracts)
    "AN": {"name": "AUSTRALIAN", "yahoo": "6A=F"},
    "BN": {"name": "BRITISH POUND", "yahoo": "6B=F"},
    "CN": {"name": "CANADIAN", "yahoo": "6C=F"},
    "DX": {"name": "US DOLLAR INDEX", "yahoo": "DX=F"},
    "FN": {"name": "EURO", "yahoo": "6E=F"},
    "JN": {"name": "JAPANESE YEN", "yahoo": "6J=F"},
    "MP": {"name": "MEXICAN PESO", "yahoo": "6M=F"},
    "NK": {"name": "NIKKEI INDEX", "yahoo": "NKD=F"},  # This is equity index but listed in Forex
    "SN": {"name": "SWISS FRANC", "yahoo": "6S=F"},
}

def check_coverage():
    """Check Yahoo Finance coverage for paper's exact contracts."""
    print("="*70)
    print("🔍 Yahoo Finance Coverage Check - Paper's Exact 50 Contracts")
    print("="*70)
    print(f"Paper: 'Deep Reinforcement Learning for Trading' (Zhang et al., 2019)")
    print(f"Total contracts in paper: {len(PAPER_CONTRACTS)}")
    print(f"Date range: 2011-01-01 to 2019-12-31")
    print("="*70)
    
    results = {
        "by_asset_class": {
            "Commodities": {"available": [], "not_available": [], "no_yahoo_ticker": []},
            "Equity Indexes": {"available": [], "not_available": [], "no_yahoo_ticker": []},
            "Fixed Income": {"available": [], "not_available": [], "no_yahoo_ticker": []},
            "Forex": {"available": [], "not_available": [], "no_yahoo_ticker": []},
        },
        "summary": {
            "total": len(PAPER_CONTRACTS),
            "available": 0,
            "not_available": 0,
            "no_yahoo_ticker": 0,
        }
    }
    
    # Categorize by asset class
    commodities = ["CC", "DA", "GI", "JO", "KC", "KW", "LB", "NR", "SB", "ZA", 
                   "ZC", "ZF", "ZG", "ZH", "ZI", "ZK", "ZL", "ZN", "ZO", "ZP", 
                   "ZR", "ZT", "ZU", "ZW", "ZZ"]
    equity_indexes = ["CA", "EN", "ER", "ES", "LX", "MD", "SC", "SP", "XU", "XX", "YM"]
    fixed_income = ["DT", "FB", "TY", "UB", "US"]
    forex = ["AN", "BN", "CN", "DX", "FN", "JN", "MP", "NK", "SN"]
    
    def get_asset_class(ticker):
        if ticker in commodities:
            return "Commodities"
        elif ticker in equity_indexes:
            return "Equity Indexes"
        elif ticker in fixed_income:
            return "Fixed Income"
        elif ticker in forex:
            return "Forex"
        return "Unknown"
    
    count = 0
    total = len(PAPER_CONTRACTS)
    
    for pinnacle_ticker, info in PAPER_CONTRACTS.items():
        count += 1
        asset_class = get_asset_class(pinnacle_ticker)
        name = info["name"]
        yahoo_ticker = info["yahoo"]
        
        print(f"\n[{count}/{total}] {pinnacle_ticker} ({name}) - {asset_class}")
        
        if yahoo_ticker is None:
            print(f"  ⚠️ No Yahoo Finance equivalent mapped")
            results["by_asset_class"][asset_class]["no_yahoo_ticker"].append({
                "pinnacle": pinnacle_ticker,
                "name": name
            })
            results["summary"]["no_yahoo_ticker"] += 1
            continue
        
        print(f"  📥 Testing Yahoo: {yahoo_ticker}...", end="")
        
        try:
            df = yf.download(yahoo_ticker, start="2011-01-01", end="2019-12-31",
                           progress=False, auto_adjust=True)
            
            if df is not None and not df.empty:
                rows = len(df)
                start_date = df.index[0].strftime('%Y-%m-%d')
                end_date = df.index[-1].strftime('%Y-%m-%d')
                print(f" ✅ Available ({rows} rows, {start_date} to {end_date})")
                results["by_asset_class"][asset_class]["available"].append({
                    "pinnacle": pinnacle_ticker,
                    "yahoo": yahoo_ticker,
                    "name": name,
                    "rows": rows,
                    "start": start_date,
                    "end": end_date
                })
                results["summary"]["available"] += 1
            else:
                print(" ❌ No data")
                results["by_asset_class"][asset_class]["not_available"].append({
                    "pinnacle": pinnacle_ticker,
                    "yahoo": yahoo_ticker,
                    "name": name
                })
                results["summary"]["not_available"] += 1
                
        except Exception as e:
            print(f" ❌ Error: {str(e)[:50]}")
            results["by_asset_class"][asset_class]["not_available"].append({
                "pinnacle": pinnacle_ticker,
                "yahoo": yahoo_ticker,
                "name": name,
                "error": str(e)
            })
            results["summary"]["not_available"] += 1
        
        # Rate limiting
        if count < total:
            random_delay()
    
    # Print summary
    print("\n" + "="*70)
    print("📊 COVERAGE SUMMARY")
    print("="*70)
    
    s = results["summary"]
    print(f"\nTotal: {s['total']} contracts")
    print(f"✅ Available on Yahoo: {s['available']} ({s['available']/s['total']*100:.0f}%)")
    print(f"❌ Not available: {s['not_available']}")
    print(f"⚠️ No Yahoo ticker mapped: {s['no_yahoo_ticker']}")
    
    print("\n📋 By Asset Class:")
    for asset_class, data in results["by_asset_class"].items():
        avail = len(data["available"])
        no_ticker = len(data["no_yahoo_ticker"])
        not_avail = len(data["not_available"])
        total_class = avail + no_ticker + not_avail
        print(f"\n  {asset_class} ({total_class} contracts):")
        print(f"    ✅ Available: {avail}")
        print(f"    ❌ Not available: {not_avail}")
        print(f"    ⚠️ No Yahoo ticker: {no_ticker}")
        
        if data["available"]:
            tickers = [d["yahoo"] for d in data["available"]]
            print(f"    Available: {', '.join(tickers)}")
        if data["no_yahoo_ticker"]:
            names = [d["name"] for d in data["no_yahoo_ticker"]]
            print(f"    No Yahoo mapping: {', '.join(names)}")
    
    # Save results
    with open("futures_coverage_paper_exact.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to futures_coverage_paper_exact.json")
    
    return results


if __name__ == "__main__":
    check_coverage()