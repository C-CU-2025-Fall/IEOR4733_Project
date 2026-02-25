# Data Sources Log

> Purpose: Track data requirements and availability for reproducing "Deep Reinforcement Learning for Trading" (Zhang, Zohren, Roberts, 2019)

---

## ⚠️ IMPORTANT CORRECTION

**The paper uses FUTURES CONTRACTS, NOT EQUITIES!**

From the paper's abstract:
> "We adopt Deep Reinforcement Learning algorithms to design trading strategies for **continuous futures contracts**. We test our algorithms on the **50 most liquid futures contracts** from 2011 to 2019, and investigate how performance varies across different asset classes including **commodities, equity indices, fixed income and FX markets**."

---

## Correct Data Requirements Summary

| # | Data Type | Paper Requirement | Source | Free? | Status | Notes |
|---|-----------|-------------------|--------|-------|--------|-------|
| 1 | **Futures Contracts** | 50 most liquid continuous futures | Pinnacle Data CLC / Others | ❓ Paid? | 🔴 Researching | Commodities, equity indices, fixed income, FX |
| 2 | **Data Period** | 2011-2019 | - | - | 🔴 Researching | ~8 years of daily data |
| 3 | **Continuous Contracts** | Back-adjusted futures | - | - | 🔴 Researching | Need CLC (Continuously Linked Contracts) |

---

## Paper's Data Specification

From the paper (Section 3 - Data):

| Specification | Details |
|---------------|---------|
| **Instrument Type** | Continuous futures contracts |
| **Number of Contracts** | 50 most liquid futures |
| **Time Period** | 2011 to 2019 |
| **Asset Classes** | Commodities, Equity Indices, Fixed Income, FX |
| **Data Frequency** | Daily |
| **Contract Type** | Continuously linked (back-adjusted) |

### Asset Class Breakdown (Typical 50 Liquid Futures)

| Asset Class | Example Contracts |
|-------------|-------------------|
| **Equity Indices** | ES (S&P 500), NQ (Nasdaq), YM (Dow), RTY (Russell) |
| **Commodities - Energy** | CL (Crude Oil), NG (Natural Gas), RB (Gasoline) |
| **Commodities - Metals** | GC (Gold), SI (Silver), HG (Copper) |
| **Commodities - Agriculture** | ZC (Corn), ZS (Soybeans), ZW (Wheat) |
| **Fixed Income** | ZN (10Y Treasury), ZB (30Y Treasury), GE (Eurodollar) |
| **FX** | 6E (Euro), 6J (Yen), 6B (Pound), 6A (Australian) |

---

## Data Source Options

### Option 1: Pinnacle Data CLC (Paper's Likely Source)
**Website:** https://pinnacledata2.com/clc.html

| Aspect | Details |
|--------|---------|
| **Product** | Continuously Linked Commodity Contracts (CLC) |
| **Coverage** | 100+ futures contracts, 30+ years |
| **Format** | CSV, OHLCV data |
| **Cost** | ❓ Paid (research needed) |
| **Pros** | Professional quality, exactly what paper uses |
| **Cons** | May require purchase |

### Option 2: Free Alternatives

| Source | Coverage | Free? | Notes |
|--------|----------|-------|-------|
| **Yahoo Finance** | Some futures (e.g., `ES=F`, `CL=F`) | ✅ Yes | Limited continuous contracts |
| **Quandl** | SCF database (continuous futures) | ✅ Free tier | Limited contracts |
| **FirstRate Data** | Futures data | ❓ Paid | Historical continuous contracts |
| **Norgate Data** | Futures | ❌ Paid | Professional quality |
| **WRDS** | Commodity Research Bureau | ✅ Academic | May have futures data |

### Option 3: Yahoo Finance (Limited Free Access)

Yahoo Finance provides some futures data with `=F` suffix:

```python
import yfinance as yf

# Example futures tickers
futures_tickers = [
    'ES=F',   # S&P 500 Futures
    'NQ=F',   # Nasdaq 100 Futures
    'YM=F',   # Dow Jones Futures
    'CL=F',   # Crude Oil Futures
    'GC=F',   # Gold Futures
    'SI=F',   # Silver Futures
    'ZN=F',   # 10-Year Treasury
    'ZB=F',   # 30-Year Treasury
    '6E=F',   # Euro FX
    '6J=F',   # Japanese Yen
]

# Download data
data = yf.download(futures_tickers, start='2011-01-01', end='2019-12-31')
```

**Limitation:** Yahoo Finance futures data may not be properly back-adjusted (continuous).

---

## Yahoo Finance Coverage Check Results (2026-02-25)

### Overall Coverage: ✅ 43/49 contracts available (88%)

| Asset Class | Available | Total | Coverage |
|-------------|-----------|-------|----------|
| **Commodities - Energy** | 5 | 5 | 100% ✅ |
| **Commodities - Metals** | 5 | 5 | 100% ✅ |
| **Commodities - Agriculture** | 10 | 10 | 100% ✅ |
| **FX** | 10 | 10 | 100% ✅ |
| **Fixed Income** | 7 | 9 | 78% ⚠️ |
| **Equity Indices** | 6 | 10 | 60% ⚠️ |

### Available Contracts by Asset Class

**Equity Indices (6/10):**
- ✅ ES=F (S&P 500), NQ=F (Nasdaq), YM=F (Dow), RTY=F (Russell 2000)
- ✅ NKD=F (Nikkei 225), DAX=F (DAX - limited data)
- ❌ EMD=F, HSI=F, FTX=F, STX=F (not available)

**Commodities - Energy (5/5):**
- ✅ CL=F (Crude Oil), NG=F (Natural Gas), RB=F (Gasoline), HO=F (Heating Oil), BZ=F (Brent)

**Commodities - Metals (5/5):**
- ✅ GC=F (Gold), SI=F (Silver), HG=F (Copper), PL=F (Platinum), PA=F (Palladium)

**Commodities - Agriculture (10/10):**
- ✅ ZC=F (Corn), ZS=F (Soybeans), ZW=F (Wheat), ZL=F (Soybean Oil), ZM=F (Soybean Meal)
- ✅ KC=F (Coffee), CT=F (Cotton), SB=F (Sugar), CC=F (Cocoa), OJ=F (Orange Juice)

**Fixed Income (7/9):**
- ✅ ZN=F (10Y Treasury), ZB=F (30Y Treasury), ZF=F (5Y Treasury), ZT=F (2Y Treasury)
- ✅ GE=F (Eurodollar), TN=F (Ultra 10Y), UB=F (Ultra Bond)
- ❌ FV=F, TU=F (not available - older contract codes)

**FX (10/10):**
- ✅ 6E=F (Euro), 6J=F (Yen), 6B=F (Pound), 6A=F (AUD), 6C=F (CAD)
- ✅ 6S=F (Swiss Franc), 6M=F (Mexican Peso), 6N=F (NZD), 6R=F (Ruble), DX=F (Dollar Index)

### Data Quality Notes
- Most contracts have ~2,260 rows (full 2011-2019 coverage)
- RTY=F only has 624 rows (started 2017-07-10)
- DAX=F only has 41 rows (limited coverage)

### Action Items

#### Completed
- [x] Check Yahoo Finance futures data coverage
- [x] Identify 50 most liquid futures contracts
- [x] Verify data covers all 4 asset classes

#### Remaining
- [ ] Download full futures data for 43 available contracts
- [ ] Note: 43 contracts exceeds paper's 50 requirement (some substitutions needed)
- [ ] Consider Pinnacle Data CLC for missing contracts (EMD, HSI, FTX, STX, FV, TU)

---

## Previous (INCORRECT) Data - To Be Replaced

The following data was downloaded based on incorrect understanding (equities instead of futures):
- ❌ `data/sp500_prices.csv` - S&P 500 stock prices (WRONG DATA TYPE)
- ✅ `data/index_data.csv` - S&P 500 index, VIX (may still be useful for benchmark)
- ✅ `data/risk_free_rate.csv` - Risk-free rate (still needed)

---

## Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-02-25 | Initial data sources log created (incorrectly assumed equities) | - |
| 2026-02-25 | **MAJOR CORRECTION**: Paper uses futures contracts, not equities | - |