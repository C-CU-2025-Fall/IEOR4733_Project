# Data Sources Log

> Purpose: Track data requirements and availability for reproducing "Deep Reinforcement Learning for Trading" (Zhang, Zohren, Roberts, 2019)

---

## ⚠️ IMPORTANT CORRECTION

**The paper uses FUTURES CONTRACTS, NOT EQUITIES!**

From the paper's abstract:
> "We adopt Deep Reinforcement Learning algorithms to design trading strategies for **continuous futures contracts**. We test our algorithms on the **50 most liquid futures contracts** from 2011 to 2019, and investigate how performance varies across different asset classes including **commodities, equity indices, fixed income and FX markets**."

---

## Yahoo Finance Coverage Check Results (2026-02-25)

> Based on paper's exact 50 contracts from Appendix A

### Overall Coverage: ✅ 40/50 contracts available (80%)

| Asset Class | Available | No Yahoo Mapping | Coverage |
|-------------|-----------|------------------|----------|
| **Commodities** | 22/25 | 3 | 88% ✅ |
| **Forex** | 9/9 | 0 | 100% ✅ |
| **Fixed Income** | 3/5 | 2 | 60% ⚠️ |
| **Equity Indexes** | 6/11 | 5 | 55% ⚠️ |

---

## Paper's Exact 50 Contracts (Appendix A)

### Commodities (25 contracts) - 22 available on Yahoo

| Pinnacle | Contract | Yahoo | Status |
|----------|----------|-------|--------|
| CC | COCOA | CC=F | ✅ |
| DA | MILK III | - | ⚠️ No mapping |
| GI | GOLDMAN SAKS C.I. | - | ⚠️ No mapping |
| JO | ORANGE JUICE | OJ=F | ✅ |
| KC | COFFEE | KC=F | ✅ |
| KW | WHEAT KC | - | ⚠️ No mapping |
| LB | LUMBER | LBS=F | ✅ |
| NR/ZR | ROUGH RICE | ZR=F | ✅ |
| SB | SUGAR #11 | SB=F | ✅ |
| ZA | PALLADIUM | PA=F | ✅ |
| ZC | CORN | ZC=F | ✅ |
| ZF | FEEDER CATTLE | GF=F | ✅ |
| ZG | GOLD | GC=F | ✅ |
| ZH | HEATING OIL | HO=F | ✅ |
| ZI | SILVER | SI=F | ✅ |
| ZK | COPPER | HG=F | ✅ |
| ZL | SOYBEAN OIL | ZL=F | ✅ |
| ZN | NATURAL GAS | NG=F | ✅ |
| ZO | OATS | ZO=F | ✅ |
| ZP | PLATINUM | PL=F | ✅ |
| ZT | LIVE CATTLE | LE=F | ✅ |
| ZU | CRUDE OIL | CL=F | ✅ |
| ZW | WHEAT | ZW=F | ✅ |
| ZZ | LEAN HOGS | HE=F | ✅ |

### Equity Indexes (11 contracts) - 6 available on Yahoo

| Pinnacle | Contract | Yahoo | Status |
|----------|----------|-------|--------|
| CA | CAC40 INDEX | - | ⚠️ No mapping (French) |
| EN | NASDAQ MINI | NQ=F | ✅ |
| ER | RUSSELL 2000 MINI | RTY=F | ✅ (624 rows, starts 2017) |
| ES | S&P 500 MINI | ES=F | ✅ |
| LX | FTSE 100 INDEX | - | ⚠️ No mapping (UK) |
| MD | S&P 400 MINI | - | ⚠️ No mapping |
| SC | S&P 500 COMPOSITE | ES=F | ✅ (same as ES) |
| SP | S&P 500 DAY SESSION | ES=F | ✅ (same as ES) |
| XU | DOW JONES EUROSTOXX50 | - | ⚠️ No mapping (European) |
| XX | DOW JONES STOXX 50 | - | ⚠️ No mapping (European) |
| YM | MINI DOW JONES | YM=F | ✅ |

### Fixed Income (5 contracts) - 3 available on Yahoo

| Pinnacle | Contract | Yahoo | Status |
|----------|----------|-------|--------|
| DT | EURO BOND BUND | - | ⚠️ No mapping (German) |
| FB | T-NOTE 5-year | ZF=F | ✅ |
| TY | T-NOTE 10-year | ZN=F | ✅ |
| UB | EURO BOBL | - | ⚠️ No mapping (German) |
| US | T-BONDS | ZB=F | ✅ |

### Forex (9 contracts) - 9 available on Yahoo (100%!)

| Pinnacle | Contract | Yahoo | Status |
|----------|----------|-------|--------|
| AN | AUSTRALIAN | 6A=F | ✅ |
| BN | BRITISH POUND | 6B=F | ✅ |
| CN | CANADIAN | 6C=F | ✅ |
| DX | US DOLLAR INDEX | DX=F | ✅ |
| FN | EURO | 6E=F | ✅ |
| JN | JAPANESE YEN | 6J=F | ✅ |
| MP | MEXICAN PESO | 6M=F | ✅ |
| NK | NIKKEI INDEX | NKD=F | ✅ |
| SN | SWISS FRANC | 6S=F | ✅ |

---

## Contracts NOT Available on Yahoo (10 total)

| Asset Class | Contract | Reason |
|-------------|----------|--------|
| Commodities | MILK III | Niche commodity |
| Commodities | GOLDMAN SAKS C.I. | Proprietary index |
| Commodities | WHEAT KC | Regional wheat variant |
| Equity Indexes | CAC40 INDEX | French index |
| Equity Indexes | FTSE 100 INDEX | UK index |
| Equity Indexes | S&P 400 MINI | MidCap futures |
| Equity Indexes | DOW JONES EUROSTOXX50 | European index |
| Equity Indexes | DOW JONES STOXX 50 | European index |
| Fixed Income | EURO BOND BUND | German government bond |
| Fixed Income | EURO BOBL | German government bond |

---

## Data Quality Notes

- Most contracts have ~2,260 rows (full 2011-2019 coverage)
- RTY=F only has 624 rows (Russell 2000 mini started 2017-07-10)
- Some contracts (SC, SP) map to same Yahoo ticker (ES=F)

---

## Action Items

### Completed
- [x] Extract exact 50 contracts from paper Appendix A
- [x] Map Pinnacle Data tickers to Yahoo Finance format
- [x] Check Yahoo Finance coverage for all 50 contracts
- [x] Verify data covers all 4 asset classes

### Remaining
- [ ] Download 40 available futures contracts from Yahoo Finance
- [ ] Consider Pinnacle Data CLC for 10 missing contracts
- [ ] Note: 40/50 (80%) coverage is sufficient for reproduction

---

## Previous (INCORRECT) Data - To Be Replaced

The following data was downloaded based on incorrect understanding (equities instead of futures):
- ❌ `data/sp500_prices.csv` - S&P 500 stock prices (WRONG DATA TYPE)
- ✅ `data/index_data.csv` - S&P 500 index, VIX (may still be useful for benchmark)
- ✅ `data/risk_free_rate.csv` - Risk-free rate (still needed)

---

## Changelog

| Date | Change |
|------|--------|
| 2026-02-25 | Initial data sources log created (incorrectly assumed equities) |
| 2026-02-25 | **MAJOR CORRECTION**: Paper uses futures contracts, not equities |
| 2026-02-25 | Extracted exact 50 contracts from paper Appendix A |
| 2026-02-25 | Coverage check: 40/50 (80%) available on Yahoo Finance |