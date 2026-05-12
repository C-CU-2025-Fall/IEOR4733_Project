# Data Directory

This directory holds market data required to reproduce the backtests. Due to licensing restrictions, raw vendor files are **not** included in this repository.

## Required Data Sources

### CLCData (Primary)

- **Source**: [CLCData](https://www.clcdata.com/) — Continuous Linked Contracts (CLC) futures data
- **Format**: Per-contract RAD (Ratio-Adjusted) CSV files — one file per ticker (e.g., `ES_RAD.CSV`)
- **Location**: Place files under `data/CLC/`
- **Contracts**: 50 futures contracts across 4 asset classes (Commodity, Equity Index, Fixed Income, Forex). See `config.py` for the full `PAPER_50` list.
- **Date range**: 2009-01-01 through 2019-12-31 (2011-01-01 through 2019-12-31 for backtesting)

### Yahoo Finance (Supplementary)

- Used as fallback for a small number of equity-index contracts
- **Location**: Place files under `data/yahoo/`
- **Format**: Standard OHLCV CSV (e.g., `ES_yahoo.csv`)

### Other files

| File | Description |
|---|---|
| `data/index_data.csv` | Pre-merged index-level daily prices |
| `data/risk_free_rate.csv` | Daily risk-free rate series |

These two files are included in the repo as they are derived/aggregated.

## config/TEMP/

The `config/TEMP/` directory expects CLCData `.ASC` files used for roll-date extraction. These are **not** included due to licensing. Contact CLCData for access.

## Quick Setup

```bash
# After obtaining CLCData files:
mkdir -p data/CLC
cp /path/to/clcdata/*_RAD.CSV data/CLC/

# Verify
python verify_setup.py
```
