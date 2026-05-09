# Task 1 Learnings: Compute Real Daily Cost Per Contract

## Overview
Successfully computed real daily cost per contract for all BP levels and updated Exhibit 5 Panel B.

## Files Created
- `drl/dqn/figures/compute_daily_cost.py` - Script to compute daily costs
- `drl/dqn/figures/data/exhibit5_daily_cost_bp1.csv`
- `drl/dqn/figures/data/exhibit5_daily_cost_bp10.csv`
- `drl/dqn/figures/data/exhibit5_daily_cost_bp20.csv`
- `drl/dqn/figures/data/exhibit5_daily_cost_bp30.csv`
- `drl/dqn/figures/data/exhibit5_daily_cost_bp45.csv`
- `drl/dqn/figures/data/exhibit5_daily_cost_all.csv` (summary)

## Files Modified
- `drl/dqn/figures/data/exhibit5_bp1.csv` - Added daily_cost column
- `drl/dqn/figures/data/exhibit5_bp10.csv` - Added daily_cost column
- `drl/dqn/figures/data/exhibit5_bp20.csv` - Added daily_cost column
- `drl/dqn/figures/data/exhibit5_bp30.csv` - Added daily_cost column
- `drl/dqn/figures/data/exhibit5_bp45.csv` - Added daily_cost column
- `drl/dqn/figures/exhibit5_tc_impact.py` - Updated Panel B to use real daily cost
- `drl/dqn/figures/exhibit5_tc_impact.pdf` - Regenerated with real cost data

## Formula Used
```
daily_cost = |position_t - position_{t-1}| × bp × price_t / 10000
avg_daily_cost = mean(daily_cost) across all contracts and days
```

## Key Patterns Discovered

### 1. positions.csv Structure
- Columns: contract, round, date, position, return
- Need to group by contract and sort by date
- Position changes indicate trading activity

### 2. Price Data Structure (from baseline_run.load_contracts)
- Returns list of dictionaries with keys:
  - 'tk': ticker symbol
  - 'prices': numpy array of prices
  - 'dates': numpy array of dates
  - 'rt': returns
  - 'sigma': volatility
  - 'start', 't1': time indices
  - 'source': data source
  - 'macd_pos': MACD position

### 3. Date Matching
- positions.csv uses string dates (YYYY-MM-DD)
- load_contracts returns numpy datetime64 dates
- Convert to pandas Timestamp then to string for matching

### 4. Asset Mapping
- Directory slugs: Commodity, Equity_Index, Fixed_Income, Forex
- Display names: Commodity, Equity Index, Fixed Income, Forex
- Need mapping between directory structure and display names

## Sample Results

### BP=1 (1 basis point)
- Commodity: 0.096346
- Equity Index: 0.558259
- Fixed Income: 0.032950
- Forex: 0.541574
- All: 0.307282

### BP=45 (45 basis points)
- Commodity: 1.779902
- Equity Index: 18.799869
- Fixed Income: 1.319052
- Forex: 27.102467
- All: 12.250323

## Implementation Notes

### Challenges Encountered
1. **KeyError: 'ticker'** - load_contracts returns 'tk' not 'ticker'
2. **Large datasets** - positions.csv files have ~54k rows each
3. **Date format mismatch** - needed conversion between datetime64 and strings

### Performance Considerations
- Used dictionary lookup for (contract, date) → price mapping
- Only computed costs for days with actual turnover (position changes)
- Processed all 5 BP levels × 4 assets = 20 combinations

## Verification Steps
1. All daily cost CSVs created with correct columns
2. All exhibit5_bp{XX}.csv files updated with daily_cost column
3. exhibit5_tc_impact.py successfully generates PDF
4. Panel B now shows "Avg Daily Cost" instead of "Cost Proxy (-E(R))"

## Related Files
- positions.csv: `drl/dqn/reports/ensemble_table2_bp/{asset}/bp{BP}/positions.csv`
- Price loader: `baseline_run.py::load_contracts()`
- Manifest: `drl/dqn/figures/data/exhibit5_manifest.json`
