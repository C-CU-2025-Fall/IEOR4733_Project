# Exhibit 4 - Task 2 Learnings

## Date: 2025-05-09

## What was built
Created `exhibit4_per_contract_sharpe.py` - a two-panel figure showing per-contract performance metrics by transaction cost level.

### Panel A: Per-Contract Annualized Sharpe Ratios
- Boxplots showing distribution of Sharpe ratios across contracts
- Grouped by asset class (Commodity, Equity Index, Fixed Income, Forex)
- Color-coded by BP level (1, 10, 20, 30, 45 basis points)

### Panel B: Trade Return per Turnover
- Boxplots showing trading efficiency metric
- Formula: sum(returns) / sum(|position_t - position_{t-1}|)
- Same grouping and coloring as Panel A

## Key Findings from Data

### Commodity (24 contracts)
- BP=1: Mean Sharpe -0.045, Mean TR/Turnover -0.000670
- Performance degrades with higher transaction costs
- BP=45 shows slight recovery in Sharpe but continued negative TR/Turnover

### Equity Index (11 contracts)
- BP=1: Mean Sharpe 0.048 (only positive at low TC)
- BP=45: Mean Sharpe -0.782 (significant degradation)
- Most sensitive to transaction costs among all asset classes

### Fixed Income (4 contracts)
- BP=1: Mean Sharpe 0.207 (best performing at low TC)
- Most stable across BP levels
- BP=45: Mean Sharpe -0.016 (still near zero)

### Forex (9 contracts)
- BP=1: Mean Sharpe -0.110
- BP=20: Worst at -0.691 Sharpe
- High variability in TR/Turnover metric

## Data Source Structure
- Source: `drl/dqn/reports/ensemble_table2_bp/{asset}/bp{BP}/positions.csv`
- Columns: contract, round, date, position, return
- Each file is already aggregated across top-5 seeds per BP level
- 4 assets: Commodity (48 contracts in universe, 24 in sample), Equity_Index (22/11), Fixed_Income (8/4), Forex (18/9)

## Implementation Details

### Trade Return/Turnover Calculation
```python
total_return = np.sum(returns)
position_changes = np.abs(np.diff(positions))
total_turnover = np.sum(position_changes)
trade_return_per_turnover = total_return / total_turnover
```

### Sharpe Ratio Calculation
```python
annualized_sharpe = mean_daily * 252 / (std_daily * np.sqrt(252))
```

### Visualization Approach
- Vertical two-panel layout (shares x-axis pattern)
- Grouped boxplots: 5 BP levels side-by-side per asset
- Consistent color scheme across panels
- Legend in top-right of Panel A
- Grid lines for easier value reading

## Technical Notes
- Used matplotlib's boxplot with custom styling
- Set alpha=0.7 for box transparency
- Included outlier markers (small circles)
- 300 DPI output for publication quality

## Files Generated
- `drl/dqn/figures/exhibit4_per_contract_sharpe.png` (319KB)
- `drl/dqn/figures/exhibit4_per_contract_sharpe.pdf` (35KB)

## Lessons Learned
1. The positions.csv files are already ensemble-aggregated (no need to process individual seeds)
2. Fixed Income shows the most resilience to transaction costs
3. Equity Index is most sensitive to transaction costs
4. Trade Return/Turnover metric shows high variance within asset classes
5. Lower BP levels generally correlate with better performance (as expected)
