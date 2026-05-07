# Exhibit 5 TC Impact — Learnings

## 2026-05-06: "All" Portfolio Backtest Implementation

### Paper Definition (DRL_main.pdf Section 4.3-4.4)
- "All" is NOT a separately trained 5th model
- "All" = equal-weighted portfolio of ALL contracts from all 4 asset classes
- Formula: `(1/N) Σ R_t^i` where N = total contracts
- Per-contract returns come from per-asset DQN models (Commodity=24, Equity=11, FI=4, Forex=9)

### Implementation: `drl/dqn/reports/generate_ensemble_table2.py`
- **`compute_all_portfolio(results_dict, port_vol_target=0.97)`** (line 453-512)
  - Takes 4 per-asset result dicts
  - Contract-count-weighted average: `(Σ n_asset × port_asset) / total_contracts`
  - Not equal-asset-weight: 48 contracts total, not 4 assets weighted equally
  - Reuses: `get_portfolio_bridge("constant_posthoc", 0.97)`, `compute_metrics()`
  - Returns same `{"metrics", "portfolio", "n_contracts"}` format as per-asset
- **main()** (line 603-610): after 4-asset loop, calls `compute_all_portfolio()`, adds to `all_results["All"]`
- **format_comparison_table()** (line 529): added "All" to display order
- **Metrics save**: automatically included via `all_results.items()` iteration
- **PAPER_TABLE2["All"]["DQN"]** exists in `config.py` with reference metrics

### Extensibility
- Works for ANY BP level (single `--tc-bp`, `--all-bp`, or default)
- No modification needed for different BP levels
- Code paths: single BP → `table2_metrics.json`, multi-BP → `bp{XX}/table2_metrics.json`
- If BP=20 models not yet trained → backtest gracefully fails with "No bundles found"

### Running Backtest
```bash
# After bp20 models exist:
python3 drl/dqn/reports/generate_ensemble_table2.py --tc-bp 0.0020

# For all BP levels:
python3 drl/dqn/reports/generate_ensemble_table2.py --all-bp
```

## 2026-05-07: BP-Aware Metrics Export Pipeline

### Data Structure
- Per-asset BP metrics: `ensemble_table2_bp/{asset_slug}/bp{XX}/metrics.json`
  - bp1: full metrics (E(R), std(R), Sharpe, Sortino, MDD, % +ve, Ave P/L, DD, Calmar, bp)
  - bp20+: sparse metrics (E(R), std(R), Sharpe, n_contracts only) — missing Sortino, MDD, % +ve, Ave P/L
- "All" portfolio: `ensemble_table2_bp/table2_metrics.json["All"]` (bp1 only)
  - No per-BP "All" data exists yet (no `bp{XX}/table2_metrics.json` files)
- Asset slug mapping: Commodity→Commodity, Equity Index→Equity_Index, Fixed Income→Fixed_Income, Forex→Forex

### Export Pipeline (export_bp_metrics.py)
- Auto-discovers BP levels from directory listing (currently: bp1, bp20)
- Exports 5 CSVs: exhibit5_commodity.csv, exhibit5_equity.csv, exhibit5_fi.csv, exhibit5_forex.csv, exhibit5_all.csv
- CSV columns: bp_bps, E(R), std(R), Sharpe, Sortino, MDD, % +ve, Ave P/L
- Missing values written as empty strings (sparse bp20 metrics)
- "All" data: tries per-BP table2_metrics.json first, falls back to top-level for bp1

### Figure Generator (exhibit5_tc_impact.py)
- Reads 5 CSVs from figures/data/ (pipeline: run export_bp_metrics.py first)
- Panel A: Sharpe vs BP (5 lines: Commodity, Equity Index, Fixed Income, Forex, All)
- Panel B: Cost proxy (-E(R)) vs BP (5 lines)
- Saves to PDF (exhibit5_tc_impact.pdf), serif font, seaborn-whitegrid style
- Partial data watermark when <5 BP levels available
- "All" line only appears for BP levels where "All" data exists (currently bp1 only)

### Cost Proxy
- Panel B uses -E(R) as cost proxy (negative expected return = cost of trading)
- Original code tried to load daily_costs.npz but that data doesn't exist for bp20
- Task specified: use -E(R) as cost proxy if no explicit cost metric available
