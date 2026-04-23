# IEOR4733_Project — Current Active Line

Reproduction workspace for Zhang, Zohren, Roberts (2019), now reduced to the
current active line:

- one current runtime baseline
- one trade-world structural reference
- one retained experimental `41/45` upper-bound reproducer
- one data-issues note
- one suspicious-paper-cells note

Historical search waves and abandoned exploration branches are intentionally no
longer narrated here. They remain in:
- [PROJECT_MEMORY.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/PROJECT_MEMORY.md)

Paper:
- [arXiv PDF](https://arxiv.org/pdf/1911.10107)

## Main Commands

```bash
pip install numpy pandas yfinance

# Current runtime baseline
python baseline_run.py --table 3 --all-metrics --sigma 0.058

# Current trade-world structural reference
python tests/run_structural_38.py --table 3
python tests/run_structural_38.py --table both
python tests/run_structural_38.py --table 3 --with-path-metrics

# Retained experimental upper-bound reproducer
python tests/run_legacy_41.py
```

## Current References

### Data Issues
- [docs/data_issues.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/docs/data_issues.md)

This is the main current note for:
- problematic contract batches
- the modified structural-38 working variant
- trade-world-only alignment for the four asset classes

### Suspicious Paper Cells
- [docs/paper_table_suspicious_cells.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/docs/paper_table_suspicious_cells.md)

Current highest-priority suspicious paper cells / transitions:
- `Equity Index / Long / DD: 0.606 -> 0.606`
- `Fixed Income / Long / MDD: 0.108 -> 0.061`
- `All / Long` risk-adjusted metrics flipping sign

### DRL Pipeline
- [docs/drl_pipeline.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/docs/drl_pipeline.md)
- [drl/dqn/README.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/drl/dqn/README.md)

Use this as the teammate handoff note for:
- shared feature preparation
- DQN training / checkpoints / logs
- unified backtesting commands
- current DRL folder responsibilities

## Eq.4, Sleeve Wealth, and Unified Backtest

There are three different layers in this repo, and they should not be mixed up.

**Eq.4 trade-return world**

- uses additive price differences:
  - `r_t = p_t - p_{t-1}`
- positions are volatility-scaled by:
  - `sigma_tgt / sigma_t`
- this is the contract-level return definition used in the active baseline stack

**Sleeve/reporting wealth world**

- when a sleeve-style comparable wealth path is built, initial sleeve capital is normalized as:
  - `capital0 = p0 * sigma_tgt / sigma0`
- this normalization is **not** part of Eq.4 itself
- it belongs to the sleeve/reporting wealth interpretation layer

**Current unified backtest**

- computes all final portfolio metrics, including `MDD` and `Calmar`, from the same simulated portfolio path
- it is not the old split reporting-path diagnostic world

Important warning:

- old discussions sometimes mixed:
  - `p0` normalization
  - sleeve `capital0` normalization
  - current unified-path `MDD / Calmar`
- these are not the same thing
- the current active baseline/backtest path uses the Eq.4 trade-return layer for contract returns and the unified portfolio path for final metrics

### Printable Structural-38 Summary
- [docs/structural38_trade_tables_paper_style_a4.png](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/docs/structural38_trade_tables_paper_style_a4.png)

This A4 asset shows:
- Table 3 and Table 2
- four asset classes only
- trade-world metrics only
- no `MDD`
- no `Calmar`

Preview:

![Structural-38 trade-world paper-style A4](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/docs/structural38_trade_tables_paper_style_a4.png)

## DQN Walk-Forward Status

Per-contract independent DQN models (LSTM[64,32] + Dueling Double DQN).

**Training**: 9 Forex contracts × 2 rounds = 18 models (50ep, patience=3)

**TC Fix (2026-04-23)**: Transaction cost now uses separate vol_scales:
- Current position: `σ_tgt / σ_{t-1}`
- Previous position: `σ_tgt / σ_{t-2}`
- Previously both used the same `σ_{t-1}`, causing TC overestimation
- Fix in `drl_shared/state_space.py:compute_eq4_reward()`

**Early Stopping**: Added per-episode check with patience param (`--early-stop 3`)

**Forex DQN Walk-Forward Results (2011-2019, σ_tgt=0.058)**:

| Metric | DQN | Paper | Error | ≤15% |
|--------|-----|-------|-------|------|
| E(R) | -0.397 | -0.198 | 100.5% | ❌ |
| std(R) | 0.435 | 0.472 | 7.8% | ✅ |
| DD | 0.306 | 0.285 | 7.4% | ✅ |
| Sharpe | -0.913 | -0.420 | 117.4% | ❌ |
| Sortino | -1.298 | -0.696 | 86.5% | ❌ |
| MDD | 0.435 | 0.219 | 98.6% | ❌ |
| Calmar | -0.125 | -0.101 | 23.8% | ❌ |
| % +ve | 0.478 | 0.491 | 2.6% | ✅ |
| Ave P/L | 0.932 | 0.966 | 3.5% | ✅ |

**4/9 metrics within 15%**. std(R), DD, % +ve, Ave P/L aligned. E(R) still too negative → cascading into Sharpe/Sortino/MDD.

## Latest Alignment Snapshot

Current retained trade-world snapshot:

- source overrides:
  - `CC: RAD_REGEN`
  - `DT: RAD`
  - `JO: RAD_REGEN`
  - `LB: RAD`
  - `ZH: RAD_REGEN`
- excluded:
  - `FB`
  - `ZA`

Trade-world metrics only:
- `E(R), std(R), DD, Sharpe, Sortino, % +ve, Ave P/L`
- no `MDD`
- no `Calmar`
- no `All`

**Table 3**

| Asset | Contracts | `<=10 / 7` | `<=15 / 7` |
| --- | ---: | ---: | ---: |
| Commodity | 24 | 5 | 7 |
| Equity Index | 11 | 5 | 7 |
| Fixed Income | 4 | 6 | 7 |
| Forex | 9 | 7 | 7 |

- Total:
  - `<=10: 23/28`
  - `<=15: 28/28`

**Table 2**  
`port_vol_target=0.97`, `bridge=rolling252_lagged`

| Asset | Contracts | `<=10 / 7` | `<=15 / 7` |
| --- | ---: | ---: | ---: |
| Commodity | 24 | 7 | 7 |
| Equity Index | 11 | 5 | 5 |
| Fixed Income | 4 | 5 | 6 |
| Forex | 9 | 7 | 7 |

- Total:
  - `<=10: 24/28`
  - `<=15: 25/28`

Most suspicious remaining paper-side cells in this snapshot:
- `Equity Index / Table 2 / DD: 0.606`
- `Equity Index / Table 2 / Sortino: 1.102`
- `Fixed Income / Table 2 / Sortino: 1.180`

## Structural-38 Trade-World Reference

`tests/run_structural_38.py` is the preferred current trade-world reference.

It intentionally prints only:
- `E(R)`
- `std(R)`
- `DD`
- `Sharpe`
- `Sortino`
- `% +ve`
- `Ave P/L`

It intentionally does **not** print:
- `MDD`
- `Calmar`

This keeps the structural reference focused on the current trade-world
comparison, without mixing in the unresolved reporting-world disputes.

If you need the same structural-38 data/source preset but want to inspect
portfolio-path `MDD` / `Calmar` from the current unified backtest stack, use:
- `python tests/run_structural_38.py --table 3 --with-path-metrics`

Important distinction:
- old bridge / reporting-path experiments could compute `MDD` / `Calmar` on a
  separate reporting-world path
- current unified backtest computes all 9 metrics from the same simulated
  portfolio path
- therefore `--with-path-metrics` is **not** a resurrection of the old
  reporting-path diagnostic world; it is the current one-path backtest view

## 41/45 Retained Status

`41/45` is retained only as an **experimental upper-bound reproducer**.

Keep/use:
- [tests/run_legacy_41.py](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/tests/run_legacy_41.py)
- [frontier_presets.py](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/frontier_presets.py)

Do not read it as the current promoted mainline interpretation.

## Attribution

The 50-contract attribution base is intentionally preserved in compressed form:

- runner:
  - [tests/er_attribution_analysis.py](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/tests/er_attribution_analysis.py)
- machine-readable base:
  - [docs/contract_version_matrix_master.csv](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/docs/contract_version_matrix_master.csv)

Older attribution markdown outputs remain in the repo as archive-level material,
but are no longer part of the main README narrative.

## Notes

- `README` is now minimal operational documentation.
- Backtesting/metrics are baseline-owned (`baseline_run.py`) for all strategies.
- DRL shared modules live in `drl_shared/`; DQN-only code lives in `drl/dqn/`.
- Shared feature prep command: `python drl_shared/prepare_features.py --asset Forex --round 1`
- Global strategy backtest command: `python run_strategy_backtest.py --strategy Long --asset Forex`
- DQN adapter backtest command: `python drl/dqn/backtest/backtest_dqn_walkforward.py --strategy Long --asset Forex`
- `run_strategy_backtest.py` defaults to no exclusions unless you pass `--exclude-contracts`
- teammate handoff note: [docs/drl_pipeline.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/docs/drl_pipeline.md)
- Historical search logic, abandoned directions, and legacy score narratives live
  only in [PROJECT_MEMORY.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/PROJECT_MEMORY.md).
