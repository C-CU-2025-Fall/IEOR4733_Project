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
- shared versioned feature preparation
- DQN training / versioned checkpoint bundles
- unified backtesting commands
- current DRL folder responsibilities

Current DRL v2 commands:

```bash
# Prepare v2 shared features
python drl_shared/prepare_features.py --ticker AN --round 1 --model-version v2

# Train one v2 DQN bundle for one contract / round
python drl/dqn/train/train_dqn_walkforward.py --ticker AN --round 1 --episodes 50 --device cpu --model-version v2

# Backtest DQN through the unified backtester; DQN only emits positions
python run_strategy_backtest.py --strategy DQN --asset Forex --model-version v2 --progress
```

DRL v2 state-space note:
- `feature 0` is now causal EWMA60 close deviation:
  - `(p_t - EMA60(p)_t) / (EWMA60(r)_t * sqrt(60))`
- old full-sample close z-score is not the active v2 spec
- old Forex checkpoints are historical `v0` compatibility artifacts unless retrained into a v2 bundle

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

The active DRL code uses **v2.1** infrastructure with STRUCTURAL_38 preset.

- training unit: one DQN model per contract per retrain round
- model artifact: versioned bundle under `drl/dqn/models/v2.1/<ticker>/r<round>/<run_id>/`
- bundle source of truth: `manifest.json` (includes preset, sigma_tgt, feature_spec)
- backtest path: bundle checkpoint -> batched position inference -> unified baseline backtester
- preset support: `--preset structural_38` propagates source_overrides + exclusions through features, training, and backtest
- parallel training: `scripts/train_v2.1_quick.py` (ProcessPoolExecutor, 8 workers)

### Key Findings (2026-04-23)

- **Long-only baseline (STRUCTURAL_38, σ=0.058)**: Trade metrics **28/28** (7/7 per asset), Total **30/36** with path metrics
- **DQN v2.1 (50ep, patience=3, σ=0.058)**:
  - Forex: 4/9 (MDD ✅ Calmar ✅ % +ve ✅ Ave P/L ✅)
  - Fixed Income: 1/9 (Ave P/L ✅ only)
  - **Root cause**: DQN selects action=0 (no position) ~69% of the time — model learns that not trading avoids TC losses
- **Feature normalization comparison**: v0 (full-sample z-score) > v2 (EWMA_std(r)*√60) > v3 (EWMA_std(p)*√252)
  - See `docs/feature0_comparison.png` for visualization
- **Version history**: v0 (Forex only) → v2 (all 50 contracts) → v2.1 (STRUCTURAL_38 preset) → v3 (experiment, worse)

### Next Steps
- Increase training episodes (200+) to let DQN explore beyond "don't trade"
- Consider reward shaping to penalize excessive inaction
- Train remaining asset classes (Equity Index, Commodity) with v2.1 + STRUCTURAL_38

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
