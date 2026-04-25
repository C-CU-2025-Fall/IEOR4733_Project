# IEOR4733_Project

Reproduction workspace for Zhang, Zohren, Roberts (2019).

The repo now has one active interpretation:

- baseline = `structural_38` long-only line
- DRL/DQN uses the same data doctrine
- one unified backtester computes all final metrics
- old versioned DQN artifact families are archive-only, not mainline

Historical search waves and abandoned branches live in:
- [PROJECT_MEMORY.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/PROJECT_MEMORY.md)

Paper:
- [arXiv PDF](https://arxiv.org/pdf/1911.10107)

## Main Commands

```bash
# Baseline references
python baseline_run.py --table 3 --all-metrics --sigma 0.058
python tests/run_structural_38.py --table 3
python tests/run_structural_38.py --table both
python tests/run_structural_38.py --table 3 --with-path-metrics

# Unified backtester
python run_strategy_backtest.py --strategy Long --asset Forex
python run_strategy_backtest.py --strategy DQN --asset Forex
python run_strategy_backtest.py --strategy DQN --asset "Fixed Income"

# DRL feature prep and training
python drl_shared/prepare_features.py --asset Forex
python drl/dqn/train/train_dqn_walkforward.py --asset Forex --episodes 50 --device cpu
python drl/dqn/train/train_dqn_walkforward.py --asset Forex --episodes 1 --device mps
python scripts/train_dqn_asset_parallel.py --round both --parallel 4 --device cuda
```

## Current Baseline

`tests/run_structural_38.py` is the authoritative reproducible baseline.

It locks:
- `STRUCTURAL_38_OVERRIDES`
- `STRUCTURAL_38_EXCLUDED`
- the current trade-world baseline interpretation

This is the baseline we actually compare against for local reproducibility. The
ideal paper/data world is useful context, but it is not the active benchmark
for this repo.

## DRL Mainline

DRL now follows the same baseline doctrine as the structural baseline:

- same `structural_38` source overrides
- same `structural_38` exclusions
- same unified backtester
- no active `v0 / v2 / v2.1 / v3` story

Current DRL paths:
- features:
  - `drl/features/<ticker>/r<round>.npz`
  - `drl/features/<asset_class>/r<round>/index.json`
- DQN bundles:
  - `drl/dqn/models/<asset_class>/r<round>/<run_id>/`

Current DQN training unit:
- one shared DQN model per asset class per retrain round
- default training covers both `r1` and `r2`
- each asset-class cycle visits every eligible contract once with a shared replay buffer
- early stopping uses chronological 90/10 train/validation monitoring with default patience `20`

DQN stabilizers retained from the paper:
- `[49]` fixed Q-targets, hard target-network copy every `1000` learn steps
- `[18]` Double DQN, online-net argmax with target-net evaluation
- `[50]` Dueling DQN, value and advantage heads

Old directories such as:
- `drl/dqn/models/walkforward/`
- `drl/dqn/models/v2.1/`

may still exist on disk, but they are archive artifacts and are not resolved by
the active default path.

## Shared State Space

Current shared state:
- `seq_len = 60`
- `feature_dim = 8`
- feature 0:
  - `(p_t - EMA60(p)_t) / EWMA60(r)_t`
- features 1-4:
  - vol-adjusted returns for `21 / 42 / 63 / 252`
- feature 5:
  - averaged MACD normalized by 63-day price volatility
- feature 6:
  - RSI(30)-style feature
- feature 7:
  - causal volatility ratio

This shared state is meant for `DQN` now, and later `PG / A2C`.

## Eq.4, Sleeve Wealth, and Unified Backtest

Three layers should not be mixed:

**Eq.4 trade-return world**
- additive price differences:
  - `r_t = p_t - p_{t-1}`
- volatility-scaled positions:
  - `sigma_tgt / sigma_t`

**Sleeve/reporting wealth world**
- comparable sleeve wealth can use:
  - `capital0 = p0 * sigma_tgt / sigma0`
- this is not Eq.4 itself

**Current unified backtest**
- computes all final portfolio metrics, including `MDD` and `Calmar`, from the
  same simulated portfolio path

## Current References

- data issues:
  - [docs/data_issues.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/docs/data_issues.md)
- suspicious paper cells:
  - [docs/paper_table_suspicious_cells.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/docs/paper_table_suspicious_cells.md)
- DRL pipeline handoff:
  - [docs/drl_pipeline.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/docs/drl_pipeline.md)
- DQN folder README:
  - [drl/dqn/README.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/drl/dqn/README.md)
- printable A4 structural summary:
  - [docs/structural38_trade_tables_paper_style_a4.png](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/docs/structural38_trade_tables_paper_style_a4.png)

## Latest Alignment Snapshot

Current active structural-38 long-only baseline:

- trade-world `<=15%`:
  - `28/28`
- unified-path total:
  - `30/36`

### Long-Only Progress Table

This is the current active progress anchor for the repo.

| Scope | Setting | Result |
| --- | --- | --- |
| Baseline | `structural_38` long-only, trade-world Table 3 | `28/28` at `<=15%` |
| Baseline + path metrics | `structural_38` long-only, unified backtest Table 3 | `30/36` at `<=15%` |
| Current interpretation | reproducible local baseline | active mainline |

Current long-only structural-38 Table 3 trade-world comparison:

| Asset | Ours | Paper | `%Err` |
| --- | --- | --- | --- |
| Commodity | `-0.263, +0.385, +0.260, -0.683, -1.009, +0.491, +0.925` | `-0.298, +0.412, +0.258, -0.723, -1.152, +0.473, +0.987` | `11.8, 6.6, 1.0, 5.5, 12.4, 3.9, 6.2` |
| Equity Index | `+0.541, +0.868, +0.682, +0.624, +0.794, +0.547, +0.920` | `+0.504, +0.928, +0.606, +0.543, +0.831, +0.541, +0.928` | `7.4, 6.5, 12.6, 14.8, 4.5, 1.2, 0.9` |
| Fixed Income | `+0.568, +0.889, +0.590, +0.639, +0.962, +0.533, +0.974` | `+0.605, +0.939, +0.561, +0.645, +1.081, +0.515, +1.048` | `6.2, 5.4, 5.2, 1.0, 11.0, 3.4, 7.0` |
| Forex | `-0.179, +0.438, +0.282, -0.409, -0.635, +0.490, +0.972` | `-0.198, +0.472, +0.285, -0.420, -0.696, +0.491, +0.966` | `9.6, 7.3, 1.0, 2.6, 8.8, 0.1, 0.6` |

Metric order:
- `E(R), std(R), DD, Sharpe, Sortino, % +ve, Ave P/L`

Current long-only structural-38 Table 3 unified-path extension:

| Asset | MDD / Calmar Ours | MDD / Calmar Paper | `%Err` |
| --- | --- | --- | --- |
| Commodity | `+0.127, -0.090` | `+0.248, -0.130` | `48.8, 30.5` |
| Equity Index | `+0.112, +0.368` | `+0.127, +0.466` | `11.5, 21.0` |
| Fixed Income | `+0.214, +0.444` | `+0.108, +0.455` | `98.0, 2.3` |
| Forex | `+0.259, -0.084` | `+0.219, -0.101` | `18.4, 17.0` |

Most suspicious remaining paper-side cells:
- `Equity Index / Table 2 / DD: 0.606`
- `Equity Index / Table 2 / Sortino: 1.102`
- `Fixed Income / Table 2 / Sortino: 1.180`

## 41/45 Status

`41/45` is retained only as an experimental upper-bound reproducer:
- [tests/run_legacy_41.py](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/tests/run_legacy_41.py)

It is not part of the active baseline or active DRL interpretation.
