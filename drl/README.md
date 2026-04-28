# DRL DQN Module — Architecture

## Overview

Two training modes, one unified backtest pipeline:

| Mode | Training | Model Storage |
|------|----------|---------------|
| **Per-contract** | Independent model per ticker | `drl/dqn/models/{TICKER}/r{N}/` |
| **Asset-class** | Shared model across asset class | `drl/dqn/models/{AssetClass}/r{N}/` |

Both use the same feature pipeline (dim=9), same reward function (Eq.4), same preset.

## Unified Backtest Pipeline

**One entry point:** `drl/dqn/backtest/engine.py` → `portfolio_metrics()`

```
portfolio_metrics(asset, strategy, round_num, training_mode, ensemble_mode, sigma_tgt)
    ↓
backtest_strategy_metrics() → compute_strategy_metrics()
    ↓
compute_portfolio_returns_from_position_provider()
    ↓
metrics.compute_metrics() → 9 metrics from single R_port
```

All strategies (Long/Sign(R)/MACD/DQN) share:
- Same preset (`current_source_policy()` → source_overrides + excluded_contracts)
- Same `compute_metrics()` (9 metrics: E(R), std, DD, Sharpe, Sortino, MDD, Calmar, %+ve, AveP/L)
- Same `sigma_tgt = 0.058`

### Ensemble Modes
- **best**: top-1 seed by val_reward
- **top3**: top-3 seeds, average Q-values then argmax

### Cross-Round Stitching
- `round_num=None` → stitched (r1 test 2011-2015 + r2 test 2016-2019)
- `round_num=1` or `2` → single round

### Multi-Process Backtest
- `max_workers=N` parameter threads through entire call chain
- Per-contract position inference parallelized via `ProcessPoolExecutor`

## Feature Generation (one-time)

```bash
python drl_shared/prepare_features.py --asset Forex --all-rounds
```
- Generates `drl/features/{TICKER}/r{N}.npz` covering burn-in + train + test
- dim=9: price_norm + 4(returns) + 3(MACD pairs) + 1(RSI)

## Training

### Per-contract
```bash
python -m drl.dqn.train._train_single_contract --ticker AN --round 1 --seeds 5
```

### Asset-class (walkforward)
```bash
python -m drl.dqn.train.train_dqn_walkforward --asset Forex --round 1 --seed 0
```

### Best Seed Selection
After training, `best_seed.json` is auto-generated:
```json
{"best_model_dir": "...", "best_val_reward": 1.17, "all_seeds": [...]}
```

## Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| FEATURE_DIM | 9 | `drl_shared/spec.py` |
| SIGMA_TGT | 0.058 | `drl_shared/spec.py` |
| WARMUP | 252 | `drl_shared/state_space.py` |
| SEQ_LEN | 60 | `drl/dqn/spec.py` |
| LSTM_HIDDEN | [256, 128] | `drl/dqn/spec.py` |
| Actions | {-1, -0.5, 0, 0.5, 1} | `drl/dqn/spec.py` |
| Early stop patience | 5 | `drl/dqn/spec.py` |

## Rounds

| Round | Train | Test |
|-------|-------|------|
| r1 | 2005-01-01 ~ 2010-12-31 | 2011-01-01 ~ 2015-12-31 |
| r2 | 2005-01-01 ~ 2015-12-31 | 2016-01-01 ~ 2019-12-31 |

## Tests

```bash
python -m pytest drl/dqn/tests/test_backtest_engine.py -v   # 46 tests (39 pass, 7 skip)
python -m pytest drl/dqn/tests/test_data_loader.py -v       # Data validation
```

## File Structure

```
drl/
├── dqn/
│   ├── backtest/
│   │   ├── engine.py                    # ⭐ Unified backtest entry: load_agents, portfolio_metrics
│   │   └── backtest_dqn_walkforward.py  # CLI wrapper
│   ├── model.py                         # DQNAgent (Dueling LSTM-DQN)
│   ├── spec.py                          # Hyperparams, MODEL_ROOT, resolve_model_bundle
│   ├── train/
│   │   ├── _train_single_contract.py    # Per-contract multi-seed training
│   │   ├── train_dqn_walkforward.py     # Asset-class walkforward training
│   │   └── train_pipeline.py            # Batch training pipeline
│   └── tests/
│       ├── test_backtest_engine.py      # Unified backtest tests (46 cases)
│       └── test_data_loader.py          # Data loader validation
├── features/{TICKER}/r{N}.npz           # Pre-computed features (dim=9)
└── models/
    ├── {TICKER}/r{N}/                   # Per-contract models
    │   ├── best_seed.json
    │   └── per_{timestamp}_s{seed}/
    └── {AssetClass}/r{N}/               # Asset-class models
        ├── best_seed.json
        └── {timestamp}_s{seed}/

drl_shared/
├── data_loader.py                       # Unified feature loading + validation
├── prepare_features.py                  # Feature generation
├── spec.py                              # FEATURE_DIM=9, preset, rounds, universe
└── state_space.py                       # ContractArrays, ContractEnv, reward

scripts/
├── run_table3_comparison.py             # ⭐ One-click Table 3 comparison
└── backtest_test_period.py              # Legacy per-contract test

baseline_run.py                          # Core backtest: Eq.4 returns, portfolio metrics
strategy_backtester.py                   # Thin wrapper, passes max_workers
metrics.py                               # compute_metrics() — 9 metrics from R_port
config.py                                # ASSET_CLASSES, PAPER_TABLE3
```
