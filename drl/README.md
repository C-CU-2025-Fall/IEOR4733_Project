# DRL DQN Module — Architecture

## Unified Data Pipeline (2026-04-27 refactor)

### Feature Generation (one-time)
```
python drl_shared/prepare_features.py --asset Forex --all-rounds
```
- Generates `drl/features/{TICKER}/r{N}.npz` covering **burn-in + train + test**
- Each npz contains: prices, returns, sigma, features, dates + metadata
- Metadata includes: train_start, train_end, test_start, test_end

### Shared Data Loader (`drl_shared/data_loader.py`)
Single source of truth for loading features:
- **`load_npz(ticker, round)`** — raw load with validation (NaN/Inf/duplicates/monotonic/dates)
- **`get_train_slice(ticker, round)`** — train period, 90/10 split, returns (contract, train_slice, val_slice, meta)
- **`get_test_slice(ticker, round)`** — test period, returns (contract, start_idx, meta)

All slicing is date-based. Raises on mismatch.

### Training
- **Per-contract**: `drl/dqn/train/_train_single_contract.py` — uses `get_train_slice()`
- **Asset-class (walkforward)**: `drl/dqn/train/train_dqn_walkforward.py` — shared agent across contracts
- Both use same data loader, same features, same reward function

### Backtest
- **Per-contract test period**: `scripts/backtest_test_period.py` — uses `get_test_slice()`
- **Per-contract all data**: `scripts/backtest_per_contract.py` — full episode
- **Asset-class**: `drl/dqn/backtest/engine.py` — portfolio level

### Unit Tests
- `drl/dqn/tests/test_data_loader.py` — validates all Forex contracts:
  - Train/test no overlap
  - Correct date ranges
  - No NaN/Inf, no duplicates, monotonic dates
  - Feature dim = FEATURE_DIM
  - Non-zero returns/sigma
  - WARMUP offset correct

### Key Parameters (from spec)
- FEATURE_DIM = 9 (not 7, not 38)
- SIGMA_TGT = 0.058
- WARMUP = 252
- SEQ_LEN = 60
- EARLY_STOPPING_PATIENCE = 20
- Feature preset: structural_38 (name, not dimension)

### Rounds
| Round | Train | Test |
|-------|-------|------|
| r1 | 2005-01-01 ~ 2010-12-31 | 2011-01-01 ~ 2015-12-31 |
| r2 | 2005-01-01 ~ 2015-12-31 | 2016-01-01 ~ 2019-12-31 |

### File Structure
```
drl/
├── dqn/
│   ├── backtest/
│   │   ├── backtest_dqn_walkforward.py  # CLI for asset-class backtest
│   │   └── engine.py                    # Portfolio-level DQN backtest
│   ├── model.py                         # DQNAgent (LSTM + DQN)
│   ├── spec.py                          # Hyperparameters, paths
│   ├── train/
│   │   ├── _train_single_contract.py    # Per-contract multi-seed training
│   │   ├── train_dqn_walkforward.py     # Asset-class walkforward training
│   │   └── train_pipeline.py            # Batch training pipeline
│   └── tests/
│       ├── test_data_loader.py          # Unit tests for data loader
│       └── verify_shared_dqn.py         # Verification script
├── features/{TICKER}/r{N}.npz          # Pre-computed features (train+test)
└── models/{TICKER}/r{N}/               # Trained models
    ├── best_seed.json                   # Best seed info
    └── per_{timestamp}_s{seed}/         # Individual seed models

drl_shared/
├── data_loader.py                       # Unified feature loading + validation
├── prepare_features.py                  # Feature generation (one-time)
├── spec.py                              # Feature spec, rounds, universe
└── state_space.py                       # ContractArrays, ContractEnv, reward

scripts/
├── backtest_test_period.py              # DQN vs Long on test period
└── backtest_per_contract.py             # Per-contract backtest

archive/
├── deprecated/                          # Old/deprecated code
└── old_scripts/                         # Historical training scripts
```
