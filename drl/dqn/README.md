# DRL/DQN (Single-Contract) + Shared Global State Space

This folder now follows:

- one model per contract
- shared state-space schema across `DQN / PG / A2C`
- Table 3 only for evaluation
- baseline-owned backtesting metrics for `Long`, `Sign(R)`, `MACD`, and `DQN`

## Directory Structure

```text
drl/dqn/
├── spec.py
├── pipeline.py  # compatibility re-export to drl_shared/state_space.py
├── model.py
├── logging_utils.py
├── train/
│   ├── prepare_dqn_walkforward.py
│   ├── prepare_dqn_data.py
│   ├── train_dqn_walkforward.py
│   ├── train_dqn_paper_aligned.py
│   └── strategy_dqn.py
├── backtest/
│   ├── engine.py
│   └── backtest_dqn_walkforward.py
├── tests/
│   └── verify_shared_dqn.py
└── docs/
    ├── dqn_alignment_notes.md
    └── shared_dqn_validation.md
```

## Locked Spec

- `seq_len = 60`
- `feature_dim = 8`
- volatility convention: return features use `EWMA(60)` on additive `r_t`
- MACD feature uses 63-window volatility normalization
- return-feature horizons: `21 / 42 / 63 / 252`, scaled by `sigma_t * sqrt(H)`
- retrain rounds:
  - `r1`: train `2005-2010`, test `2011-2015`
  - `r2`: train `2005-2015`, test `2016-2019`
- action adapters:
  - `DQN/PG`: discrete `{-1,0,1}`
  - `A2C`: continuous `[-1,1]`

## Quick Start

### 1) Prepare global shared features
```bash
python drl_shared/prepare_features.py --ticker AN --round 1
```

### 2) Train one contract model
```bash
python drl/dqn/train/train_dqn_walkforward.py --ticker AN --round 1 --episodes 50
```

### 3) Backtest baseline strategies on Table 3 (via baseline stack)
```bash
python drl/dqn/backtest/backtest_dqn_walkforward.py --strategy Long --asset Forex
python drl/dqn/backtest/backtest_dqn_walkforward.py --strategy MACD --asset Forex
```

### 4) Backtest DQN on Table 3 (DQN adapter -> baseline stack)
```bash
python drl/dqn/backtest/backtest_dqn_walkforward.py --strategy DQN --asset Forex
```

## Logging

Each training run writes to:

`logs/rl/<algo>/<ticker>/<round>/<run_id>/`

Artifacts include:
- `train.log`
- `config.json`
- `episode_metrics.csv`
- `checkpoint_metadata.json`
