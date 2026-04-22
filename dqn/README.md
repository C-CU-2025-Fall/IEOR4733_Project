# DQN Walk-Forward Trading

Deep Reinforcement Learning (DQN) implementation for futures trading with walk-forward validation.

## Directory Structure

```
dqn/
├── train/                    # Training scripts
│   ├── strategy_dqn.py       # DQN strategy implementation
│   ├── train_dqn_walkforward.py  # Walk-forward training
│   ├── prepare_dqn_walkforward.py # Data preparation
│   ├── prepare_dqn_data.py   # Full-period data prep
│   └── train_dqn_paper_aligned.py # Original paper-aligned training
│
├── models/                   # Trained models
│   ├── walkforward/          # Walk-forward models (18 total)
│   │   ├── AN_r1.pt, BN_r1.pt, ... (Round 1: 05-09→10-14)
│   │   └── AN_r2.pt, BN_r2.pt, ... (Round 2: 05-14→15-19)
│   └── full/                 # Full-period models (optional)
│
├── backtest/                 # Backtest framework
│   └── backtest_dqn_walkforward.py
│
└── docs/                     # Documentation
    └── dqn_alignment_notes.md
```

## Walk-Forward Scheme

| Round | Training Period | Test Period | Models |
|-------|----------------|-------------|--------|
| 1 | 2005-2009 (5y) | 2010-2014 (5y) | `*_r1.pt` |
| 2 | 2005-2014 (10y) | 2015-2019 (5y) | `*_r2.pt` |

## Quick Start

### Train (Forex, Round 1)
```bash
python dqn/train/train_dqn_walkforward.py --round 1 --asset Forex --episodes 50
```

### Backtest
```bash
python dqn/backtest/backtest_dqn_walkforward.py --round 1 --asset Forex
python dqn/backtest/backtest_dqn_walkforward.py --all --asset Forex
```

## Model Details

- **Architecture**: LSTM[64, 32] + Double DQN + Fixed Q-targets
- **Action Space**: Discrete {-1, 0, +1}
- **State Space**: 8-dimensional (price, returns, MACD, RSI, vol)
- **Hyperparameters**: Paper Table 1 aligned

## Current Status

- ✅ Round 1: 9/9 Forex models trained
- ✅ Round 2: 9/9 Forex models trained
- ⏳ Backtest framework: Basic (Long Only baseline)
- ⏳ DQN inference integration: Pending
