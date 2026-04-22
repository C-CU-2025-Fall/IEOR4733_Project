# Shared-Model DQN Walk-Forward

Paper-faithful DQN infrastructure for futures trading with a shared-model,
round-based walk-forward setup.

## Directory Structure

```
dqn/
├── spec.py                   # Canonical DQN spec / rounds / paths
├── pipeline.py               # Shared state + reward construction
├── model.py                  # Dueling LSTM DQN + agent
├── train/
│   ├── strategy_dqn.py              # Runtime helpers + compatibility CLI
│   ├── train_dqn_walkforward.py     # Shared round training
│   ├── prepare_dqn_walkforward.py   # Shared round data prep
│   ├── prepare_dqn_data.py          # Compatibility wrapper
│   └── train_dqn_paper_aligned.py   # Compatibility alias
├── backtest/
│   └── backtest_dqn_walkforward.py  # Shared round inference/backtest
└── docs/
    └── dqn_alignment_notes.md
```

## Shared Round Scheme

| Round | Training Period | Test Period | Models |
|-------|----------------|-------------|--------|
| 1 | 2005-2010 | 2011-2015 | one shared checkpoint |
| 2 | 2005-2015 | 2016-2019 | one shared checkpoint |

## Quick Start

### Prepare Round Data
```bash
python dqn/train/prepare_dqn_walkforward.py --round 1 --asset Forex
```

### Train Shared Round Model
```bash
python dqn/train/train_dqn_walkforward.py --round 1 --asset Forex --episodes 50
```

### Backtest Shared Round Model
```bash
python dqn/backtest/backtest_dqn_walkforward.py --round 1 --asset Forex
```

## Model Details

- **Architecture**: LSTM[64, 32] + Leaky-ReLU + Dueling DQN + Double DQN + Fixed Q-targets
- **Action Space**: Discrete {-1, 0, +1}
- **State Space**: Shared 8-dimensional schema with 60-step windows
- **Hyperparameters**: Paper Table 1 aligned
- **Training Mode**: Shared model weights per round, not per-contract models

## Current Status

- ✅ Shared DQN spec locked
- ✅ Shared state / reward pipeline centralized
- ✅ Dueling DQN added
- ✅ Backtest wired for shared-model inference
- ⏳ No new shared-model training run yet
