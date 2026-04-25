# DRL/DQN — Mainline

This folder contains the DQN-only part of the active DRL stack.

The mainline is now intentionally simple:

- one model per asset class per retrain round
- one shared state-space schema
- one unified baseline/backtest stack
- no active version ladder

## Ownership Boundaries

- `drl_shared`
  - shared state schema
  - shared feature builder
  - Eq.4-style reward helper
- `drl/dqn`
  - DQN model
  - DQN training loop
  - checkpointing
  - inference adapter
  - run logging
- baseline/backtester
  - contract return simulation
  - portfolio aggregation
  - final metrics and paper comparison

DQN produces positions. The baseline/backtest stack evaluates them.

## Current Pipeline

```bash
# Prepare shared features
python drl_shared/prepare_features.py --asset Forex

# Train one asset-class model per round; default round is both r1 and r2
python drl/dqn/train/train_dqn_walkforward.py --asset Forex --episodes 50 --device cpu

# Unified backtest
python run_strategy_backtest.py --strategy DQN --asset Forex
python run_strategy_backtest.py --strategy DQN --asset "Fixed Income"
```

Important runtime notes:
- default `sigma_tgt = 0.058`
- DQN uses the same `structural_38` source policy as the baseline
- missing checkpoints fail explicitly
- the active path does not fall back to archived model directories

## Shared State Space

The shared state uses:
- `seq_len = 60`
- `feature_dim = 8`

Features:
- feature 0:
  - `(p_t - EMA60(p)_t) / (EWMA60(r)_t * sqrt(60))`
- features 1-4:
  - vol-adjusted returns for `21 / 42 / 63 / 252`
- feature 5:
  - averaged MACD normalized by 63-day price volatility
- feature 6:
  - RSI(30)-style feature
- feature 7:
  - causal volatility ratio

Locked conventions:
- return features use `EWMA(60)` sigma of additive `r_t`
- MACD normalization uses the 63-day price-vol window
- DQN action set is discrete `{-1, 0, +1}`

## DQN Architecture

There is one DQN trainer, not multiple DQN families.

Current retained architecture:
- LSTM `[64, 32]`
- Leaky-ReLU
- `[49]` fixed Q-targets with hard target-network copy every `1000` learn steps
- `[18]` Double DQN target construction
- `[50]` dueling value / advantage heads

Training scheme:
- one shared `DQNAgent` per asset class and retrain round
- one `ContractEnv` per eligible contract
- shuffled round-robin cycles over contracts
- shared replay buffer within the asset class
- chronological 90/10 train/validation split
- early stopping default patience is `20` validation cycles

## Artifacts

Prepared features:
- `drl/features/<ticker>/r<round>.npz`
- `drl/features/<asset_class>/r<round>/index.json`

Active DQN bundles:
- `drl/dqn/models/<asset_class>/r<round>/<run_id>/`

Bundle contents:
- `checkpoint.pt`
- `manifest.json`
- `train_config.json`
- `feature_spec.json`
- `train.log`
- `episode_metrics.csv`
- `contract_metrics.csv`
- `validation_metrics.csv`
- `checkpoint_metadata.json`

Archived directories such as `models/walkforward/` and `models/v2.1/` may still
exist on disk, but they are not part of the active default path.

## Evaluation

Eq.4-style reward/inference uses:
- additive price differences:
  - `r_t = p_t - p_{t-1}`
- volatility-scaled positions
- transaction cost in the same additive-price world

Final portfolio evaluation does not happen here.

Final metrics come from the unified baseline/backtest stack:
- `E(R), std(R), DD, Sharpe, Sortino, % +ve, Ave P/L`
- `MDD, Calmar`

## Guardrail

This README does not describe:
- an old shared cross-contract DQN interpretation
- an old reporting-path metric world
- a DQN-owned table evaluator

The active repo has one baseline and one metrics world.
