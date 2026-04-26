# DRL/DQN — Mainline

This folder contains the DQN-only part of the active DRL stack.

## Current Pipeline

```bash
# Prepare shared features
python drl_shared/prepare_features.py --asset Forex

# Train one asset-class model per round
python drl/dqn/train/train_dqn_walkforward.py --asset Forex --episodes 200 --device cuda

# Resume from latest checkpoint (reproducible)
python drl/dqn/train/train_dqn_walkforward.py --asset Forex --round 1 --episodes 200 --resume --device cuda

# Unified backtest
python run_strategy_backtest.py --strategy DQN --asset Forex
```

## Shared State Space (Paper Section 3.1)

- `seq_len = 60` (past 60 observations per feature)
- `feature_dim = 7`

Features (matching paper exactly):
- feature 0: Normalized close price series (`p_t / rolling_std(p, 60)`)
- features 1-4: Vol-adjusted returns for horizons 21/42/63/252 days, formula: `(p_t - p_{t-H}) / (σ_t * √H)`
- feature 5: Averaged MACD normalized by 63-day price volatility (Eq.3)
- feature 6: RSI(30) normalized to [-1, 1]

## DQN Architecture (Paper Table 1)

- 2-layer LSTM [64, 32] with Leaky-ReLU
- Dueling DQN (value + advantage heads)
- Fixed Q-targets: hard copy every τ=1000 learn steps
- Double DQN: online net selects action, target net evaluates

Hyperparameters:
- LR = 0.0001, Adam
- γ = 0.3
- bp = 0.002
- Batch size = 64
- Memory = 5000
- ε: 0.3 → 0.05 over 50000 steps
- Episodes = 200 per training round

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
- `checkpoint_metadata.json`

## Evaluation

Eq.4 reward function:
- Additive price differences: `r_t = p_t - p_{t-1}`
- Volatility-scaled positions: `σ_tgt / σ_{t-1}`
- Transaction cost: `bp * p_{t-1} * |scaled_position_change|`
