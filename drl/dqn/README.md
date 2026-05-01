# DRL/DQN — Mainline

This folder contains the DQN-only part of the active DRL stack.

## Current Pipeline

```bash
# Prepare shared features
python3 drl_shared/prepare_features.py --asset Forex

# Train one asset-class model per round (with dropout + early stopping)
python3 drl/dqn/train/train_dqn_walkforward.py --asset Forex --episodes 200 --device cuda

# Resume from latest checkpoint (reproducible)
python3 drl/dqn/train/train_dqn_walkforward.py --asset Forex --round 1 --episodes 200 --resume --device cuda

# Unified backtest
python3 run_strategy_backtest.py --strategy DQN --asset Forex
```

## Shared State Space (Pruned v2 — 2026-04-30)

- `seq_len = 60` (past 60 observations per feature)
- `feature_dim = 5` (= `market_feature_dim`, no prev_action channel)
- `state_spec_version = structural_38_pruned_v2_5d`

Feature pruning from 9D to 5D based on correlation analysis across Forex contracts:
- 15 pairs with |r| > 0.7 in original 9D space; effective rank ~4-5
- F0 (p_t/rolling_std) removed: non-stationary, scale [4, 342] vs others [-3, 3]
- F2 (ret_42d), F3 (ret_63d) removed: corr > 0.72 with ret_21d, redundant at gamma=0.3
- F4 (ret_252d) removed: annual momentum, gamma=0.3 can't exploit
- F7 (MACD 32,96) removed: corr 0.77 with MACD(16,48), too slow

| Index | Feature | Formula | Key Property |
|-------|---------|---------|--------------|
| 0 | ret_1d_vol_norm | `r_t / sigma_t` | Replaces F0. Near-orthogonal (max |r|=0.27, VIF=1.26). Scale ~N(0,1) |
| 1 | ret_21d_vol_norm | `(p_t - p_{t-21}) / (sigma_t * sqrt(21))` | Core short-term momentum |
| 2 | macd_8_24 | `(EMA_8 - EMA_24) / sigma_63(p)`, then `/ sigma_252(q)` | Fastest MACD trend signal |
| 3 | macd_16_48 | Same formula, spans (16, 48) | Medium MACD, ~0.6 corr with macd_8_24 |
| 4 | rsi_30 | `(RSI_30 - 50) / 50`, Wilder smoothing | Mean-reversion complement, well-scaled [-1, 1] |

## DQN Architecture (Paper Table 1 + Published JFDS 2020)

- 2-layer LSTM [64, 32] with Leaky-ReLU (slope=0.01)
- Dropout (p=0.2) after LSTM layers
- Dueling DQN (value + advantage heads)
- Fixed Q-targets: hard copy every τ=1000 learn steps
- Double DQN: online net selects action, target net evaluates

Hyperparameters:
- LR = 0.0001, Adam
- gamma = 0.3
- bp = 0.002
- Batch size = 64
- Memory = 5000
- epsilon: 0.3 -> 0.05 linear decay over `n_contracts * 25000` steps
- Episodes = 200 per training round
- Dropout = 0.2
- Validation split = 10% (chronological)
- Early stopping patience = 20 cycles

## Training Methodology

Per published paper (JFDS 2020):
- One shared model per asset class per round (anti-overfitting via cross-contract training)
- Each contract split 90/10: train on first 90%, validate on last 10% (chronological, no shuffle)
- Training episodes run to natural env termination; no active 1500-step truncation
- Validation reward evaluated every cycle with greedy policy (ε=0)
- Early stopping: if validation reward doesn't improve for 20 cycles, stop and save best checkpoint
- Resume support: full RNG state (torch/numpy/python) saved for reproducible continuation

## Artifacts

Prepared features:
- `drl/features/<ticker>/r<round>.npz`
- `drl/features/<asset_class>/r<round>/index.json`

Active DQN bundles:
- `drl/dqn/models/<asset_class>/r<round>/<run_id>/`

Bundle contents:
- `checkpoint.pt` (weights + optimizer + replay buffer + RNG state)
- `manifest.json`
- `train_config.json`
- `feature_spec.json`
- `train.log`
- `episode_metrics.csv` (per-episode: reward, loss, epsilon, learn_steps, target_updates, replay_size)
- `contract_metrics.csv`
- `checkpoint_metadata.json`

## Evaluation

Eq.4 reward function:
- Additive price differences: `r_t = p_t - p_{t-1}`
- Volatility-scaled positions: `sigma_tgt / sigma_{t-1}`
- Transaction cost: `bp * p_{t-1} * |scaled_position_change|`

## Feature Correlation Analysis (2026-04-30)

Original 9D space had severe multicollinearity:

| Metric | Value |
|--------|-------|
| High-corr pairs (|r|>0.7) | 15 |
| Max VIF | 28.2 (RSI), 18.7 (MACD 16,48) |
| Standardized PCA | 4 PCs = 90%, 5 PCs = 95% |
| Effective rank | ~4-5 out of 9 |

New F0 (ret_1d/sigma) analysis:
- Scale: mean=0.002, std=0.98 (consistent with other features)
- Max |r| with others: 0.27 (near-orthogonal)
- VIF: 1.26 (no multicollinearity)
- Adds 1 genuine independent dimension (PCA: 5 PCs/95% with F0 vs 4 PCs/95% without)
