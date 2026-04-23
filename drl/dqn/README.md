# DRL/DQN — Current Code Truth

This folder contains the **DQN-only** part of the current DRL stack.

The current implementation is:

- one model per contract per retrain round
- one shared state-space schema across `DQN / PG / A2C`
- one unified baseline/backtest metrics stack

This folder is **not**:

- a shared cross-contract DQN trainer
- a separate DQN metrics pipeline
- a separate Table 2 / Table 3 evaluation world

## Ownership Boundaries

- `drl_shared`
  - shared state schema
  - shared feature builder
  - Eq.4-style reward helper
  - action adapters
- `drl/dqn`
  - DQN model
  - DQN training loop
  - checkpointing
  - DQN inference adapter
  - run logging
- baseline/backtester
  - contract return simulation
  - portfolio aggregation
  - final metrics and table comparison

In other words:

- DQN produces positions
- the baseline/backtest stack evaluates those positions

## Current Pipeline

The implemented workflow is:

1. prepare shared features per contract / per round
2. train one DQN checkpoint per contract / per round
3. run inference from that contract checkpoint to positions
4. pass positions into the unified baseline backtester
5. compute final portfolio metrics there, including `MDD / Calmar`

Current entrypoints:

```bash
# Prepare shared features
python drl_shared/prepare_features.py --ticker AN --round 1 --model-version v2

# Train one contract model
python drl/dqn/train/train_dqn_walkforward.py --ticker AN --round 1 --episodes 50 --device cpu --model-version v2

# Unified baseline backtest
python run_strategy_backtest.py --strategy Long --asset Forex

# DQN adapter backtest
python drl/dqn/backtest/backtest_dqn_walkforward.py --strategy DQN --asset Forex --model-version v2 --progress
```

Important runtime notes:

- default `sigma = 0.058`
- DQN requires per-contract checkpoints
- missing checkpoints fail explicitly; there is no fallback to `Long`

## Shared State Space

The current shared state uses `seq_len = 60` and `feature_dim = 8`.

Features for active `v2`:

- feature 0: causal EWMA60 close-price deviation
  - `(p_t - EMA60(p)_t) / (EWMA60(r)_t * sqrt(60))`
- features 1-4: vol-adjusted returns for horizons `21 / 42 / 63 / 252`
- feature 5: averaged MACD feature normalized by 63-day price volatility
- feature 6: RSI(30)-style feature
- feature 7: volatility ratio feature

Locked conventions:

- return features use `EWMA(60)` sigma of additive `r_t`
- feature 0 does not use full-sample z-score
- MACD normalization uses the 63-day price-vol window
- active DQN action set is discrete `{-1, 0, +1}`
- the shared state space is intended for later `PG / A2C`, but those are not yet implemented as full sibling folders

## DQN Architecture

There is **one DQN trainer**, not three separate DQN models.

Training granularity and network head structure are separate:

- training granularity: one model per contract per round
- current retained Forex checkpoints: single-contract LSTM DQN with one FC Q-head
- new code path can train/load the dueling-head DQN architecture below

That one trainer currently includes three retained mechanisms:

- target network with periodic hard update
- Double DQN target construction
- dueling value / advantage heads

The retained architecture is:

- LSTM `[64, 32]`
- Leaky-ReLU
- target network copied every `TAU=1000` training steps
- Double DQN target construction:
  - online net selects `argmax`
  - target net evaluates that action
- dueling output heads:
  - value head
  - advantage head

The retained Forex checkpoints under `models/walkforward/` are still
single-contract checkpoints. They use a single FC Q-head checkpoint format
(`q` / `t`) rather than the newer dueling-head checkpoint format
(`q_net` / `target_net`).

## Eq.4 and Evaluation

The shared reward helper follows the current Eq.4-style trade-return convention:

- additive price differences:
  - `r_t = p_t - p_{t-1}`
- volatility-scaled positions:
  - `sigma_tgt / sigma_t`
- transaction cost with raw `p_{t-1}`

**TC uses separate vol_scales (fixed 2026-04-23)**:

At time `t`, the position change being costed is `Δ(pos_{t-1}, pos_{t-2})`:
- `pos_{t-1} = A_{t-1} × σ_tgt / σ_{t-1}` (current)
- `pos_{t-2} = A_{t-2} × σ_tgt / σ_{t-2}` (previous)
- `TC = bp × p_{t-1} × |pos_{t-1} - pos_{t-2}|`

`ContractEnv` tracks `last_sigma` to provide `σ_{t-2}` across steps.

Final portfolio evaluation does **not** happen here.

Final metrics are computed by the unified baseline/backtest stack, which owns:

- contract return simulation
- portfolio aggregation
- `E(R), std(R), DD, Sharpe, Sortino, % +ve, Ave P/L`
- `MDD, Calmar`

## Logging and Artifacts

Active v2 training writes a complete model bundle to:

`drl/dqn/models/v2/<ticker>/r<round>/<run_id>/`

Artifacts include:

- `checkpoint.pt`
- `manifest.json`
- `train_config.json`
- `feature_spec.json`
- `train.log`
- `episode_metrics.csv`
- `checkpoint_metadata.json`

Expected locations:

- prepared features:
  - `drl/features/v2/<ticker>/r<k>.npz`
- DQN checkpoints:
  - `drl/dqn/models/v2/<ticker>/r<k>/<run_id>/checkpoint.pt`
- retained Forex GPU checkpoint compatibility path:
  - `drl/dqn/models/walkforward/<ticker>_r<k>.pt`

The code resolves v2 bundles for active training/backtests. The old
`models/walkforward` files are historical `v0` compatibility artifacts and
should not be mixed with v2 feature results.

`manifest.json` is the source of truth for:

- model version
- state spec version
- feature formulas
- `sigma_tgt`
- transaction-cost convention
- action space
- architecture and DQN hyperparameters
- ticker / source / retrain round dates

## Status

What is true today:

- the DRL infrastructure is implemented
- the state-space / training / inference / backtest interfaces are wired
- actual prepared features and checkpoints depend on the local or server environment
- active v2 requires retraining before v2 DQN results should be interpreted

This README does **not** claim:

- that all features are already prepared
- that all checkpoints already exist
- that there is a shared cross-contract DQN training run in the current codebase

## Guardrails

This README does **not** describe:

- the older shared-model DQN interpretation
- the historical reporting-path metric world
- a DQN-owned table evaluator

For current evaluation, the only metrics world is the unified baseline/backtest stack.
