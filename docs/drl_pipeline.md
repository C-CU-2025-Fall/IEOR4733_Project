# DRL Pipeline Handoff

This note is the fastest way for a teammate to pick up the current DRL stack.

## Current Design

The repo now has one active DRL interpretation:

1. baseline/backtester
- `baseline_run.py`
- `strategy_backtester.py`
- the only portfolio/metric stack

2. shared DRL state/features
- `drl_shared/state_space.py`
- `drl_shared/prepare_features.py`

3. DQN-only code
- `drl/dqn/model.py`
- `drl/dqn/train/train_dqn_walkforward.py`
- `drl/dqn/backtest/engine.py`

Important:
- DQN does not own a separate metrics world
- DQN uses the same `structural_38` data doctrine as the baseline
- there is no active versioned DRL mainline anymore

## Workflow

### 1. Prepare shared features

```bash
python drl_shared/prepare_features.py --asset Forex --round 1
python drl_shared/prepare_features.py --ticker AN --round 1
```

Output:
- `drl/features/<ticker>/r<round>.npz`

### 2. Train one DQN model per contract

```bash
python drl/dqn/train/train_dqn_walkforward.py --ticker AN --round 1 --episodes 50 --device cpu
```

Output:
- `drl/dqn/models/<ticker>/r<round>/<run_id>/`

### 3. Run unified backtests

```bash
python run_strategy_backtest.py --strategy Long --asset Forex
python run_strategy_backtest.py --strategy DQN --asset Forex
python run_strategy_backtest.py --strategy DQN --asset "Fixed Income"
```

## Locked Defaults

- default `sigma_tgt = 0.058`
- state window: `60`
- feature dimension: `8`
- close-price feature:
  - `(p_t - EMA60(p)_t) / (EWMA60(r)_t * sqrt(60))`
- return horizons:
  - `21 / 42 / 63 / 252`
- return-feature vol normalization:
  - `EWMA(60)` on additive `r_t`
- MACD normalization:
  - 63-day price volatility
- retrain rounds:
  - `r1`: train through 2010, test 2011-2015
  - `r2`: train through 2015, test 2016-2019
- DQN action space:
  - `{-1, 0, +1}`

## Active Artifact Layout

- features:
  - `drl/features/<ticker>/r<round>.npz`
- DQN bundles:
  - `drl/dqn/models/<ticker>/r<round>/<run_id>/`

Old directories like `drl/dqn/models/walkforward/` and `drl/dqn/models/v2.1/`
are archive artifacts only.

## Fast Sanity Checks

```bash
python tests/run_structural_38.py --table 3
python tests/run_structural_38.py --table 3 --with-path-metrics
python run_strategy_backtest.py --strategy Long --asset Forex
python run_strategy_backtest.py --strategy DQN --asset Forex
python -m unittest tests.test_drl_v2
```
