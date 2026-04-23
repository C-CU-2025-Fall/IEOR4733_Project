# DRL Pipeline Handoff

This note is the fastest way for a teammate to pick up the current DRL stack.

## Current Design

The repo now separates responsibilities into three layers:

1. `baseline_run.py` / `strategy_backtester.py`
- the only metric / portfolio backtest stack
- all 9 metrics come from the same simulated portfolio path
- used for `Long`, `Sign(R)`, `MACD`, and `DQN`

2. `drl_shared/`
- shared state-space and feature preparation for non-baseline models
- intended to be reused by `DQN`, later `PG`, later `A2C`

3. `drl/dqn/`
- DQN-only code
- model definition
- training loop
- checkpoint loading
- DQN position inference adapter

Important:
- there is no DQN-local metric world anymore
- there is no active reporting-path backtest lane anymore
- current DRL evaluation is always routed back into the unified baseline backtest stack

## Current Workflow

### 1) Prepare shared features

Example:

```bash
python drl_shared/prepare_features.py --asset Forex --round 1
python drl_shared/prepare_features.py --ticker AN --round 1 --model-version v2
```

Output:
- `drl/features/v2/<ticker>/r<k>.npz`

Stored fields:
- `prices`
- `returns`
- `sigma`
- `features`
- `dates`
- `source`
- round metadata
- `model_version`
- `state_spec_version`
- serialized `feature_spec`

### 2) Train one DQN model per contract

Example:

```bash
python drl/dqn/train/train_dqn_walkforward.py --ticker AN --round 1 --episodes 50 --device cpu --model-version v2
```

Inputs:
- prepared shared feature file for that ticker/round

Outputs:
- versioned model bundle:
  - `drl/dqn/models/v2/<ticker>/r<k>/<run_id>/`
- required bundle files:
  - `checkpoint.pt`
  - `manifest.json`
  - `train_config.json`
  - `feature_spec.json`
  - `episode_metrics.csv`
  - `train.log`

Compatibility note:
- the current Forex GPU checkpoints are retained under the older path:
  - `drl/dqn/models/walkforward/<ticker>_r<k>.pt`
- those Forex checkpoints are `v0` compatibility artifacts, not active v2 evidence
- those checkpoints are still per-contract models; their checkpoint format
  uses a single FC Q-head (`q` / `t`) rather than the newer dueling-head format
  (`q_net` / `target_net`)

### 3) Run unified backtests

Global baseline-owned CLI:

```bash
python run_strategy_backtest.py --strategy Long --asset Forex
python run_strategy_backtest.py --strategy MACD --asset Forex
python run_strategy_backtest.py --strategy DQN --asset Forex --model-version v2 --progress
```

DQN adapter CLI:

```bash
python drl/dqn/backtest/backtest_dqn_walkforward.py --strategy DQN --asset Forex --model-version v2 --progress
```

Important:
- `Long`, `Sign(R)`, `MACD` do not require checkpoints
- `DQN` requires checkpoints for the contracts used in the selected asset class
- if checkpoints are missing, DQN fails explicitly and does not fall back to `Long`

## Locked Defaults

- default `sigma_tgt = 0.058`
- state window: `60`
- feature dimension: `8`
- active state spec: `v2_ewma60_close_deviation`
- close-price feature:
  - `(p_t - EMA60(p)_t) / (EWMA60(r)_t * sqrt(60))`
- return-feature horizons: `21 / 42 / 63 / 252`
- return-feature vol normalization: `EWMA(60)` on additive `r_t`
- MACD feature normalization: `63`-window volatility
- retrain rounds:
  - `r1`: train through 2010, test 2011-2015
  - `r2`: train through 2015, test 2016-2019
- DQN action space:
  - discrete `{-1, 0, +1}`

## Main Files

- shared feature/state logic:
  - `drl_shared/state_space.py`
- shared feature prep CLI:
  - `drl_shared/prepare_features.py`
- unified backtest wrapper:
  - `strategy_backtester.py`
- unified backtest CLI:
  - `run_strategy_backtest.py`
- DQN model/spec:
  - `drl/dqn/model.py`
  - `drl/dqn/spec.py`
- DQN training:
  - `drl/dqn/train/train_dqn_walkforward.py`
- DQN inference adapter:
  - `drl/dqn/backtest/engine.py`

## Current Training Status

- v2 infrastructure is implemented; full v2 training is not assumed complete
- old Forex GPU walk-forward checkpoints exist for 9 contracts x 2 rounds as `v0` compatibility artifacts
- local machines without `torch` can inspect metadata but cannot run DQN inference/training
- no PG folder yet
- no A2C folder yet
- no top-level DRL experiment registry yet

## Fast Sanity Checks

```bash
python tests/run_structural_38.py --table 3
python run_strategy_backtest.py --strategy Long --asset Forex
python drl_shared/prepare_features.py --ticker AN --round 1 --model-version v2
python drl/dqn/tests/verify_shared_dqn.py --asset Forex --round 1 --ticker AN --require-prepared
python -m unittest tests.test_drl_v2
```

These cover:
- baseline structural reference
- unified backtest CLI
- shared feature generation
- DRL state-space consistency
