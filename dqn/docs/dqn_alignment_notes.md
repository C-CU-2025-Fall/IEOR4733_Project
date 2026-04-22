# DQN Alignment Notes

## Current paper-faithful interpretation

The retained direction is now:

- **shared state schema**
- **shared model weights per retraining round**
- discrete action space `{-1, 0, +1}`
- Eq.4-style additive reward
- LSTM `[64, 32]` + Leaky-ReLU
- fixed Q-targets
- Double DQN
- **dueling DQN**

## Locked shared state space

One authoritative 8-dimensional feature schema is now used everywhere:

1. normalized price
2. 21-day vol-adjusted return
3. 42-day vol-adjusted return
4. 63-day vol-adjusted return
5. 252-day vol-adjusted return
6. multi-scale MACD averaged across `(8,24)`, `(16,48)`, `(32,96)`
7. RSI(30)
8. volatility ratio

State window:
- `seq_len = 60`

This schema is shared across:
- data preparation
- training environments
- inference
- backtest

## Locked training interpretation

- **Action space**: discrete `{-1, 0, +1}`
- **Reward**: `reward = gross - transaction_cost`
- **Transaction cost**: same price-aware Eq.4 treatment used in the retained DQN pipeline
- **Retraining cadence**:
  - round 1: train `2005-2010`, test `2011-2015`
  - round 2: train `2005-2015`, test `2016-2019`

## What was fixed in the current refactor

- duplicated feature builders removed from active training/backtest paths
- backtest no longer silently falls back to `Long`
- checkpoint format is now shared-model-first
- shared round paths are explicit:
  - `dqn/data/shared_rounds/...`
  - `dqn/models/shared_rounds/...`

## What is intentionally not done yet

- no new GPU/shared-model training run
- no performance claims from the new shared-model path yet
- no `MDD/Calmar` integration in this DQN phase

**Last Updated**: 2026-04-22
