# Shared DQN Validation

This note records the minimum verification flow for the shared-model DQN pipeline.

## Validation Steps

### 1. Verify state schema without prepared files
Check that the canonical state builder works directly from raw futures data.

```bash
python dqn/tests/verify_shared_dqn.py --asset Forex --round 1 --ticker AN
```

Expected:
- `feature_shape=(T, 8)`
- `window_shape=(60, 8)`
- finite feature values
- valid backtest contract payloads

### 2. Verify prepared round data after data prep
Run after:

```bash
python dqn/train/prepare_dqn_walkforward.py --round 1 --asset Forex
```

Then verify:

```bash
python dqn/tests/verify_shared_dqn.py --asset Forex --round 1 --ticker AN --require-prepared
```

Expected:
- saved `.npz` exists under `dqn/data/shared_rounds/forex/r1/`
- saved `prices`, `returns`, `sigma`, and `features` match recomputed arrays

### 3. Verify checkpoint presence after training
Run after training creates a shared round checkpoint.

```bash
python dqn/tests/verify_shared_dqn.py --asset Forex --round 1 --ticker AN --require-prepared --require-checkpoint
```

Expected:
- checkpoint exists under `dqn/models/shared_rounds/forex/r1.pt`

### 4. Verify backtest uses shared checkpoint
Run:

```bash
python dqn/backtest/backtest_dqn_walkforward.py --round 1 --asset Forex
```

Expected:
- the script does **not** fall back to `Long`
- if checkpoint is missing, it fails explicitly
- if checkpoint exists, it loads shared-model inference and prints trade-world metrics

## Validation Coverage

The verification script checks:
- canonical 8-feature state generation
- 60-step state windows
- raw backtest input payload shape
- prepared round file consistency
- checkpoint presence for shared-model runs
