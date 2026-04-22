# DQN Validation (Single-Contract + Shared State Space)

This note records the minimum non-training validation flow.

## 1) Verify state schema and backtest long metrics

```bash
python drl/dqn/tests/verify_shared_dqn.py --asset Forex --round 1 --ticker AN
```

Checks:
- state feature shape is `(N, 8)`
- one sampled window shape is `(60, 8)`
- features are finite
- Table 3 long-only backtest returns include `MDD` and `Calmar`

## 2) Verify prepared contract data

```bash
python drl_shared/prepare_features.py --ticker AN --round 1
python drl/dqn/tests/verify_shared_dqn.py --asset Forex --round 1 --ticker AN --require-prepared
```

Checks:
- prepared `.npz` exists at `drl/features/contract_rounds/AN/r1.npz`
- `prices/returns/sigma/features/source` match recomputed values

## 3) Verify checkpoint existence (after training)

```bash
python drl/dqn/tests/verify_shared_dqn.py --asset Forex --round 1 --ticker AN --require-prepared --require-checkpoint
```

Checks:
- checkpoint exists at `drl/dqn/models/contract_rounds/dqn/AN/r1.pt`

## 4) Manual smoke for universal backtester

```bash
python drl/dqn/backtest/backtest_dqn_walkforward.py --strategy Long --asset Forex
python drl/dqn/backtest/backtest_dqn_walkforward.py --strategy DQN --asset Forex
```

Notes:
- `Long` should run without checkpoints.
- `DQN` requires contract checkpoints for all contracts in the selected asset class.
