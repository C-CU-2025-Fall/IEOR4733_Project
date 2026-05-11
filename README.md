# IEOR4733_Project

Reproduction of Zhang, Zohren, Roberts (2019) — Deep Reinforcement Learning for Trading.

Paper:
- [Published JFDS 2020](references/Deep-Reinforcement-Learning-for-Trading.pdf) — **canonical version**
- [arXiv v1](https://arxiv.org/pdf/1911.10107) — working paper (superseded)

> Use the published JFDS version. It adds dropout, cross-validation, early stopping,
> and an explicit "Procedures for Controlling Overfitting" section not in the arXiv draft.

---

## Quick Start

```bash
# 1. Prepare features (run once per asset class, or --all)
python3 drl_shared/prepare_features.py --asset Forex
python3 drl_shared/prepare_features.py --all

# 2. Train DQN (9D, 10 seeds, asset-class-shared)
python3 drl/dqn/train/train_dqn_walkforward.py --asset Forex --episodes 100
python3 drl/dqn/train/train_dqn_walkforward.py --asset Forex --round 1 --seed 42  # single seed
python3 drl/dqn/train/train_dqn_walkforward.py --asset Forex --resume             # resume from checkpoint

# 3. Backtest
python3 run_strategy_backtest.py --strategy Long --asset Forex      # baseline
python3 run_strategy_backtest.py --strategy DQN  --asset Forex      # DQN
python3 run_strategy_backtest.py --strategy DQN  --asset Forex --round 1  # single round

# 4. Baseline reproduction (Table 3)
python3 baseline_run.py --table 3 --all-metrics --sigma 0.058
python3 tests/run_structural_38.py --table 3

# 5. Tests
python3 -m unittest tests.test_drl_v2 -v
```

---

## Project Structure

```
IEOR4733_Project/
├── README.md                    # this file
├── config.py                    # data source overrides, contract config
├── deck/                        # presentation slides
├── drl/
│   ├── dqn/                     # DQN training, backtest, figures
│   │   ├── README.md            # DQN methodology and CLI
│   │   ├── model.py / spec.py
│   │   ├── train/               # walkforward training pipeline
│   │   ├── backtest/            # backtest engine
│   │   ├── figures/             # paper figure generation
│   │   └── reports/             # ensemble results, seed selection
│   ├── a2c/                     # A2C workspace (reserved for teammate)
│   └── features/                # precomputed npz (50 contracts × 2 rounds)
├── drl_shared/                  # shared data pipeline
│   ├── data_loader.py
│   ├── prepare_features.py
│   ├── spec.py                  # 9D state space definition
│   └── state_space.py
├── baseline_run.py              # Table 2/3 baseline reproduction
├── run_strategy_backtest.py     # unified backtest entry point
├── batch_prepare_features.py    # batch feature preparation
├── batch_train_dqn.py           # batch DQN training
├── docs/                        # pipeline docs, validation reports
├── tests/                       # integration and unit tests
├── references/                  # paper PDFs
├── config/                      # CLC ASC raw data
└── archive/                     # historical scripts
```

---

## Feature Space (9D — Active)

The active state space is `structural_38_close_norm_9d`: **9 features × 60 timesteps**.

Defined in `drl_shared/spec.py` (`FEATURE_DIM = 9`, `SEQ_LEN = 60`).

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | price / rolling_std | Normalized close price (z-score) |
| 1 | return_21d | 21-day horizon return |
| 2 | return_42d | 42-day horizon return |
| 3 | return_63d | 63-day horizon return |
| 4 | return_252d | 252-day horizon return |
| 5 | MACD(8,24) | EMA(8) − EMA(24), vol-normalized |
| 6 | MACD(16,48) | EMA(16) − EMA(48), vol-normalized |
| 7 | MACD(32,96) | EMA(32) − EMA(96), vol-normalized |
| 8 | RSI_30 | (RSI_30 − 50) / 50 |

Action space: `{−1, 0, +1}` (short, flat, long).

### 12D Experiment (Archived)

A 12D enhanced feature set (`structural_38_enhanced_12d`) was tested as an extension:
ret_1d/5d/21d/126d, macd_8_24, rsi_5, rsi_30, atr_norm, vol_norm, oi_chg, drawdown, gap_overnight.

The 12D experiment is documented in `docs/` and `drl/dqn/finetuning_comparisons.md`.
All reported results in this repo use the **9D** feature space.

---

## DQN Training Configuration

All hyperparameters in `drl/dqn/spec.py`:

| Parameter | Value | Source |
|-----------|-------|--------|
| Features | 9D × 60 steps | structural_38 |
| Seeds | 42–51 (10 seeds) | multi-seed ensemble |
| Episodes | 100 per seed | |
| Gamma | 0.6 | 3-gamma comparison (0.5/0.6/0.7) |
| Learning rate | 0.0001 | Paper Table 1 |
| Batch size | 64 | Paper Table 1 |
| LSTM | 2-layer 64→32, Dueling | Paper |
| Dropout | 0.2 after LSTM | Published JFDS 2020 |
| Early stopping | patience 20 on val reward | |
| Target net copy | every 1000 learn steps | |
| Optimizer | Adam, gradient clip max_norm=1.0 | |

Walk-forward rounds (defined in `drl_shared/spec.py RETRAIN_ROUNDS`):
- **r1**: train 2005–2010, test 2011–2015
- **r2**: train 2005–2015, test 2016–2019

Ensemble: top-5 seeds by validation reward → Q-value averaging.

See [drl/dqn/README.md](drl/dqn/README.md) for full methodology and results.

---

## Baseline

`tests/run_structural_38.py` is the authoritative reproducible baseline.

Trade-world `<=15%` error: **28/28** metrics across 4 asset classes.

| Asset | E(R) Ours | E(R) Paper | %Err |
| --- | --- | --- | --- |
| Commodity | −0.263 | −0.298 | 11.8 |
| Equity Index | +0.541 | +0.504 | 7.4 |
| Fixed Income | +0.568 | +0.605 | 6.2 |
| Forex | −0.179 | −0.198 | 9.6 |

---

## Deck / Presentations

See [deck/](deck/) for presentation slides:
- `DL_presentation_v3.pdf` — final presentation

---

## References

- [docs/data_issues.md](docs/data_issues.md) — known data quality issues
- [docs/drl_pipeline.md](docs/drl_pipeline.md) — pipeline architecture
- [drl/dqn/README.md](drl/dqn/README.md) — DQN methodology, results, figures
- [PROJECT_MEMORY.md](PROJECT_MEMORY.md) — historical search notes

---

## Historical Note

`41/45` is retained only as an experimental upper-bound reproducer in
[tests/run_legacy_41.py](tests/run_legacy_41.py). It is not part of the
active baseline or DRL interpretation.
