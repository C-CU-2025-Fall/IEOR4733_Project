# PROJECT_MEMORY.md
# Last updated: 2026-04-30

Read this first when resuming work on this repo.

## 0. Current Truth

The repo now has one active interpretation:

1. baseline
- `tests/run_structural_38.py`
- this is the true reproducible baseline
- it locks the `structural_38` source overrides and exclusions

2. unified backtester
- one portfolio/metric stack for baseline and DRL
- all final metrics come from the same simulated portfolio path

3. DRL mainline
- DQN uses the same `structural_38` data doctrine as the baseline
- DQN trains one shared model per asset class per retrain round
- **Strictly follows ZZR2019 JFDS 2020 published paper** (Table 1 hyperparams, Section 3.1 state space, Eq.4 reward)

## 0.5. Paper Version

Two versions exist:
- **arXiv v1** (`references/DRL_main.pdf`): working paper, Nov 2019
- **Published JFDS 2020** (`references/Deep-Reinforcement-Learning-for-Trading.pdf`): **canonical**

Key additions in published version:
- Dropout 0.2 after LSTM layers
- 10% cross-validation split with 20-epoch early stopping
- Explicit "Procedures for Controlling Overfitting" section
- Cross-contract training as overfitting control
- Updated references (Lim, Zohren, Roberts 2019)

Core methodology (LSTM arch, asset-class grouping, 5-year retrain, vol scaling) is identical.

## 1. Baseline

Primary commands:
- `python baseline_run.py --table 3 --all-metrics --sigma 0.058`
- `python tests/run_structural_38.py --table 3`

## 2. DRL Mainline

### State Space (Pruned v2 — 2026-04-30)

Feature pruning analysis on the original 9D space found severe multicollinearity (15 pairs with |r| > 0.7, effective rank ~4-5). The pruned 5D set reduces noise while retaining independent signal:

- `seq_len = 60`, `feature_dim = 5`, `market_feature_dim = 5`
- `state_spec_version = structural_38_pruned_v2_5d`

| Index | Feature | Formula | Rationale |
|-------|---------|---------|-----------|
| 0 | ret_1d_vol_norm | `r_t / sigma_t` | Replaces non-stationary `p_t/rolling_std(p,60)`. Scale [-5, 5], max |r|=0.27 with other features, VIF=1.26 |
| 1 | ret_21d_vol_norm | `(p_t - p_{t-21}) / (sigma_t * sqrt(21))` | Core short-term momentum, best aligned with gamma=0.3 |
| 2 | macd_8_24 | `(EMA_8 - EMA_24) / sigma_63(p)`, then `/ sigma_252(q)` | Fastest MACD, short-term trend |
| 3 | macd_16_48 | Same formula, spans (16, 48) | Medium MACD, ~0.6 corr with MACD(8,24) |
| 4 | rsi_30 | `(RSI_30 - 50) / 50`, Wilder smoothing | Mean-reversion complement, well-scaled [-1, 1] |

**Removed from original 9D:**
- `p_t / rolling_std(p, 60)` — scale [4, 342] vs others [-3, 3], non-stationary
- `ret_42d`, `ret_63d` — corr > 0.72 with ret_21d, redundant at gamma=0.3
- `ret_252d` — annual momentum, gamma=0.3 can't exploit
- `MACD(32, 96)` — corr 0.77 with MACD(16,48), too slow
- `prev_action` — removed; agent infers position from reward signal

### DQN Architecture (Paper Table 1 + JFDS 2020 additions)
- 2-layer LSTM [64, 32] + Leaky-ReLU
- Dropout 0.2 after LSTM layers (published paper addition)
- Dueling DQN + Double DQN + Fixed Q-targets (tau=1000)
- LR=0.0001, gamma=0.3, bp=0.002, batch=64
- memory: dynamic `n_contracts * 25000` (paper default 5000)
- epsilon: 0.3 -> 0.05 linear decay over `n_contracts * 25000` steps; greedy 0.0 for validation/inference
- 10% validation split, early stopping patience=20

### Training
- 200 cycles per round (paper default); each cycle visits every contract once
- 10% chronological validation split, early stopping patience=20
- no active 1500-step truncation; episodes run to natural env termination
- round extensibility: add entries to `drl_shared/spec.py RETRAIN_ROUNDS` for r3+
- 2026-04-29 audit fixes:
  - backtest engine loads features from `.npz` (bit-exact with training); warns if fallback
  - `checkpoint_metadata()` correctly records `linear_decay` epsilon and dynamic memory
  - 8 new unit tests: Eq.4 reward hand-verification, epsilon dynamics, feature causality
- 2026-04-28 leakage fix:
  - prepared feature artifacts persist explicit round boundaries
  - training slices only the true train period, last 10% as validation
  - backtest/inference only uses the true round test period
- Fail-fast: RuntimeError if 0 val envs constructed
- Pipeline checks: data sanity, env preflight, agent health, cycle monitoring
- One model per asset class per round

### Commands
```bash
python drl_shared/prepare_features.py --asset Forex
python drl/dqn/train/train_dqn_walkforward.py --asset Forex --episodes 200 --device cuda
python run_strategy_backtest.py --strategy DQN --asset Forex
```

### Artifacts
- Features: `drl/features/<ticker>/r<round>.npz`
- DQN bundles: `drl/dqn/models/<asset_class>/r<round>/<run_id>/`
- Active state spec: `structural_38_pruned_v2_5d`
- All 96 contract artifacts (48 tickers x 2 rounds) regenerated with 5 market features

## 3. Eq.4 Reward

- Additive `r_t = p_t - p_{t-1}`
- Volatility-scaled positions: `sigma_tgt / sigma_{t-1}`
- Transaction cost: `bp * p_{t-1} * |scaled_position_change|`

## 4. Alignment Snapshot

Current retained structural-38 snapshot:
- Trade-world metrics only: Table 3 <=10: 23/28, <=15: 28/28
- Table 2 <=10: 24/28, <=15: 25/28
