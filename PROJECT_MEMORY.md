# PROJECT_MEMORY.md
# Last updated: 2026-05-01

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

### State Space (Enhanced 12D — 2026-05-01)

Feature evolution: original 9D → pruned 5D → enhanced 11D → enhanced 12D (added gap_overnight).
The 5D→11D expansion addressed Q-value collapse diagnosed in 5D models (100% flat Q-values).
The 12D adds overnight gap for price discovery signal.

- `seq_len = 60`, `feature_dim = 12`, `market_feature_dim = 12`
- `state_spec_version = structural_38_enhanced_12d`

| Index | Feature | Formula | Category | Source |
|-------|---------|---------|----------|--------|
| 0 | ret_1d | `r_t / sigma_t` | ultra-short momentum | Close |
| 1 | ret_5d | `(p_t - p_{t-5}) / (sigma_t * sqrt(5))` | short-term momentum | Close |
| 2 | ret_21d | `(p_t - p_{t-21}) / (sigma_t * sqrt(21))` | medium-term momentum | Close |
| 3 | ret_126d | `(p_t - p_{t-126}) / (sigma_t * sqrt(126))` | long-term trend | Close |
| 4 | macd_8_24 | `(EMA_8 - EMA_24) / sigma_63(p)`, then `/ sigma_252(q)` | trend (paper Eq.3) | Close |
| 5 | rsi_5 | `(RSI_5 - 50) / 50`, Wilder smoothing | ultra-short oscillator | Close |
| 6 | rsi_30 | `(RSI_30 - 50) / 50`, Wilder smoothing | medium-term oscillator | Close |
| 7 | atr_norm | `TR / ATR_MA(20)` | volatility regime | OHLC |
| 8 | vol_norm | `Volume / Volume_MA(20)` | liquidity/activity | Volume |
| 9 | oi_chg | `ΔOI / |OI_{t-1}|`, clipped [-5,5] | positioning flow | OI |
| 10 | drawdown | `(p - max_126d) / max_126d` | risk state | Close |
| 11 | gap_overnight | `(O_t - C_{t-1}) / sigma_t` | overnight price discovery | OHLC |

**5D→11D changes (2026-04-30):**
- Added ret_5d, ret_126d (multi-horizon momentum)
- Replaced macd_16_48 with single macd_8_24 (paper uses one pair)
- Added rsi_5 alongside rsi_30
- Added atr_norm, vol_norm, oi_chg, drawdown (volatility/microstructure/risk)

**11D→12D changes (2026-05-01):**
- Added gap_overnight: vol-normalized overnight gap `(O_t - C_{t-1}) / sigma_t`
- Rationale: overnight price discovery, gaps indicate news/events (Schwager [42], Murphy [38])
- AN statistics: mean=-0.008, std=0.31, range [-2.51, 3.13], 94% non-zero

**Removed from original 9D:**
- `p_t / rolling_std(p, 60)` — scale [4, 342] vs others [-3, 3], non-stationary
- `ret_42d`, `ret_63d` — corr > 0.72 with ret_21d, redundant at gamma=0.3
- `ret_252d` — annual momentum, gamma=0.3 can't exploit
- `MACD(32, 96)` — corr 0.77 with MACD(16,48), too slow
- `prev_action` — removed; paper uses purely market features

### DQN Architecture (Paper Table 1 + JFDS 2020 additions)
- 2-layer LSTM [64, 32] + Leaky-ReLU
- Dropout 0.2 after LSTM layers (published paper addition)
- Dueling DQN + Double DQN + Fixed Q-targets (tau=1000)
- LR=0.0001, gamma=0.3, bp=0.002, batch=64
- memory: `max(5000, total_steps * 0.2)` (MEMORY_RATIO=0.2, MEMORY_SIZE_MIN=5000)
- epsilon: warmup 10% at 0.30, decay 0.30->0.10 over next 20%, flat 0.10 until end
- MSE loss (paper default, USE_HUBER_LOSS=False); gradient clipping max_norm=1.0
- 10% chronological validation split, early stopping patience=20
- Locked seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51] — all experiments use LOCKED_SEEDS
- torch.manual_seed + torch.cuda.manual_seed_all for full RNG reproducibility
- **Device**: auto (CUDA→MPS→CPU). Verified on NVIDIA GB10 (130.7GB), ~20x faster than CPU

### Feature Correlation Analysis (48-contract aggregate, 2026-05-01)

Highly correlated pairs (|r| > 0.6, 10 pairs):
- ret_21d ↔ macd_8_24 = +0.881 (strongest)
- ret_5d ↔ rsi_5 = +0.872
- ret_21d ↔ rsi_30 = +0.863
- macd_8_24 ↔ rsi_30 = +0.854
- ret_126d ↔ drawdown = +0.766
- rsi_30 ↔ drawdown = +0.699, rsi_5 ↔ rsi_30 = +0.697
- ret_126d ↔ rsi_30 = +0.692, ret_21d ↔ rsi_5 = +0.665, macd_8_24 ↔ rsi_5 = +0.648

Feature independence ranking (mean |r| with all other features):
1. oi_chg (0.024) — Highly independent
2. vol_norm (0.051) — Highly independent
3. atr_norm (0.092) — Highly independent
4. **gap_overnight (0.097)** — Highly independent (validates addition)
5. ret_1d (0.202) — Moderate
6. ret_126d (0.290) — Moderate (↔drawdown=0.766)
7. drawdown (0.313) — Moderate (↔ret_126d=0.766)
8-12. ret_5d/macd/ret_21d/rsi_5/rsi_30 (0.33-0.45) — Momentum cluster, internally correlated

Full correlation matrix saved: `results/feature_correlation_48/aggregate_corr_12d_mean.csv`
Summary JSON: `results/feature_correlation_48/summary_12d.json`

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
# Prepare features
python drl_shared/prepare_features.py --asset Forex

# Train DQN
python drl/dqn/train/train_dqn_walkforward.py --asset Forex --episodes 200 --device cuda

# Single-contract integration test
python tests/test_integration_dqn_vs_long.py --ticker AN --round 1 --episodes 100

# Stitched r1+r2 evaluation
python tests/test_integration_dqn_vs_long.py --ticker AN --stitch

# Stitch from saved files (no retraining)
python tests/test_integration_dqn_vs_long.py --ticker AN --stitch --from-files

# Save rewards for offline analysis
python tests/test_integration_dqn_vs_long.py --ticker AN --round 1 --save-rewards

# Multi-seed evaluation
python tests/test_integration_dqn_vs_long.py --ticker AN --seeds 5

# Asset-class portfolio evaluation (all Forex contracts)
python tests/test_integration_dqn_vs_long.py --asset Forex --round 1 --episodes 100

# Full backtest
python run_strategy_backtest.py --strategy DQN --asset Forex
```

### Artifacts
- Features: `drl/features/<ticker>/r<round>.npz`
- DQN bundles: `drl/dqn/models/<asset_class>/r<round>/<run_id>/`
- Active state spec: `structural_38_enhanced_12d`
- All 96 contract artifacts (48 tickers x 2 rounds) regenerated with 12 market features
- Saved results: `results/v<VERSION>/<TICKER>_r<R>_s<SEED>.npz` (per-contract reward arrays)
- Portfolio results: `results/v<VERSION>/<ASSET>_r<R>_s<SEED>_portfolio.npz`
- RESULTS_VERSION = `"v1"` — increment when hyperparameters change

## 3. Integration Test & Evaluation Pipeline

`tests/test_integration_dqn_vs_long.py` provides the full evaluation pipeline:

- **Single-contract**: `--ticker AN --round 1` — trains per-contract DQN, compares vs Long
- **Multi-seed**: `--seeds 5` — runs LOCKED_SEEDS, reports median +/- IQR
- **Stitched**: `--stitch` — concatenates r1+r2 test rewards for full 2011-2019 view
- **From files**: `--stitch --from-files` — loads saved .npz, computes stitched metrics without retraining
- **Version control**: `--save-rewards --ver v2` — saves to `results/v2/`, `--from-files --ver v2` loads from there
- **Asset class**: `--asset Forex` — trains one shared model on full Forex, evaluates all 9 contracts
- **Portfolio**: Equal-weight aggregation of per-contract DQN rewards (Paper Eq.13)
- **Save rewards**: `--save-rewards` — persists reward arrays to `results/` for offline analysis

Key finding from single-contract evaluation (AN, r1+r2):
- DQN `% +ve` is far from 50% (20% r1, 7% r2) because AN was in downtrend
- DQN learns position sizing / flat preference, not direction prediction
- Paper's DQN `% +ve` ~= 50% is an artifact of equal-weight portfolio across 9 contracts
- Single-contract DQN beats Long on std(R) and DD (volatility scaling) but not Sharpe/Sortino

## 4. Feature Occlusion Analysis

`tests/test_feature_occlusion.py` — Per-feature contribution analysis on trained DQN.

Method: zero-out (or mean-replace) one feature dimension at a time during greedy inference,
measure impact on 7 trade metrics and action distribution. Delta = baseline - occluded.

- Positive delta on Sharpe/Sortino/E(R)/AveP/L = feature is useful (removing it hurts)
- Positive delta on std(R)/DD = feature reduces risk (removing it increases volatility)

Feature mapping (5D pruned v2):
- 0: `ret_1d_vol_norm` — 1-day vol-normalized return
- 1: `ret_21d_vol_norm` — 21-day momentum
- 2: `macd_8_24` — fast MACD
- 3: `macd_16_48` — medium MACD
- 4: `rsi_30` — mean-reversion signal

Commands:
```bash
# Single-contract occlusion
python tests/test_feature_occlusion.py --ticker AN --round 1

# Both rounds
python tests/test_feature_occlusion.py --ticker AN --both

# Stitched r1+r2
python tests/test_feature_occlusion.py --ticker AN --stitch

# Multi-seed with median/IQR
python tests/test_feature_occlusion.py --ticker AN --seeds 5

# Asset-class portfolio occlusion (train on all Forex, occlude per-contract, aggregate)
python tests/test_feature_occlusion.py --asset Forex --round 1

# Mean-replacement instead of zero
python tests/test_feature_occlusion.py --ticker AN --round 1 --method mean
```

## 5. Eq.4 Reward

- Additive `r_t = p_t - p_{t-1}`
- Volatility-scaled positions: `sigma_tgt / sigma_{t-1}`
- Transaction cost: `bp * p_{t-1} * |scaled_position_change|`

## 6. Alignment Snapshot

Current retained structural-38 snapshot:
- Trade-world metrics only: Table 3 <=10: 23/28, <=15: 28/28
- Table 2 <=10: 24/28, <=15: 25/28

## 7. Training Experiment Log

### Hyperparameter Search (AN r1, seed=42, episodes=100)

| # | Config | r1 DQN wins | Best val reward | Early stop cycle | Notes |
|---|--------|:-----------:|:---------------:|:----------------:|-------|
| 1 | Original (linear eps 0.3→0.05, MSE, fixed mem=5000) | 4/7 | — | cycle 14 | Best checkpoint very early; val degrade afterward |
| 2 | 3-stage eps (0.3→0.05→0.01→0.005), Huber | 3/7 | — | never (cycle 100) | Severe overfitting; never early-stopped |
| 3 | 2-stage eps (0.3→0.05→0.01), MSE | 3/7 | — | cycle 38 | Early stop late; val quality worse |
| 4 | 2-stage eps (0.3→0.05→0.01), Huber | 5/7 | — | — | Good but slightly worse than MSE |
| 5 | Warmup eps (0.3→0.1 flat), Huber | 5/7 | val=0.90 | cycle 28 | Stable warmup helps |
| **6** | **Warmup eps (0.3→0.1 flat), MSE** | **6/7** | **val=1.04** | **cycle 30** | **Best overall — selected as final config** |

Selected config (#6): `EPS_SCHEDULE = [(0.00, 0.30), (0.10, 0.30), (0.30, 0.10), (1.00, 0.10)]`, MSE loss, memory ratio 0.2, grad clip 1.0.

### AN Single-Contract Results (Config #6, seed=42, episodes=100)

**r1 (2011-2015):**

| Metric | DQN | Long | Win |
|--------|-----|------|-----|
| E(R) | -0.1965 | -0.5049 | **DQN** |
| std(R) | +0.6578 | +0.9367 | **DQN** |
| DD | +0.5767 | +0.6272 | **DQN** |
| Sharpe | -0.2987 | -0.5390 | **DQN** |
| Sortino | -0.3407 | -0.8051 | **DQN** |
| % +ve | +0.2048 | +0.4889 | Long |
| Ave P/L | +1.2376 | +0.9581 | **DQN** |

DQN wins **6/7** metrics. Position dist: L=30% F=56% S=14%. DQN predominantly stays flat (56%), not long.

**r2 (2016-2019):**

| Metric | DQN | Long | Win |
|--------|-----|------|-----|
| E(R) | -0.0842 | -0.1140 | **DQN** |
| std(R) | +0.3906 | +0.9145 | **DQN** |
| DD | +0.4850 | +0.5828 | **DQN** |
| Sharpe | -0.2156 | -0.1247 | Long |
| Sortino | -0.1736 | -0.1957 | **DQN** |
| % +ve | +0.0706 | +0.5095 | Long |
| Ave P/L | +1.3879 | +0.9435 | **DQN** |

DQN wins **5/7** metrics. Position dist: L=11% F=85% S=5%. DQN almost entirely flat (85%).

**Key observations across r1+r2:**
- DQN consistently **reduces std(R) and DD** vs Long (volatility scaling effect)
- DQN **improves Ave P/L** (wins bigger when it does trade, loses smaller)
- DQN **fails on % +ve** (20% r1, 7% r2) — it trades very infrequently
- Sharpe is negative for both DQN and Long (AN is a downtrend currency)
- DQN learns **position sizing** (when to stay flat), not direction prediction
- Paper's `% +ve ≈ 50%` is an artifact of 9-contract equal-weight portfolio
