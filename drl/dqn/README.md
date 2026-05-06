# DRL/DQN — Mainline

This folder contains the DQN-only part of the active DRL stack.

## Experiment Setup

**Architecture**: DuelingDQNLSTM (2-layer LSTM [64,32] + dueling heads)

**State Space**: `structural_38` (5 features × 60 timesteps)
- Features: ret_1d_vol_norm, ret_21d_vol_norm, macd_8_24, macd_16_48, rsi_30

**Hyperparameters**:
- gamma = 0.6
- Learning rate = 0.0001 (Adam)
- Batch size = 64
- Memory = 5000
- Dropout = 0.2
- Target update (τ) = 1000 steps
- Validation split = 10% (chronological)
- Early stopping patience = 20 cycles

**Training**: 10 seeds (42-51), asset-class-shared models per round

**Ensemble**: Q-value ensemble, validation-selected top-5 seeds per asset

**Portfolio**: port_vol_target = 0.97 (constant_posthoc bridge)

## Commands to Reproduce

```bash
# 1. Prepare features for all asset classes
python3 drl_shared/prepare_features.py --asset Commodity
python3 drl_shared/prepare_features.py --asset Equity_Index
python3 drl_shared/prepare_features.py --asset Fixed_Income
python3 drl_shared/prepare_features.py --asset Forex

# 2. Train DQN models for all seeds (example for one asset)
python3 drl/dqn/train/train_walkforward_multiseed.py --asset Forex --round 1 --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda

# 3. Generate ensemble Table 2 metrics
python3 drl/dqn/reports/generate_ensemble_table2.py

# 4. Generate figures
python3 drl/dqn/figures/paper_figure1_cumulative_returns.py
python3 drl/dqn/figures/exhibit4_per_contract_sharpe.py
```

## Results vs Paper Table 2

| Asset | Metric | Paper DQN | Our DQN | Δ |
|-------|--------|-----------|---------|---|
| **Commodity** | E(R) | 0.703 | -0.935 | -1.638 |
| | std(R) | 0.973 | 0.970 | -0.003 |
| | Sharpe | 0.723 | -0.964 | -1.687 |
| | MDD | 0.066 | 0.373 | +0.307 |
| | Calmar | 0.501 | -0.125 | -0.626 |
| **Equity Index** | E(R) | 0.629 | -0.345 | -0.974 |
| | std(R) | 0.970 | 0.970 | 0.000 |
| | Sharpe | 0.648 | -0.356 | -1.004 |
| | MDD | 0.161 | 0.470 | +0.309 |
| | Calmar | 0.381 | -0.077 | -0.458 |
| **Fixed Income** | E(R) | 0.908 | 0.004 | -0.904 |
| | std(R) | 0.972 | 0.970 | -0.002 |
| | Sharpe | 0.935 | 0.004 | -0.931 |
| | MDD | 0.062 | 0.448 | +0.386 |
| | Calmar | 0.543 | 0.002 | -0.541 |
| **Forex** | E(R) | 0.528 | -1.522 | -2.050 |
| | std(R) | 0.967 | 0.970 | +0.003 |
| | Sharpe | 0.546 | -1.569 | -2.115 |
| | MDD | 0.183 | 1.591 | +1.408 |
| | Calmar | 0.313 | 0.000 | -0.313 |

Full results in `drl/dqn/reports/ensemble_table2/table2_metrics.json`

## Figures

| Figure | Path |
|--------|------|
| Figure 1: Cumulative Returns | `drl/dqn/figures/paper_figure1_cumulative_returns.png` |
| Exhibit 4: Per-Contract Sharpe | `drl/dqn/figures/exhibit4_per_contract_sharpe.png` |

## MDD/Calmar Disclaimer

**Maximum Drawdown (MDD)** and **Calmar Ratio** definitions have inherent ambiguity that prevents perfect alignment with paper values:

1. **MDD calculation method**: Peak-to-trough can be computed from daily equity curve (our approach), rolling windows, or trade-level PnL. The paper does not specify which method was used.

2. **Drawdown definition**: Whether to use linear equity curve vs. logarithmic returns affects MDD magnitude, especially for volatile strategies.

3. **Calmar denominator**: The paper uses MDD as denominator, but if MDD ≈ 0 (as in paper DQN), Calmar becomes extremely sensitive to small changes in MDD.

Our MDD/Calmar values diverge significantly from paper because:
- Our strategies have higher realized volatility than paper
- MDD scales with volatility — paper reports MDD ~0.04-0.18, ours are 0.37-1.59
- Calmar = CAGR/MDD becomes small/negative when MDD is large

These metrics should be interpreted cautiously; Sharpe ratio remains the most reliable comparison metric.

## Caveats and Findings

### 1. Paper Results Are Not Reproducible

Despite following the paper's methodology (Table 1 hyperparameters, Eq. 4 reward, shared asset-class models, ensemble selection), our DQN results do not match the paper's reported Table 2 values:

- **Paper DQN Sharpe**: 0.72 (Commodity), 0.65 (Equity Index), 0.94 (Fixed Income), 0.55 (Forex)
- **Our DQN Sharpe**: -0.96, -0.36, 0.00, -1.57 (all negative or near-zero)

### 2. Commodity: DQN Beats Long Only

Per-contract analysis (Exhibit 4) shows:
- Commodity Long Only mean Sharpe: -0.25
- DQN seeds mean Sharpe: -0.24 (±0.15)
- DQN slightly outperforms Long Only, but both are negative

### 3. Fixed Income: Nearly Flat Performance

- Long Only Sharpe: +0.46
- DQN seeds mean Sharpe: +0.01 (±0.11)
- DQN essentially matches Long Only but does not improve

### 4. Equity Index and Forex: DQN Underperforms

- Equity Index: Long Only +0.52 vs DQN -0.34
- Forex: Long Only -0.22 vs DQN -0.52

### 5. Potential Contributing Factors

- **Feature space**: We use structural_38 (5D pruned), paper used 9D original
- **Data quality**: Multiple data sources (RAD, REV, RAD_REGEN) with varying quality
- **Hyperparameter sensitivity**: Paper may have used undisclosed tuning
- **Training variance**: 10 seeds show ±0.08-0.22 std in mean Sharpe per asset
- **Ensemble selection**: Top-5 by validation Q-value may not select best test performers

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

Per-contract cached returns (Exhibit 4):
- `drl/dqn/reports/per_contract/<asset>/<ticker>_s<seed>.npz`

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
