# DRL/DQN — Mainline

This folder contains the DQN-only part of the active DRL stack.

## §1 Experiment Setup

**Model**: DuelingDQNLSTM (2-layer LSTM 64→32, dueling value+advantage heads)

**Features**: structural_38, 9 features × 60 timesteps

| Index | Feature | Description |
|-------|---------|-------------|
| F0 | price/rolling_std | Normalized close price |
| F1-F4 | return horizons | Returns at 21/42/63/252 day horizons |
| F5-F7 | MACD pairs | MACD signals for (8,24), (16,48), (32,96) spans |
| F8 | RSI_30 | RSI with 30-day window |

**Training**: 10 seeds (42-51), asset-class-shared models per round
- gamma = 0.6
- Episodes = 100

**Ensemble**: Top-5 validation-selected Q-value averaging

**Portfolio**: port_vol_target=0.97 (constant_posthoc bridge)

**Reference**: spec.py for all hyperparameters

## §2 Hyperparameter Selection Process

### 2.1 Exploration (Epsilon Schedule)

The training uses a 4-phase epsilon schedule:

- **Phase 1**: ε=1.0 (pure random) for first 5000 transitions to fill replay buffer
- **Phase 2**: ε=0.30→0.10 over cycles 0-20 (structured exploration)
- **Phase 3**: ε=0.10→0.01 over cycles 20-90 (fine-tuning)
- **Phase 4**: ε=0.01 fixed for cycles 90-100 (exploitation)

Fraction-based interpolation via spec.py EPS_SCHEDULE.

### 2.2 Gamma Discount Factor Tuning

Summary from `drl/dqn/finetuning_comparisons.md`:

Tested gamma ∈ {0.5, 0.6, 0.7} on Forex with 5 seeds each:

- **Gamma=0.6** wins 9/9 metrics on r1, 8/9 on r2
- **Gamma=0.5**: over-aggressive (26% trade rate), over-trades
- **Gamma=0.7**: over-passive (2% trade rate), nearly flat
- **Gamma=0.6**: optimal balance (~8% trade rate, strong long bias)

Full analysis in finetuning_comparisons.md

### 2.3 Replay Buffer Design

StratifiedReplayBuffer supports 3 modes:

- **uniform**: random sampling (paper default, used in our experiments)
- **action_balanced**: equal samples per action class to combat action imbalance
- **reward_stratified**: percentile-based bins prioritizing high-impact rewards

We use uniform (paper default) for all reported results.

## §3 Feature Space

Full 9-dimensional feature space:

| Index | Feature | Formula | Property |
|-------|---------|---------|----------|
| 0 | Normalized close price | p_t / std_60(p) | Scale normalization |
| 1-4 | Multi-horizon returns | (p_t-p_{t-H})/(σ_t·√H), H∈{21,42,63,252} | Momentum at 4 scales |
| 5-7 | MACD pairs | q_t/std_252(q_t), q_t=(EMA_S-EMA_L)/std_63(p) | Trend signals |
| 8 | RSI_30 | (RSI-50)/50 | Mean-reversion |

State spec version: structural_38_close_norm_9d

## §4 Training Methodology

- One shared DQN per asset class per round (cross-contract generalization)
- Interleaved training: steps alternated across contracts for balanced replay
- Walk-forward: r1 train 2005-2010 test 2011-2015, r2 train 2005-2015 test 2016-2019
- Validation: 10% chronological split, early stopping patience=20

Reproduce command:

```bash
python3 drl/dqn/train/train_dqn_walkforward.py --asset "Equity Index" --round 1 --gamma 0.6 --seed 42 --episodes 100 --device cuda
```

Note: Training of all 60 models (3 assets × 2 rounds × 10 seeds) was parallelized across 4 NVIDIA GB10 GPUs. Logs at /tmp/g06_p*.log.

## §5 Ensemble Methodology

- Q-value ensemble (NOT majority voting): mean Q-values from top-k models → argmax
- Top-5 selected by validation best_val_reward (no test info)
- Different seeds per round: r1 models ranked by r1 val, r2 models ranked by r2 val

Reproduce:

```bash
python3 drl/dqn/reports/generate_ensemble_table2.py
```

## §6 Results vs Paper Table 2

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
| | MDD | 0.161 | 0.460 | +0.299 |
| | Calmar | 0.381 | -0.079 | -0.460 |
| **Fixed Income** | E(R) | 0.908 | 0.004 | -0.904 |
| | std(R) | 0.972 | 0.970 | -0.002 |
| | Sharpe | 0.935 | 0.004 | -0.931 |
| | MDD | 0.062 | 0.490 | +0.428 |
| | Calmar | 0.543 | 0.002 | -0.541 |
| **Forex** | E(R) | 0.528 | -1.522 | -2.050 |
| | std(R) | 0.967 | 0.970 | +0.003 |
| | Sharpe | 0.546 | -1.569 | -2.115 |
| | MDD | 0.183 | 1.553 | +1.370 |
| | Calmar | 0.313 | 0.000 | -0.313 |

Full results in `drl/dqn/reports/ensemble_table2/table2_metrics.json`

std(R) aligned to 0.97. Our Sharpe ratios are negative because our models underperform the paper claims — this is the documented irreproducibility finding.

## §7 Figures

| Figure | Path |
|--------|------|
| Figure 1: Cumulative Trade Returns (5 panels) | figures/paper_figure1_cumulative_returns.png |
| Exhibit 4: Per-Contract Sharpe Boxplot | figures/exhibit4_per_contract_sharpe.png |
| Supplementary: Rolling Sharpe (252-day) | figures/supp_rolling_sharpe.png |
| Supplementary: Drawdown Curves | figures/supp_drawdown.png |
| Supplementary: Monthly Returns Heatmap | figures/supp_monthly_heatmap.png |
| Supplementary: Year-by-Year Performance | figures/supp_yearly_bars.png |

## §8 MDD/Calmar Disclaimer

**Maximum Drawdown (MDD)** and **Calmar Ratio** definitions have inherent ambiguity that prevents perfect alignment with paper values:

1. **MDD calculation method**: Peak-to-trough can be computed from daily equity curve (our approach), rolling windows, or trade-level PnL. The paper does not specify which method was used.

2. **Drawdown definition**: Whether to use linear equity curve vs. logarithmic returns affects MDD magnitude, especially for volatile strategies.

3. **Calmar denominator**: The paper uses MDD as denominator, but if MDD ≈ 0 (as in paper DQN), Calmar becomes extremely sensitive to small changes in MDD.

Our MDD/Calmar values diverge significantly from paper because:
- Our strategies have higher realized volatility than paper
- MDD scales with volatility — paper reports MDD ~0.04-0.18, ours are 0.37-1.55
- Calmar = CAGR/MDD becomes small/negative when MDD is large

These metrics should be interpreted cautiously; Sharpe ratio remains the most reliable comparison metric.

## §9 Caveats and Findings

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

- **Feature space**: We use structural_38 (9D), paper used 9D original
- **Data quality**: Multiple data sources (RAD, REV, RAD_REGEN) with varying quality
- **Hyperparameter sensitivity**: Paper may have used undisclosed tuning
- **Training variance**: 10 seeds show ±0.08-0.22 std in mean Sharpe per asset
- **Ensemble selection**: Top-5 by validation Q-value may not select best test performers

## §10 Artifacts

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

Ensemble results:
- `drl/dqn/reports/ensemble_table2/`

Per-contract cached returns (Exhibit 4):
- `drl/dqn/reports/per_contract/<asset>/<ticker>_s<seed>.npz`

## §11 Historical Note: 5D Feature Pruning (April 2026)

Prior to selecting the 9D structural_38 space, a 5D pruned feature space was tested (April 30, 2026):

- Based on correlation analysis: 15 high-correlation pairs (|r|>0.7), 4-5 effective dimensions
- Pruning removed F0 (non-stationary), F2/F3 (redundant returns), F4 (long-term at low gamma), F7 (slow MACD)

This space was NOT used in the final experiments; all reported results use the full 9D space. See spec.py HISTORY for details.
