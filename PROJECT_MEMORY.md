# PROJECT_MEMORY.md
# Last updated: 2026-04-26

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

Core methodology (LSTM arch, asset-class grouping, memory 5000, 5-year retrain, vol scaling) is identical.

## 1. Baseline

Primary commands:
- `python baseline_run.py --table 3 --all-metrics --sigma 0.058`
- `python tests/run_structural_38.py --table 3`

## 2. DRL Mainline

### State Space (Paper Section 3.1)
- `seq_len = 60`, `feature_dim = 7`
- Feature 0: Normalized close price (`p_t / rolling_std(p, 60)`)
- Features 1-4: Returns over 21/42/63/252 days, normalized by `σ_t * √H`
- Feature 5: Averaged MACD normalized by 63-day price volatility
- Feature 6: RSI(30) normalized to [-1, 1]

### DQN Architecture (Paper Table 1 + JFDS 2020 additions)
- 2-layer LSTM [64, 32] + Leaky-ReLU
- Dropout 0.2 after LSTM layers (published paper addition)
- Dueling DQN + Double DQN + Fixed Q-targets (τ=1000)
- LR=0.0001, γ=0.3, bp=0.002, batch=64, memory=5000
- ε: 0.3 → 0.05 over 50000 steps
- 10% validation split, early stopping patience=20

### Training
- 200 episodes per round (paper default)
- 10% validation split, early stopping patience=20
- Fail-fast: RuntimeError if 0 val envs constructed (catches missing imports)
- Pipeline checks: data sanity, env preflight, agent health, cycle monitoring
- Unit tests: `drl/dqn/train/test_training_pipeline.py` (19 tests)
- One model per asset class per round
- Batch scheduler: `drl/dqn/train/train_all_assets.py`

### Commands
```bash
python drl_shared/prepare_features.py --asset Forex
python drl/dqn/train/train_dqn_walkforward.py --asset Forex --episodes 200 --device cuda
python run_strategy_backtest.py --strategy DQN --asset Forex
```

### Artifacts
- Features: `drl/features/<ticker>/r<round>.npz`
- DQN bundles: `drl/dqn/models/<asset_class>/r<round>/<run_id>/`

## 3. Eq.4 Reward

- Additive `r_t = p_t - p_{t-1}`
- Volatility-scaled positions: `σ_tgt / σ_{t-1}`
- Transaction cost: `bp * p_{t-1} * |scaled_position_change|`

## 4. Alignment Snapshot

Current retained structural-38 snapshot:
- Trade-world metrics only: Table 3 ≤10: 23/28, ≤15: 28/28
- Table 2 ≤10: 24/28, ≤15: 25/28
