# IEOR4733_Project

Reproduction workspace for Zhang, Zohren, Roberts (2019).

The repo now has one active interpretation:

- baseline = `structural_38` long-only line
- DRL/DQN uses the same data doctrine
- one unified backtester computes all final metrics
- old versioned DQN artifact families are archive-only, not mainline

Historical search waves and abandoned branches live in:
- [PROJECT_MEMORY.md](PROJECT_MEMORY.md)

Paper:
- [Published JFDS 2020](references/Deep-Reinforcement-Learning-for-Trading.pdf) — **canonical version**
- [arXiv v1](https://arxiv.org/pdf/1911.10107) — working paper (superseded)

> ⚠️ Use the published JFDS version. It adds dropout, cross-validation, early stopping,
> and an explicit "Procedures for Controlling Overfitting" section not in the arXiv draft.

## Quick Start (One-Liners)

```bash
# 1. Prepare shared features (run once per asset class)
python3 drl_shared/prepare_features.py --asset Forex
python3 drl_shared/prepare_features.py --all

# 2. Train DQN
python3 drl/dqn/train/train_dqn_walkforward.py --asset Forex --episodes 200 --device cpu
python3 drl/dqn/train/train_dqn_walkforward.py --asset Forex --round 1 --tickers AN BN  # debug 1-2 contracts
python3 drl/dqn/train/train_dqn_walkforward.py --asset Forex --resume  # resume from checkpoint

# 3. Backtest (any strategy)
python3 run_strategy_backtest.py --strategy Long --asset Forex
python3 run_strategy_backtest.py --strategy DQN  --asset Forex
python3 run_strategy_backtest.py --strategy DQN  --asset Forex --round 1  # single round

# 4. Baseline reproduction
python3 baseline_run.py --table 3 --all-metrics --sigma 0.058
python3 tests/run_structural_38.py --table 3

# 5. Run tests
python3 -m unittest tests.test_drl_v2 -v

# 6. DQN evaluation (integration test)
python tests/test_integration_dqn_vs_long.py --ticker AN --round 1          # single-contract
python tests/test_integration_dqn_vs_long.py --ticker AN --stitch            # stitched r1+r2
python tests/test_integration_dqn_vs_long.py --ticker AN --stitch --from-files  # from saved .npz
python tests/test_integration_dqn_vs_long.py --ticker AN --stitch --from-files --ver v2  # different version
python tests/test_integration_dqn_vs_long.py --ticker AN --seeds 5           # multi-seed
python tests/test_integration_dqn_vs_long.py --asset Forex --round 1         # portfolio (9 contracts)
python tests/test_integration_dqn_vs_long.py --ticker AN --save-rewards      # persist to results/v1/
python tests/test_integration_dqn_vs_long.py --ticker AN --save-rewards --ver v2  # persist to results/v2/

# 7. Feature occlusion analysis
python tests/test_feature_occlusion.py --ticker AN --round 1                  # single round
python tests/test_feature_occlusion.py --ticker AN --both                     # both rounds
python tests/test_feature_occlusion.py --ticker AN --stitch                   # stitched r1+r2
python tests/test_feature_occlusion.py --ticker AN --seeds 5                  # multi-seed
python tests/test_feature_occlusion.py --asset Forex --round 1               # portfolio-level
python tests/test_feature_occlusion.py --ticker AN --round 1 --method mean   # mean-replacement
```

## Current Baseline

`tests/run_structural_38.py` is the authoritative reproducible baseline.

It locks:
- `STRUCTURAL_38_OVERRIDES`
- `STRUCTURAL_38_EXCLUDED`
- the current trade-world baseline interpretation

This is the baseline we actually compare against for local reproducibility. The
ideal paper/data world is useful context, but it is not the active benchmark
for this repo.

## DRL Mainline

DRL now follows the same baseline doctrine as the structural baseline:

- same `structural_38` source overrides
- same `structural_38` exclusions
- same unified backtester
- no active `v0 / v2 / v2.1 / v3` story

Current DRL paths:
- features:
  - `drl/features/<ticker>/r<round>.npz`
  - `drl/features/<asset_class>/r<round>/index.json`
- DQN bundles:
  - `drl/dqn/models/<asset_class>/r<round>/<run_id>/`

Current DQN training unit:
- one shared DQN model per asset class per retrain round
- default training covers both `r1` and `r2`
- each asset-class cycle visits every eligible contract once with a shared replay buffer
- dropout (0.2) after LSTM layers (published paper)
- chronological 90/10 train/validation split per contract
- early stopping with patience 20 on validation reward
- ε: warmup 10% at 0.30, decay 0.30→0.10 over next 20%, flat at 0.10
- memory: `max(5000, total_steps * 0.2)` (MEMORY_RATIO=0.2, MEMORY_SIZE_MIN=5000)
- MSE loss (paper default, USE_HUBER_LOSS=False); gradient clipping max_norm=1.0
- locked seeds: `[42, 43, 44, 45, 46]`
- validation envs crash-fast: if 0 val envs constructed, `RuntimeError` aborts training
- resume: `--resume` restores weights, optimizer, replay buffer, and full RNG state for reproducible continuation
- round extensibility: add entries to `drl_shared/spec.py RETRAIN_ROUNDS` for r3+

Training pipeline checks (fail-fast):
- **Data sanity**: NaN/Inf, length consistency, feature shape `(n, 5)`, sigma range, date monotonicity
- **Env preflight**: usable steps >0, valid initial state, bounded first-step reward
- **Agent health**: parameter count, device/GPU confirmation
- **Cycle monitoring**: reward explosion, NaN loss, Q-value range, epsilon sanity, buffer overflow timing
- **Unit tests**: 23 pipeline tests in `tests/test_drl_v2.py` (Eq.4 reward, epsilon dynamics, feature causality, training smoke)

DQN stabilizers retained from the paper:
- `[49]` fixed Q-targets, hard target-network copy every `1000` learn steps
- `[18]` Double DQN, online-net argmax with target-net evaluation
- `[50]` Dueling DQN, value and advantage heads

Old directories such as:
- `drl/dqn/models/walkforward/`
- `drl/dqn/models/v2.1/`

may still exist on disk, but they are archive artifacts and are not resolved by
the active default path.

## Shared State Space

Current shared state (enhanced 12D, 2026-05-01):

Evolution: original 9D → pruned 5D → enhanced 11D → enhanced 12D (added gap_overnight).
The 5D→11D expansion addressed Q-value collapse diagnosed in 5D models.
The 12D adds overnight gap for price discovery signal.

- `seq_len = 60`
- `feature_dim = 12` (= `market_feature_dim`, no prev_action channel)
- `state_spec_version = structural_38_enhanced_12d`

| Index | Feature | Formula | Category | Source |
|-------|---------|---------|----------|--------|
| 0 | ret_1d | `r_t / sigma_t` | ultra-short momentum | Close |
| 1 | ret_5d | `(p_t - p_{t-5}) / (sigma_t * sqrt(5))` | short-term momentum | Close |
| 2 | ret_21d | `(p_t - p_{t-21}) / (sigma_t * sqrt(21))` | medium-term momentum | Close |
| 3 | ret_126d | `(p_t - p_{t-126}) / (sigma_t * sqrt(126))` | long-term trend | Close |
| 4 | macd_8_24 | `(EMA_8 - EMA_24) / sigma_63(p)`, then `/ sigma_252(q)` | trend | Close |
| 5 | rsi_5 | `(RSI_5 - 50) / 50` | ultra-short oscillator | Close |
| 6 | rsi_30 | `(RSI_30 - 50) / 50` | oscillator | Close |
| 7 | atr_norm | `TR / ATR_MA(20)` | volatility regime | OHLC |
| 8 | vol_norm | `Volume / Volume_MA(20)` | liquidity | Volume |
| 9 | oi_chg | `ΔOI / |OI_{t-1}|` | positioning flow | OI |
| 10 | drawdown | `(p - max_126d) / max_126d` | risk state | Close |
| 11 | gap_overnight | `(O_t - C_{t-1}) / sigma_t` | overnight gap | OHLC |

### Feature Correlation Analysis (48-contract aggregate, 2026-05-01)

**Highly correlated pairs (|r| > 0.6, 10 pairs):**

| Feature A | Feature B | r |
|-----------|-----------|---|
| ret_21d | macd_8_24 | +0.881 |
| ret_5d | rsi_5 | +0.872 |
| ret_21d | rsi_30 | +0.863 |
| macd_8_24 | rsi_30 | +0.854 |
| ret_126d | drawdown | +0.766 |
| rsi_30 | drawdown | +0.699 |
| rsi_5 | rsi_30 | +0.697 |
| ret_126d | rsi_30 | +0.692 |
| ret_21d | rsi_5 | +0.665 |
| macd_8_24 | rsi_5 | +0.648 |

**Feature independence ranking (by mean |r| with all other features):**

| Rank | Feature | mean\|r\| | max\|r\| | Assessment |
|------|---------|-----------|----------|------------|
| 1 | oi_chg | 0.024 | 0.074 | Highly independent |
| 2 | vol_norm | 0.051 | 0.277 | Highly independent |
| 3 | atr_norm | 0.092 | 0.277 | Highly independent |
| 4 | gap_overnight | 0.097 | 0.266 | Highly independent |
| 5 | ret_1d | 0.202 | 0.548 | Moderate |
| 6 | ret_126d | 0.290 | 0.766 | Moderate (↔drawdown) |
| 7 | drawdown | 0.313 | 0.766 | Moderate (↔ret_126d) |
| 8 | ret_5d | 0.332 | 0.872 | Redundant cluster |
| 9 | macd_8_24 | 0.378 | 0.881 | Redundant cluster |
| 10 | ret_21d | 0.394 | 0.881 | Redundant cluster |
| 11 | rsi_5 | 0.403 | 0.872 | Redundant cluster |
| 12 | rsi_30 | 0.446 | 0.863 | Most redundant |

**Interpretation:** The 12D set has two clear clusters:
1. **Independent block** (indices 7-11): atr_norm, vol_norm, oi_chg, drawdown, gap — max |r| ≤ 0.28, truly orthogonal signals
2. **Momentum/oscillator cluster** (indices 0-6): ret_1d/5d/21d/126d + macd + rsi_5/30 — internally correlated but captures different time horizons. LSTM can learn to weight these adaptively.

**gap_overnight** adds unique information (mean|r|=0.097, max|r|=0.266 with ret_1d) — validates the feature addition.

### Feature Reconsideration History

**5D → 11D (2026-04-30):**
- Diagnosed Q-value collapse in 5D: Forex model outputs 100% flat on random input
- Root cause: feature redundancy → weak LSTM signal → flat dominates replay → uniform sampling reinforces flat
- Added: ret_5d, ret_126d, rsi_5, atr_norm, vol_norm, oi_chg, drawdown
- Replaced: macd_16_48 → single macd_8_24 (paper uses one pair)

**11D → 12D (2026-05-01):**
- Added gap_overnight: vol-normalized overnight gap `(O_t - C_{t-1}) / sigma_t`
- Rationale: overnight price discovery, gaps indicate news/events (Schwager [42], Murphy [38])
- Correlation validation: 4th most independent feature, no new high-corr pairs introduced

Removed from original 9D: norm_price (non-stationary), ret_42d/ret_63d (redundant),
ret_252d (too slow for gamma=0.3), MACD(32,96) (redundant), prev_action (not in paper).

This shared state is meant for `DQN` now, and later `PG / A2C`.

## Eq.4, Sleeve Wealth, and Unified Backtest

Three layers should not be mixed:

**Eq.4 trade-return world**
- additive price differences:
  - `r_t = p_t - p_{t-1}`
- volatility-scaled positions:
  - `sigma_tgt / sigma_t`

**Sleeve/reporting wealth world**
- comparable sleeve wealth can use:
  - `capital0 = p0 * sigma_tgt / sigma0`
- this is not Eq.4 itself

**Current unified backtest**
- computes all final portfolio metrics, including `MDD` and `Calmar`, from the
  same simulated portfolio path

## Current References

- data issues:
  - [docs/data_issues.md](docs/data_issues.md)
- suspicious paper cells:
  - [docs/paper_table_suspicious_cells.md](docs/paper_table_suspicious_cells.md)
- DRL pipeline handoff:
  - [docs/drl_pipeline.md](docs/drl_pipeline.md)
- DQN folder README:
  - [drl/dqn/README.md](drl/dqn/README.md)
- printable A4 structural summary:
  - [docs/structural38_trade_tables_paper_style_a4.png](docs/structural38_trade_tables_paper_style_a4.png)

## Latest Alignment Snapshot

Current active structural-38 long-only baseline:

- trade-world `<=15%`:
  - `28/28`
- unified-path total:
  - `30/36`

### Long-Only Progress Table

This is the current active progress anchor for the repo.

| Scope | Setting | Result |
| --- | --- | --- |
| Baseline | `structural_38` long-only, trade-world Table 3 | `28/28` at `<=15%` |
| Baseline + path metrics | `structural_38` long-only, unified backtest Table 3 | `30/36` at `<=15%` |
| Current interpretation | reproducible local baseline | active mainline |

Current long-only structural-38 Table 3 trade-world comparison:

| Asset | Ours | Paper | `%Err` |
| --- | --- | --- | --- |
| Commodity | `-0.263, +0.385, +0.260, -0.683, -1.009, +0.491, +0.925` | `-0.298, +0.412, +0.258, -0.723, -1.152, +0.473, +0.987` | `11.8, 6.6, 1.0, 5.5, 12.4, 3.9, 6.2` |
| Equity Index | `+0.541, +0.868, +0.682, +0.624, +0.794, +0.547, +0.920` | `+0.504, +0.928, +0.606, +0.543, +0.831, +0.541, +0.928` | `7.4, 6.5, 12.6, 14.8, 4.5, 1.2, 0.9` |
| Fixed Income | `+0.568, +0.889, +0.590, +0.639, +0.962, +0.533, +0.974` | `+0.605, +0.939, +0.561, +0.645, +1.081, +0.515, +1.048` | `6.2, 5.4, 5.2, 1.0, 11.0, 3.4, 7.0` |
| Forex | `-0.179, +0.438, +0.282, -0.409, -0.635, +0.490, +0.972` | `-0.198, +0.472, +0.285, -0.420, -0.696, +0.491, +0.966` | `9.6, 7.3, 1.0, 2.6, 8.8, 0.1, 0.6` |

Metric order:
- `E(R), std(R), DD, Sharpe, Sortino, % +ve, Ave P/L`

Current long-only structural-38 Table 3 unified-path extension:

| Asset | MDD / Calmar Ours | MDD / Calmar Paper | `%Err` |
| --- | --- | --- | --- |
| Commodity | `+0.127, -0.090` | `+0.248, -0.130` | `48.8, 30.5` |
| Equity Index | `+0.112, +0.368` | `+0.127, +0.466` | `11.5, 21.0` |
| Fixed Income | `+0.214, +0.444` | `+0.108, +0.455` | `98.0, 2.3` |
| Forex | `+0.259, -0.084` | `+0.219, -0.101` | `18.4, 17.0` |

Most suspicious remaining paper-side cells:
- `Equity Index / Table 2 / DD: 0.606`
- `Equity Index / Table 2 / Sortino: 1.102`
- `Fixed Income / Table 2 / Sortino: 1.180`

## 41/45 Status

`41/45` is retained only as an experimental upper-bound reproducer:
- [tests/run_legacy_41.py](tests/run_legacy_41.py)

It is not part of the active baseline or active DRL interpretation.
