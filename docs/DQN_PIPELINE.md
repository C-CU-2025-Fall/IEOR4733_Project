# DQN Pipeline Architecture

> Single source of truth. All data flow paths, reward calculations, and index conventions are documented here.

## Data Flow

```
CLC CSV files (data/CLC/{TK}_{SOURCE}.CSV)
    │
    ▼
load_clc_full(ticker, source, start_date, anchor_date)
    │
    ├─► baseline_run._prepare_contract_cached()
    │     start_date='2009-01-01' (default)
    │     returns: rd dict {prices[4343], rt[4343], sigma[4343], dates[2269], start=506, t1=2774}
    │     dates = rd['prices'][start:t1+1] dates = test period only
    │     prices/rt/sigma = FULL array from 2009 to ~2026
    │
    └─► prepare_features.py
          start_date = train_start - 1 year (e.g. 2004-01-02 for R1)
          → build_contract_arrays() → DROP first 252 rows burn-in
          → npz: {prices, returns, sigma, features[9D], dates}
          npz covers 2004-12~2026-03 (3786 rows for AN R2)
          ⚠️ npz length ≠ baseline prices length (different start_date)
```

## Data Consistency Status

- **prices are from the same source**: both use `load_clc_full` + same source (RAD/REV)
- **Overlapping dates have identical prices**: diff = 0
- **But npz has more history**: starts from 2004 with burn-in, baseline starts from 2009
- **burn-in must be kept**: 9D features need historical data for EWMA sigma / rolling returns / MACD
- **npz[252:] and baseline features align after test period**: diff < 0.001

## Index Conventions

| Variable | Meaning |
|------|------|
| `rd['prices']` | Full price array (2009~2026), length ~4343 |
| `rd['start']` | Test period start index (≈506, i.e. 2011-01) |
| `rd['t1']` | Test period end index (≈2774, i.e. ~2015 or 2019) |
| `rd['dates']` | Dates for rd['prices'][start:t1+1], length ~2269 |
| `rd['rt']` | Additive returns: prices[t] - prices[t-1], same length as prices |
| npz features | Start from 2004-12, after removing 252-day burn-in, length ~3786 |

**⚠️ Index trap**: `rd['prices'][i]` and `npz['features'][j]` on the same date have **i ≠ j**
- baseline prices[0] = 2009-01-02
- npz features[0] = 2004-12-31 (after burn-in)
- Alignment must be through dates, not indices

## Reward: Paper Eq.4

```
R_t = A_{t-1} × (σ_tgt / σ_{t-1}) × r_t − bp × p_{t-1} × |A_{t-1}×σ_tgt/σ_{t-1} − A_{t-2}×σ_tgt/σ_{t-2}|
```

- `A_{t-1}` = **position held at time t** (decided at t-1)
- `A_{t-2}` = **position held at time t-1** (decided at t-2)
- Even with unchanged position, σ changes generate small TC (vol-scaled position drift)

### Training (ContractEnv.step) — Fixed ✅

```python
# Paper's A_{t-1} = self.last_position, A_{t-2} = self.prev_last_position
compute_eq4_reward(
    idx=self.idx,
    action=self.last_position,         # A_{t-1}
    prev_action=self.prev_last_position, # A_{t-2}
    prev_sigma=self.last_sigma,          # σ_{t-2}
)
```

### Backtest (baseline_run, vectorized) — Always correct ✅

```python
sp = pos[t-1] * sigma_tgt / sigma[t-1]    # A_{t-1} scaled
spp = pos[t-2] * sigma_tgt / sigma[t-2]   # A_{t-2} scaled
```

## Features (9D)

```
spec: structural_38_close_norm_9d
  [0] price_norm = price[t] / (sigma[t] * sqrt(252))
  [1-4] returns: 21d, 42d, 63d, 252d
  [5-7] MACD: 3 pairs averaged
  [8] RSI(30)
```

- Requires 252-day burn-in history, otherwise first 252 rows of feature values are 0/empty
- **Cannot remove burn-in**

## Consistency Requirements

All three paths must use **the same set of prices/returns/sigma** to be meaningful:
1. Long-only baseline backtest → baseline prices ✅
2. DQN backtest reward → baseline prices ✅ (same rd)
3. DQN training reward → npz prices (⚠️ more history, but same source)

Training and backtest rewards are essentially identical in test period (<0.001 diff), because:
- Same price source
- sigma converges after sufficient warm-up

## Hyperparameters (Forex)

```
γ=0.3, ε=0.01 fixed, τ=1000, LR=0.0001, batch=64
memory=5000, dropout=0.2, episodes=200, σ_tgt=0.058
arch: LSTM(9,64)→LSTM(64,32)→FC(32)→Dueling(3)
```

## Checkpoints

```
drl/dqn/models/Forex/r1/20260504T074157/  ← current unique R1
Forex/r2/20260504T074703/  ← current unique R2
```

## Known Issues

| Status | Issue |
|------|------|
| ✅ fixed | Training reward used pos[t] → changed to self.last_position (A_{t-1}) |
| ✅ fixed | Redundant checkpoints cleaned up |
| cosmetic | On-the-fly features have length mismatch, but data is consistent |

## Round Config

```
R1: train 2005-01-01 ~ 2010-12-31, test 2011-01-01 ~ 2015-12-31
R2: train 2005-01-01 ~ 2015-12-31, test 2016-01-01 ~ 2019-12-31
```
