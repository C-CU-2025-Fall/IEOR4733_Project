# DQN Pipeline Architecture

> 单一事实来源。所有数据流经路径、reward 计算、索引约定都在这里。

## Data Flow

```
CLC CSV files (data/CLC/{TK}_{SOURCE}.CSV)
    │
    ▼
load_clc_full(ticker, source, start_date, anchor_date)
    │
    ├─► baseline_run._prepare_contract_cached()
    │     start_date='2009-01-01' (默认)
    │     返回: rd dict {prices[4343], rt[4343], sigma[4343], dates[2269], start=506, t1=2774}
    │     dates = rd['prices'][start:t1+1] 的日期 = test period only
    │     prices/rt/sigma = FULL array from 2009 to ~2026
    │
    └─► prepare_features.py
          start_date = train_start - 1 year (e.g. 2004-01-02 for R1)
          → build_contract_arrays() → DROP 前 252 行 burn-in
          → npz: {prices, returns, sigma, features[9D], dates}
          npz 覆盖 2004-12~2026-03 (3786 rows for AN R2)
          ⚠️ npz 长度 ≠ baseline prices 长度 (不同 start_date)
```

## 数据一致性现状

- **prices 是同一来源**: 都用 `load_clc_full` + 同一个 source (RAD/REV)
- **重叠日期价格完全相同**: diff = 0
- **但 npz 更多历史**: 从 2004 起含 burn-in, baseline 从 2009 起
- **burn-in 必须保留**: 9D features 需要历史数据计算 EWMA sigma / rolling returns / MACD
- **npz[252:] 和 baseline 对齐后 features 一致**: test period 差异 < 0.001

## Index Conventions

| 变量 | 含义 |
|------|------|
| `rd['prices']` | 完整价格数组 (2009~2026), 长度 ~4343 |
| `rd['start']` | test period 起始索引 (≈506, 即 2011-01) |
| `rd['t1']` | test period 结束索引 (≈2774, 即 ~2015 或 2019) |
| `rd['dates']` | rd['prices'][start:t1+1] 的日期, 长度 ~2269 |
| `rd['rt']` | 加法收益: prices[t] - prices[t-1], 与 prices 同长 |
| npz features | 从 2004-12 开始, 去掉 252 天 burn-in, 长度 ~3786 |

**⚠️ 索引陷阱**: `rd['prices'][i]` 和 `npz['features'][j]` 在同一日期的 **i ≠ j**
- baseline prices[0] = 2009-01-02
- npz features[0] = 2004-12-31 (burn-in 后)
- 对齐必须通过日期，不是索引

## Reward: Paper Eq.4

```
R_t = A_{t-1} × (σ_tgt / σ_{t-1}) × r_t − bp × p_{t-1} × |A_{t-1}×σ_tgt/σ_{t-1} − A_{t-2}×σ_tgt/σ_{t-2}|
```

- `A_{t-1}` = **t 时刻持有的仓位**（t-1 时刻决定的）
- `A_{t-2}` = **t-1 时刻持有的仓位**（t-2 时刻决定的）
- 即使仓位不变，σ 变化也会产生微小 TC（vol-scaled position drift）

### Training (ContractEnv.step) — 已修复 ✅

```python
# Paper's A_{t-1} = self.last_position, A_{t-2} = self.prev_last_position
compute_eq4_reward(
    idx=self.idx,
    action=self.last_position,         # A_{t-1}
    prev_action=self.prev_last_position, # A_{t-2}
    prev_sigma=self.last_sigma,          # σ_{t-2}
)
```

### Backtest (baseline_run, vectorized) — 始终正确 ✅

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

- 需要 252 天 burn-in 历史，否则前 252 行 feature 值为 0/空
- **不能去掉 burn-in**

## 一致性要求

三条路径必须用**同一组 prices/returns/sigma** 才有意义:
1. Long-only baseline 回测 → baseline prices ✅
2. DQN 回测 reward → baseline prices ✅ (同一个 rd)
3. DQN 训练 reward → npz prices (⚠️ 更多历史，但同一来源)

训练和回测 reward 在 test period 基本一致 (<0.001 diff), 因为:
- 同一价格源
- sigma 经过足够 warm-up 后收敛

## Hyperparameters (Forex)

```
γ=0.3, ε=0.01 fixed, τ=1000, LR=0.0001, batch=64
memory=5000, dropout=0.2, episodes=200, σ_tgt=0.058
arch: LSTM(9,64)→LSTM(64,32)→FC(32)→Dueling(3)
```

## Checkpoints

```
drl/dqn/models/Forex/r1/20260504T074157/  ← 当前唯一 R1
Forex/r2/20260504T074703/  ← 当前唯一 R2
```

## Known Issues

| 状态 | 问题 |
|------|------|
| ✅ fixed | Training reward 用 pos[t] → 改用 self.last_position (A_{t-1}) |
| ✅ fixed | 多余 checkpoint 清理 |
| cosmetic | on-the-fly features 因长度不匹配, 但数据一致 |

## Round Config

```
R1: train 2005-01-01 ~ 2010-12-31, test 2011-01-01 ~ 2015-12-31
R2: train 2005-01-01 ~ 2015-12-31, test 2016-01-01 ~ 2019-12-31
```
