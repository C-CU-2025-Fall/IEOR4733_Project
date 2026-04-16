# IEOR4733_Project — Deep Reinforcement Learning for Trading

Reproduction of **"Deep Reinforcement Learning for Trading"** by Zhang, Zohren, and Roberts (Oxford, 2019)

📄 Paper: https://arxiv.org/pdf/1911.10107

> **🤖 AI Context**: Read [`PROJECT_MEMORY.md`](./PROJECT_MEMORY.md) first for full project state, known issues, and next steps.

---

## Quick Start

```bash
pip install numpy pandas

# Run Table 3 (per-contract vol scaling only)
python baseline_run.py

# Run Table 2 (+ portfolio-level vol scaling)
python baseline_run.py --table 2

# Reproduction reference run used in the current notes
python baseline_run.py --table 3 --all-metrics --sigma 0.058

# Pure trade-world reference for all 9 metrics
python baseline_run.py --table 3 --all-metrics --sigma 0.058 --report-source trade

# Current split-world start:
# keep 7 trade metrics in Eq. 4 / Eq. 13 trade space,
# report only MDD/Calmar from the risk-price-sigma0 sleeve-capital bridge
python baseline_run.py --table 3 --all-metrics --sigma 0.058 \
  --report-source RISK_PRICE_SIGMA0

# Single asset class
python baseline_run.py --asset Commodity --all-metrics

# Validate all 50 contracts
python tests/roll_validation_final.py

# Run per-contract E(R) decomposition
python tests/decomposition_audit.py
```

---

## Methodology

### Paper Equations

**Eq 4 — Trade return per contract:**
```
R_t = A_{t-1} × (σ_tgt / σ_{t-1}) × r_t − bp × p_{t-1} × |Δscaled_pos|
```
- `r_t = p_t − p_{t-1}` (additive profits on RAD prices)
- `σ_{t-1}` = EWMA(60) std of r_t
- `A_{t-1}` = position signal (Long=+1, Sign(R), MACD)
- `bp = 0.0020` (20 bps transaction cost)

**Eq 13 — Portfolio:**
```
R_port = (1/N) × Σ R_i    (equal-weight)
```

### Table 2 vs Table 3
- **Table 3**: Per-contract vol scaling (Eq 4) only
- **Table 2**: Table 3 + portfolio-level vol scaling → target std ≈ 0.97
- **Calmar reporting default**: additive-wealth CAGR / MDD for the trade world

### Current Metric-World Interpretation

- **Trade world**:
  - `A_t` is the position signal / action
  - Eq. 4 `R_t` is a **standardized additive trading reward**, not an equal-dollar return
  - Eq. 13 averages those standardized rewards equally across contracts
  - this is the home for:
    - `E(R)`, `std(R)`, `DD`, `Sharpe`, `Sortino`, `% +ve`, `Ave P/L`
- **Reporting world**:
  - `MDD`, `Calmar`
  - current start point uses sleeve initial capital
    \[
    C_{i,0} = p_{i,0}\times \sigma_{tgt}/\sigma_{i,0}
    \]
  - then
    \[
    w_{i,t}=1+\sum_{s \le t} R_{i,s}/C_{i,0}
    \]
  - and portfolio wealth is the equal-weight average of sleeve wealth paths

---

## Project Structure

```
├── baseline_run.py              # Main entry (Table 2 & 3)
├── config.py                    # Parameters + paper target values + 50 contracts
├── data_loader.py               # CLC data loading (auto RAD_v2 for damaged contracts)
├── strategies.py                # Long / Sign(R) / MACD signals
├── metrics.py                   # 9 portfolio metrics
├── vol_scaling.py               # Volatility scaling utilities
├── indicators.py                # Technical indicators
├── train_dqn_paper_aligned.py   # DQN training (future work)
├── PROJECT_MEMORY.md            # AI context pickup file
│
├── data/CLC/                    # 50 futures contracts (*_RAD.CSV, *_NON.CSV, *_REV.CSV)
├── config/TEMP/                 # ASC files for cross-validation
│
└── tests/                       # Validation & diagnostic scripts
    ├── roll_validation_final.py       # 50-contract RAD cross-validation
    ├── validate_commodity_rad.py      # 26-contract 3-check validation
    ├── test_rad_algorithm.py          # RAD_v2 generation + math proof
    ├── generate_rad_v2_validated.py   # RAD_v2 CSV generator
    ├── decomposition_audit.py         # Per-contract E(R) decomposition
    └── investigate_tc.py              # TC & σ_t diagnostic
```

---

## Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Transaction cost (bp) | 0.0020 | Paper Table 1 |
| EWMA span | 60 | Paper Section 3.2 |
| σ_tgt (CLI default) | 0.0630 | Current live baseline default |
| σ_tgt (current reference comparison) | 0.0580 | Current Table 3 reproduction reference |
| Trading days/year | 252 | Standard |
| Sign(R) lookback | 252 | Paper Eq 10 |
| MACD pairs | (8,24),(16,48),(32,96) | Ref [4] |
| Test period | 2011-2019 | Paper Section 4.1 |

### 50 Contracts

| Asset Class | # | Tickers |
|-------------|---|---------|
| Commodity | 25 | CC,DA,GI,JO,KC,KW,LB,NR,SB,ZA,ZC,ZF,ZG,ZH,ZI,ZK,ZL,ZO,ZP,ZR,ZT,ZU,ZW,ZZ,ZN |
| Equity Index | 11 | CA,EN,ER,ES,LX,MD,SC,SP,XU,XX,YM |
| Fixed Income | 5 | DT,FB,TY,UB,US |
| Forex | 9 | AN,BN,CN,DX,FN,JN,MP,NK,SN |

> ZN = 24HR NATL GAS (Natural Gas), not 10-Year T-Note.

### Excluded Contracts (Current Live Baseline)

**Current**: `US` only
- live baseline keeps `US` out for now because of RAD damage / repaired-series uncertainty
- older exclusion frontiers are now treated as historical search states, not current doctrine

---

## Data Validation

### RAD Cross-Validation (2026-04-14)

**50/50 contracts validated** via deterministic roll detection:
- 27 VERIFIED (ASC cross-checked)
- 23 CROSS_VALIDATED (REV adj noise = exact 0)

### 4 RAD_v2 Contracts (damaged vendor RAD repaired)

| Contract | Issue | Fix |
|----------|-------|-----|
| ZH | Vendor RAD all-zero | RAD_v2 from REV+NON |
| ZU | Vendor RAD all-zero | RAD_v2 from REV+NON |
| US | Vendor RAD 99% NaN | RAD_v2 from REV+NON |
| ZN | Vendor RAD quarterly only | RAD_v2 from REV+NON |

**RAD_v2 algorithm**: detect rolls (REV adj_change≠0) → compute ratio=NON[t]/NON[t+1] → forward-accumulate → RAD=NON×cum_ratio. Math proof: non-roll returns match NON exactly (corr=1.000).

### CLC Data Relationships

```
NON = raw prices              RAD = NON × cum_ratio  (forward, paper uses this)
ASC = vendor roll records     REV = NON + cum_adj    (backward, for validation)
```

---

## Current Results (2026-04-16)

This README now keeps only the **live baseline state** and the current split-world start point. Older failed global reporting attempts are preserved in `PROJECT_MEMORY.md`, not here.

### Live Table 3 Reference Run

Reference command:

```bash
python baseline_run.py --table 3 --all-metrics --sigma 0.058
```

Current live setup behind that run:

- exclusions: none
- aggregation: `variable_n`
- `sigma_tgt = 0.058` for the reference comparison
- live source overrides: `25`
- default split:
  - trade lane for `E(R), std(R), DD, Sharpe, Sortino, % +ve, Ave P/L`
  - `RISK_PRICE_SIGMA0` reporting lane for `MDD, Calmar`
- default `Calmar`: additive-wealth `CAGR / MDD`

Current default full-9 score:

- `<=10%: 25/45`
- `<=15%: 34/45`

| Asset Class | # | E(R) | Paper | Sharpe | Paper | DD | Paper | MDD | Paper | n10 | n15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Commodity | 25 | -0.264 | -0.298 | -0.678 | -0.723 | 0.265 | 0.258 | 0.209 | 0.248 | 5 | 7 |
| Equity Index | 11 | +0.523 | +0.504 | +0.623 | +0.543 | 0.659 | 0.606 | 0.144 | 0.127 | 6 | 8 |
| Fixed Income | 5 | +0.478 | +0.605 | +0.560 | +0.645 | 0.556 | 0.561 | 0.123 | 0.108 | 4 | 6 |
| Forex | 9 | -0.177 | -0.198 | -0.419 | -0.420 | 0.273 | 0.285 | 0.220 | 0.219 | 6 | 9 |
| All | 50 | +0.033 | -0.013 | +0.098 | -0.036 | 0.236 | 0.230 | 0.121 | 0.037 | 4 | 4 |

### Reporting-Lane Options

Useful commands:

```bash
# current start point
python baseline_run.py --table 3 --all-metrics --sigma 0.058 --report-source RISK_PRICE_SIGMA0

# pure trade-world reference
python baseline_run.py --table 3 --all-metrics --sigma 0.058 --report-source trade

```

Current reading:

- `RISK_PRICE_SIGMA0` is the first split-world bridge that improves the full Table 3 total cleanly:
  - pure trade world: `24/45`, `33/45`
  - `RISK_PRICE_SIGMA0`: `25/45`, `34/45`
- it is still a starting point, not a solved endpoint

### Current TODO

- `Calmar` is still not convincingly recovered.
- The new `RISK_PRICE_SIGMA0` bridge materially improves `MDD`, but `Calmar` still looks like a partially unresolved reporting-definition problem rather than a finished solution.
- So the next search focus should be:
  - keep the current split between trade metrics and reporting `MDD/Calmar`
  - continue tuning the trade world from this start point
  - separately re-audit the `Calmar` numerator/definition instead of treating it as solved

---

## Replication Notes — 复现要点

### 1. Portfolio 聚合：mean() 不做 dropna

Eq 13 的等权组合用 `mean(axis=1)` 而不是 `dropna().mean()`。不同合约在不同交易所，假日不同，dropna 只保留所有合约都有数据的日期，会丢失大量交易日。无 dropna 的 mean 自动跳过 NaN，等价于每天的合约数 N 可以不同。

```
# 正确：每天只平均有数据的合约
Rp = df.T.mean(axis=1)

# 错误：丢弃任何合约缺数据的日期
Rp = df.T.dropna().mean(axis=1)  # ← 不要用
```

### 2. σ_tgt 不影响 Sharpe/Sortino/P/L/+ve

σ_tgt 只同比例缩放 E(R) 和 std(R)，不影响比率型指标（Sharpe, Sortino, Ave P/L）和方向型指标（% +ve）。调整 σ_tgt 无法独立修复 E(R) 偏差——它同时拉高或拉低 E(R) 和 std(R)。

### 3. p_0 归一化在 additive framework 下等价

用测试期起始价格 p_0 归一化后，r_t/p_0 和 σ/p_0 同比例缩放，σ_tgt/σ 不变，TC 项也不变。数学证明：所有 Eq 4 的项都含 p_0 且相互抵消。因此代码使用原始价格（不归一化）。

⚠️ **但 p0=prices[0]（序列开头）归一化和 p0=prices[test_start]（测试期开头）不等价**——因为 EWMA 从序列开头累积，norm_p 在测试期不是 1.0，改变了 σ 的绝对水平和 TC 的量纲。旧版用 prices[0] 归一化时 Equity E(R) 误差 3-5%，新版用 raw price 时 15%。两者都是合法实现，我们采用 raw price（更简单，且和论文符号一致）。

### 4. TC 公式验证

论文 Eq 4 的交易成本项是 `bp × |p_{t-1}| × |Δscaled_pos|`。实验对比：
- `bp × |p_{t-1}| × |Δpos|` → Forex E(R) 9.1% 误差 ✅
- `bp × |Δpos|`（去掉 |p|）→ Forex E(R) 72.7% 误差 ❌

确认论文的 TC 确实包含价格水平项。

### 5. Yahoo Finance 不是替代数据源

实测 Yahoo Finance 期货数据 = CLC NON（未调整价格），pct_corr=0.96~0.99。YF 不是连续合约，不能替代 RAD。

---

## Problem Contracts — 5 个问题合约

> Historical diagnostic note from the earlier static-baseline phase.
> The current working frontier no longer uses this exact “exclude 5” setup; see the current-results section above.

| 合约 | 问题 | 所有数据源 E(R) |
|------|------|----------------|
| CC | RAD E(R)=-0.053 vs 论文方向负 | NON=+0.011, REV=-0.027, RAD=-0.053, YF=+0.031 |
| LB | Long 策略就是正收益 | NON=+0.280, REV=+0.177, RAD=+0.165, YF=+0.260 |
| JO | E(R) 在零附近，数据源敏感 | NON=-0.123, REV=-0.028, RAD=-0.044, YF=-0.140 |
| ZO | ASC 不覆盖测试期（0天） | NON=-0.056, REV=+0.049, RAD=+0.047, YF=-0.026 |
| ZH | vendor RAD all-zero | NON=-0.213, REV=-0.241, RAD=-0.263, YF=-0.153 |

**共同特征**：Long E(R) 在零附近，无论用哪个数据源都无法对齐论文。这是数据特征，不是数据质量。

排除 5 个后 Commodity E(R) 从 22.8% 降到 5.7%，n15 从 3 提升到 5。

---

## Known Issues

1. **MDD 计算异常**：additive wealth 可能为负 → MDD 爆炸（Commodity 2.3 vs 0.25）。论文可能用 multiplicative wealth。
2. **FI E(R) 偏差 ~14%**：5 个 FI 合约数据路径不同，无法通过参数修复。
3. **Equity E(R) 偏差 ~15%**：raw price 方案下 E(R) 偏高。若用 p0=prices[0] 归一化可降到 3-5%，但会影响其他资产类别。
4. **Sign(R) / MACD**: 尚未跑。

---

## TODO

- [x] 50/50 contract RAD cross-validation
- [x] 4 damaged contracts repaired (RAD_v2)
- [x] Baseline Long strategy framework
- [x] Data source comparison (NON/REV/RAD/YF/YF_RAD)
- [x] TC formula verification
- [x] p_0 normalization analysis
- [x] Table 2 & Table 3 baseline runner (baseline_run.py)
- [x] Move from static RAD-only baseline to source-aware Table 3 frontier
- [x] All portfolio row added
- [ ] Run Sign(R) and MACD baselines
- [ ] DQN training and comparison
- [ ] Final presentation
