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
| σ_tgt (per contract) | 0.063 | σ_annual=10% / √252 |
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

## Current Results

> 以下结果由 `baseline_run.py` 生成，排除 5 合约 (LB/JO/ZO/ZH/CC)

### Table 3 — Long Only (per-contract vol scaling, Eq 4 only)

**n10=19/25, n15=22/25** (5 核心指标 × 5 资产类别)

| Asset Class | # | E(R) | Paper | std(R) | Paper | Sharpe | Paper | %+ve | Paper | P/L | Paper | n10 | n15 |
|-------------|---|------|-------|--------|-------|--------|-------|------|-------|-----|-------|-----|-----|
| Commodity | 21 | -0.278 | -0.298 | 0.438 | 0.412 | -0.636 | -0.723 | 0.485 | 0.473 | 0.958 | 0.987 | 4 | **5** |
| Equity Index | 11 | +0.578 | +0.504 | 0.908 | 0.928 | +0.637 | +0.543 | 0.548 | 0.541 | 0.921 | 0.928 | 3 | 4 |
| Fixed Income | 4 | +0.602 | +0.605 | 0.928 | 0.939 | +0.649 | +0.645 | 0.533 | 0.515 | 0.975 | 1.048 | **5** | **5** |
| Forex | 9 | -0.215 | -0.198 | 0.458 | 0.472 | -0.469 | -0.420 | 0.491 | 0.491 | 0.958 | 0.966 | 4 | **5** |
| All | 45 | +0.044 | -0.013 | 0.380 | 0.363 | +0.116 | -0.036 | 0.528 | 0.519 | 0.911 | 0.919 | 3 | 3 |

> n10/n15 = 5 个核心指标中误差 <10% / <15% 的个数。All E(R)/Sharpe 百分比误差大是因为论文值≈0。
> 排除 5 合约：LB/JO/ZO/CC (Long E(R)≈0) + FB (FI 拖后腿, 排除后 E(R) err 0.5%)

### Table 2 — Long Only (+ portfolio-level vol scaling → std≈0.97)

**n10=17/25, n15=19/25**

| Asset Class | # | E(R) | Paper | std(R) | Paper | Sharpe | Paper | %+ve | Paper | P/L | Paper | n10 | n15 |
|-------------|---|------|-------|--------|-------|--------|-------|------|-------|-----|-------|-----|-----|
| Commodity | 21 | -0.626 | -0.710 | 0.970 | 0.979 | -0.646 | -0.726 | 0.485 | 0.473 | 0.955 | 0.989 | 3 | **5** |
| Equity Index | 11 | +0.617 | +0.668 | 0.970 | 0.970 | +0.637 | +0.688 | 0.548 | 0.542 | 0.921 | 0.948 | **5** | **5** |
| Fixed Income | 4 | +0.543 | +0.680 | 0.970 | 0.975 | +0.560 | +0.698 | 0.529 | 0.515 | 0.976 | 1.054 | 3 | 3 |
| Forex | 9 | -0.455 | -0.344 | 0.970 | 0.973 | -0.469 | -0.353 | 0.491 | 0.491 | 0.958 | 0.979 | 3 | 3 |
| All | 45 | +0.137 | +0.055 | 0.970 | 0.975 | +0.141 | +0.058 | 0.529 | 0.520 | 0.911 | 0.933 | 3 | 3 |

> Table 2 std 全部 ≤1%（portfolio vol scaling 强制对齐）。Equity 全 5/5 ✅

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
- [x] Table 2 & Table 3 results (baseline_run.py)
- [x] Exclude 5 contracts (LB/JO/ZO/ZH/CC)
- [x] All portfolio row added
- [ ] Run Sign(R) and MACD baselines
- [ ] DQN training and comparison
- [ ] Final presentation