# IEOR4733_Project — Deep Reinforcement Learning for Trading

Reproduction of **"Deep Reinforcement Learning for Trading"** by Zhang, Zohren, and Roberts (Oxford, 2019)

📄 Paper: https://arxiv.org/pdf/1911.10107

---

## Quick Start

```bash
# Install dependencies
pip install numpy pandas

# Run Table 3 (per-contract vol scaling only)
python baseline_run.py

# Run Table 2 (+ portfolio-level vol scaling)
python baseline_run.py --table 2

# Run both tables
python baseline_run.py --table both

# Single asset class
python baseline_run.py --asset "Equity Index"

# Custom σ_tgt
python baseline_run.py --sigma 0.058

# Custom test period
python baseline_run.py --test-start 2015-01-01 --test-end 2019-12-31

# Run tests
python test_baseline.py
```

---

## Methodology

### Paper Equations (as implemented)

**Eq 4 — Trade return per contract:**
```
R_t = A_{t-1} × (σ_tgt / σ_{t-1}) × r_t − bp × p_{t-1} × |Δscaled_pos|
```
- `r_t = p_t − p_{t-1}` (additive profits, p0-normalized) [Paper Section 3.2]
- `σ_{t-1}` = EWMA(60) std of r_t [Paper Section 3.2]
- `A_{t-1}` = position signal at t-1 (no look-ahead)
- `bp = 0.0020` [Paper Table 1]
- `σ_tgt` = volatility target (free parameter)

**Eq 10 — Sign(R) signal:**
```
A_t = sign(r_{t-252:t}) = sign(p_t − p_{t-252})
```

**Eq 3,11,12 — MACD signal:**
```
MACD_t = q_t / std(q_{t-252:t})
q_t = (m(S) − m(L)) / std(p_{t-63:t})
A_t = φ(MACD_t) where φ(x) = x·exp(−x²/4)/0.89
```
Time-scale pairs: (8,24), (16,48), (32,96) [Paper reference [4] Baz et al.]

**Eq 13 — Portfolio:**
```
R_port = (1/N) × Σ R_i    (equal-weight average)
```

### Table 2 vs Table 3

- **Table 3**: Per-contract vol scaling (Eq 4) only. Each contract scaled to σ_tgt, then averaged.
- **Table 2**: Table 3 + additional portfolio-level vol scaling → target std ≈ 0.97

### Metrics (9 per strategy per asset class)

1. E(R) = mean(R) × 252
2. std(R) = std(R) × √252
3. DD = √(mean(min(0,R)²)) × √252
4. Sharpe = E(R) / std(R)
5. Sortino = E(R) / DD
6. MDD = max((peak − wealth) / peak)  [running max method]
7. Calmar = realised_annual_return / MDD
8. %+ve = fraction of positive days
9. Ave P/L = mean(R>0) / |mean(R<0)|

Wealth = N × W₀ + cumsum(R_port) (additive accumulation, not multiplicative)

---

## Project Structure

```
IEOR4733_Project/
├── baseline_run.py          # Main entry point (Table 2 & 3)
├── test_baseline.py         # Tests
├── config.py                # Parameters + paper target values
├── data_loader.py           # CLC data loading
├── strategies.py            # Long / Sign(R) / MACD signals
├── metrics.py               # 9 portfolio metrics
├── vol_scaling.py           # Volatility scaling utilities
├── indicators.py            # Technical indicators (MACD, RSI, etc.)
├── train_dqn_paper_aligned.py  # DQN training (future work)
│
├── data/
│   ├── CLC/                 # 96 futures contracts (*_RAD.CSV)
│   ├── index_data.csv       # VIX index
│   └── risk_free_rate.csv   # Risk-free rate (DTB3)
│
├── references/              # Paper PDF
├── deck.md                  # Proposal deck
├── deck-v1.1.pptx           # Presentation v1.1
├── deck-v2.pptx             # Presentation v2
├── DRL_Trading_Midterm_draft.pptx
└── archive/                 # Old scripts (13 files)
```

---

## Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Transaction cost (bp) | 0.0020 | Paper Table 1 |
| EWMA span | 60 | Paper Section 3.2 |
| σ_tgt (per contract) | 0.064 | Derived from Long std match |
| Trading days/year | 252 | Standard |
| Sign(R) lookback | 252 | Paper Eq 10 |
| MACD pairs | (8,24),(16,48),(32,96) | Paper Eq 12, ref [4] |
| MACD vol window | 63 | Paper Eq 3, ref [4] |
| MACD std window | 252 | Paper Eq 3, ref [4] |
| Portfolio vol target | 0.97 | Paper Table 2 |
| Test period | 2011-2019 (9 years) | Paper Section 4.1 |
| Discount factor (γ) | 0.3 | Paper Table 1 (RL only) |
| Retrain interval | 5 years | Paper Section 4.1 (RL only) |

### Asset Classes & Contracts (50 contracts total)

| Asset Class | # | Tickers |
|-------------|---|---------|
| Commodity | 25 | CC,DA,GI,JO,KC,KW,LB,NR,SB,ZA,ZC,ZF,ZG,ZH,ZI,ZK,ZL,ZO,ZP,ZR,ZT,ZU,ZW,ZZ,**ZN** |
| Equity Index | 11 | CA,EN,ER,ES,LX,MD,SC,SP,XU,XX,YM |
| Fixed Income | 5 | DT,FB,TY,UB,US |
| Forex | 9 | AN,BN,CN,DX,FN,JN,MP,NK,SN |

**Note**: ZN in CLC data = 24HR NATL GAS (Natural Gas), not 10-Year T-Note. Moved to Commodity.

### RAD Data Status (deterministic cross-validation, 2026-04-13)

**Method**: REV adj 是分段常数 → `adj_change ≠ 0` = 确定性 roll 检测（无阈值）

**Vendor RAD 可用**: 46/50
**Vendor RAD 损坏**: 4/50 → 使用 RAD_v2

| 损坏合约 | 问题 | Roll Rule | CME 验证 |
|---------|------|-----------|---------|
| ZH | 全零 | H<M>UZ | ✅ |
| ZU | 全零 | 12个月 | ✅ |
| US | 99% NaN | H<M>UZ | ✅ |
| ZN | 只有季度调整(缺月度) | ALL<K> (12月/年) | ✅ |

**`data_loader.py`**: ZH, ZU, US, ZN → 使用 `*_RAD_v2.CSV`

### 交叉验证结果 (`tests/roll_validation_final.py`)

| 状态 | # | 说明 |
|------|---|------|
| ✅ VERIFIED | 27 | 有 ASC，ASC vs REV price error <0.25% (mean 0.01%) |
| ✅ CROSS_VALIDATED | 21 | 无 ASC，REV adj 非roll日噪声 = 精确 0 |
| ❌ INCOMPLETE | 1 (ZN) | vendor RAD 只有季度调整 |
| ❌ CORRUPT | 1 (US) | vendor RAD 99% NaN |

- REV adj 非roll日噪声 = 精确 0: **50/50**
- ASC vs REV price error: mean=0.01%, max=0.23%
- CLC roll rules vs CME 官方: **全部一致**

---

## 📊 CLC 数据关系

```
NON = 纯原始价格（无 roll 信息）
ASC = NON + vendor roll_date + prev_close + new_close（版本1）
  → NON[roll_date] = prev_close（旧合约收盘价）
  → 次日 RAD ratio 跳变, REV adj 跳变
RAD = NON × cumulative_ratio (forward adjustment, 论文用这个)
REV = NON + cumulative_adj (backward adjustment, 独立验证源)

任意两源可精确推导 roll_date + prev_close + new_close
REV adj 是分段常数 → adj_change ≠ 0 就是确定性 roll 检测
```

---

## Current Results

### Full Baseline Alignment (Table 3, σ_tgt = 0.064, 2011-2019)

**Long Only 4×9 (Tier A metrics): ≤10%: 18/36 (50%) | ≤15%: 22/36 (61%)**

*Note: ZN moved from Fixed Income to Commodity (ZN in CLC data is Natural Gas). Fixed Income: 5 contracts, Commodity: 25 contracts.*

**Key Changes (2026-04-13)**:
- `fillna(0)` for holidays: On exchange holidays, missing contracts contribute R_t=0 to portfolio average, keeping denominator = N contracts every day.
- TC formula: `bp × p_{t-1} × |Δsp|` uses raw prices (paper Eq 4 literal implementation).
- p0 normalization removed: It's a no-op for ratio metrics (Sharpe, etc.).

**DD Fix (2026-04-12)**: DD calculation updated per Paper Section 4.4 definition (std of negative returns only).
Improved DD ≤15%: 6/12 → 7/12.

---

### Commodity (25 contracts)

**Note**: ZN added (moved from Fixed Income — ZN in CLC data is Natural Gas, RAD regenerated using RAD_v2 method).

| Strategy | E(R) | std | DD | Sharpe | Sortino | MDD | Calmar | %+ve | AveP/L | ≤10% | ≤15% |
|----------|------|-----|----|--------|---------|-----|--------|------|--------|----|----|
| **Long** | -0.237 | 0.412 | **0.258** | -0.574 | -0.886 | 0.102 | -0.088 | 0.497 | 0.930 | 4/9 | 4/9 |
| Paper | -0.298 | 0.412 | **0.258** | -0.723 | -1.152 | 0.248 | -0.130 | 0.473 | 0.987 | | |
| **%Err** | 20.6% | **0.0%** | **0.0%** | 20.6% | 23.1% | 58.9% | 32.3% | **5.1%** | **5.8%** | | |
| **Sign(R)** | -0.043 | 0.307 | 0.219 | -0.139 | -0.195 | 0.047 | -0.036 | 0.496 | 0.994 | 3/9 | 3/9 |
| **MACD** | -0.178 | 0.237 | 0.175 | -0.751 | -1.017 | 0.064 | -0.115 | 0.483 | 0.945 | 3/9 | 3/9 |

---

### Equity Index (11 contracts) ✅

| Strategy | E(R) | std | DD | Sharpe | Sortino | MDD | Calmar | %+ve | AveP/L | ≤10% | ≤15% |
|----------|------|-----|----|--------|---------|-----|--------|------|--------|----|----|
| **Long** | +0.555 | 0.917 | 0.695 | +0.605 | +0.799 | 0.113 | +0.412 | 0.547 | 0.910 | **5/9** | **8/9** |
| Paper | +0.504 | 0.928 | 0.606 | +0.543 | +0.831 | 0.127 | +0.466 | 0.541 | 0.928 | | |
| **%Err** | **10.1%** | **1.2%** | 14.7% | **11.4%** | 3.8% | 11.0% | 11.6% | **1.1%** | **1.9%** | | |
| **Sign(R)** | -0.050 | 0.791 | 0.598 | -0.063 | -0.084 | 0.222 | -0.023 | 0.516 | 0.929 | 3/9 | 4/9 |
| **MACD** | -0.252 | 0.617 | 0.461 | -0.409 | -0.547 | 0.270 | -0.087 | 0.502 | 0.921 | 3/9 | 3/9 |

---

### Fixed Income (5 contracts)

**Note**: ZN moved to Commodity category (ZN in CLC data is Natural Gas, not 10-Year T-Note).

| Strategy | E(R) | std | DD | Sharpe | Sortino | MDD | Calmar | %+ve | AveP/L | ≤10% | ≤15% |
|----------|------|-----|----|--------|---------|-----|--------|------|--------|----|----|
| **Long** | +0.458 | **0.925** | 0.629 | +0.495 | +0.730 | 0.285 | +0.219 | 0.524 | 0.956 | 3/9 | 4/9 |
| Paper | +0.605 | **0.939** | 0.561 | +0.645 | +1.081 | 0.108 | +0.455 | 0.515 | 1.048 | | |
| **%Err** | 24.3% | **1.5%** | 12.1% | 23.3% | 32.5% | 163.9% | 51.9% | **1.7%** | **8.8%** | | |
| **Sign(R)** | -0.361 | **0.803** | 0.564 | -0.450 | -0.640 | 0.829 | -0.086 | 0.500 | 0.926 | 3/9 | 4/9 |
| **MACD** | -0.562 | **0.640** | 0.434 | -0.879 | -1.295 | 1.066 | -0.105 | 0.456 | 1.023 | 3/9 | 3/9 |

---

### Forex (9 contracts)

| Strategy | E(R) | std | DD | Sharpe | Sortino | MDD | Calmar | %+ve | AveP/L | ≤10% | ≤15% |
|----------|------|-----|----|--------|---------|-----|--------|------|--------|----|----|
| **Long** | -0.213 | 0.458 | **0.299** | -0.465 | -0.713 | 0.321 | -0.086 | 0.490 | 0.953 | 4/9 | 5/9 |
| Paper | -0.198 | 0.472 | **0.285** | -0.420 | -0.696 | 0.219 | -0.101 | 0.491 | 0.966 | | |
| **%Err** | **7.7%** | **3.0%** | **4.9%** | 10.7% | 2.4% | 46.6% | 14.9% | **0.2%** | **1.3%** | | |
| **Sign(R)** | -0.370 | 0.550 | 0.407 | -0.673 | -0.911 | 0.406 | -0.101 | 0.483 | 0.954 | 3/9 | 3/9 |
| **MACD** | -0.364 | 0.446 | 0.329 | -0.815 | -1.106 | 0.382 | -0.106 | 0.470 | 0.976 | 3/9 | 3/9 |

---

### Key Findings

**✅ Long Only 4×9 Summary**:
- **≤10%: 18/36 (50%) | ≤15%: 22/36 (61%)**
- Equity: 5/9 ≤10%, Forex: 4/9 ≤10%, Commodity: 4/9 ≤10%, Fixed Income: 3/9 ≤10%

**✅ Volatility Scaling Verified**:
- **All asset classes std(R) error <15%** (Commodity 0.0%, Equity 1.2%, Fixed Income 1.5%, Forex 3.0%)
- → **Equation 4 implementation correct**

**✅ Holiday Handling **(fillna(0))
- On exchange holidays, missing contracts contribute R_t=0 to portfolio average
- Keeps denominator = N contracts every day (avoids artificial amplification)
- std, %+ve, P/L fully aligned; E(R) bias is data-level, not methodology

**⚠️ E(R) Bias Analysis **(Commodity 21%, Fixed Income 23%)
- **TC drag analysis**: Fixed Income TC = -0.243 (6× Commodity), due to low σ → high |Δsp|
- **Return-only E(R)**: Commodity -0.195 vs paper -0.298; Fixed Income +0.710 vs paper +0.605
- **Conclusion**: Bias is data-level (CLC RAD generation, vendor differences), not code logic
- TC formula: raw_p (paper Eq 4 literal) is best compromise; norm_p breaks 3/4 asset classes

**📊 Data Quality Impact** (from cross-validation v3):
- **Equity Index**: 8/11 A/B grade → Best replication ✅
- **Forex**: 7/9 A/B grade → std precise, E(R) bias
- **Commodity**: Partial C grade → Larger errors
- **Fixed Income**: **ZN excluded** (data corrupted); remaining 5 contracts: std ✅ (0.9%), E(R) bias

---

### All Contracts (50 contracts combined)

*No Paper comparison — for reference only*

| Strategy | E(R) | std | DD | Sharpe | Sortino | MDD | Calmar | %+ve | AveP/L |
|----------|------|-----|----|--------|---------|-----|--------|------|--------|
| **Long** | +0.022 | 0.348 | 0.241 | +0.064 | +0.093 | 0.029 | +0.016 | 0.517 | 0.944 |
| **Sign(R)** | -0.125 | 0.288 | 0.208 | -0.435 | -0.601 | 0.029 | -0.086 | 0.501 | 0.926 |
| **MACD** | -0.268 | 0.232 | 0.164 | -1.154 | -1.639 | 0.045 | -0.120 | 0.467 | 0.936 |

---

## 📊 CLC 数据文件概念图

### 核心关系：Roll Date 是唯一真相

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Roll Date (唯一)                                │
│   换月发生的真实日期 - 这是所有验证的核心                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
          ▼                         ▼                         ▼
   ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
   │    ASC      │          │    NON      │          │  理论规则   │
   │  (黄金标准)  │          │  (原始价格)  │          │  (MPDM_N)  │
   │             │          │             │          │             │
   │ 记录:       │          │ 用于:       │          │ 用于:       │
   │ - roll_date │          │ - 检测价格跳 │          │ - 推测日期  │
   │ - prev_close│          │ - 推导 ratio │          │ - 无 ASC 时  │
   │ - new_close │          │ - 推导 adj   │          │   做 backup │
   └─────────────┘          └─────────────┘          └─────────────┘
          │                         │
          └─────────────┬───────────┘
                        │
                        ▼
        ┌───────────────────────────────────┐
        │        推导调整参数 (唯一)          │
        │                                   │
        │  ratio = prev_close / new_close   │
        │  adjustment = new_close - prev_close │
        └───────────────────────────────────┘
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
   ┌─────────────┐            ┌─────────────┐
   │    RAD      │            │    REV      │
   │  (乘法调整)  │            │  (加法调整)  │
   │             │            │             │
   │ RAD = NON   │            │ REV = NON   │
   │     × ratio │            │     + adj   │
   │             │            │             │
   │ 用于回测     │            │ 用于回测     │
   └─────────────┘            └─────────────┘
```

### 关键理解

1. **Roll Date 只有一个** - ASC 记录的是真相，NON 价格跳变可以检测，理论规则可以推测
2. **REV 和 RAD 是独立的** - 它们用不同方法调整，不是等价的，不需要互相验证
3. **ASC + NON → 推导调整参数** - ratio 和 adjustment 都从 ASC 的价格记录推导
4. **理论规则是 backup** - 当没有 ASC 时，用 MPDM 规则 + NON 价格跳变来推测 roll date
5. **数据完整 = 4 类齐全** - ASC + NON + REV + RAD 同时存在才能完整验证

### 交叉验证矩阵 (修正版)

| 验证目标 | 数据源 A | 数据源 B | 验证方法 | 期望 |
|---------|---------|---------|---------|------|
| **Roll Date 真相** | ASC | NON (跳变) | ASC 日期 vs 检测跳变日 | 100% 匹配 |
| **Roll Price 真相** | ASC | NON | prev_close/new_close 对比 | <0.1% 差异 |
| **理论规则可靠性** | 理论规则 | ASC | 理论日期 vs ASC 日期 | ≤1 天差异 |
| **RAD 推导正确** | ASC+NON | RAD | 推导 ratio vs 实际 ratio | 完全匹配 |
| **REV 推导正确** | ASC+NON | REV | 推导 adj vs 实际 adj | 完全匹配 |
| **数据完整性** | 4 类文件 | - | ASC/NON/REV/RAD 都存在 | A 类完整 |

### 50 合约数据状态 (修正版)

| 类别 | 合约数 | 文件齐全 | Roll Date 来源 | 验证状态 |
|------|--------|---------|---------------|----------|
| **A 类 (完整)** | ~20 | ASC+NON+REV+RAD | ASC (黄金标准) | ✅ 可完整验证 |
| **B 类 (部分)** | ~15 | NON+RAD(+REV?) | NON 跳变 + 理论规则 | ⚠️ 部分验证 |
| **C 类 (仅 NON)** | ~11 | NON only | 理论规则 (backup) | ❌ 无法验证 |
| **损坏修复** | 4 | NON+RAD_v2 | 理论规则 + 交易日 | ✅ RAD_v2 生成 |

**损坏合约修复 (RAD_v2)**:
- ZH: MPDM_11, H,M,U,Z → 生成成功 (vendor RAD 全零)
- ZN: MPDM_24, H,M,U,Z → 生成成功 (vendor RAD 21.9x 异常)
- ZU: MPDM_11, 12 个月份 → 生成成功 (vendor RAD 全零)
- US: MPDM_24, H,M,U,Z → 生成成功 (vendor RAD 全 NaN)
- ZI: ✅ vendor RAD 正常 (ratio=1.32±0.06，不需要 RAD_v2)

---

## 🔍 交叉验证设计

### 阶段 1: Roll Date 验证 (核心)
**目标**: 确认 Roll Date 的唯一真相

| 合约类型 | 验证方法 | 期望 |
|---------|---------|------|
| **A 类 (有 ASC)** | ASC.roll_date vs NON 跳变检测 | 100% 匹配 |
| **A 类 (有 ASC)** | ASC.roll_date vs 理论规则 | ≤1 天差异 |
| **B/C 类 (无 ASC)** | 理论规则 vs NON 跳变检测 | ≤1 天差异 |

**意义**: 验证理论规则作为 backup 的可靠性

### 阶段 2: Roll Price 验证 (核心)
**目标**: 确认 ASC 记录的价格与 NON 一致

| 验证项 | 对比 | 期望 |
|-------|------|------|
| prev_close | ASC.prev_close vs NON[roll_date-1] | <0.1% 差异 |
| new_close | ASC.new_close vs NON[roll_date] | <0.1% 差异 |
| 隐含 ratio | ASC.prev_close / ASC.new_close | 用于验证 RAD |

**意义**: 如果价格对不上，ASC 的可信度降低

### 阶段 3: 调整参数验证
**目标**: 验证 RAD/REV 使用的调整参数正确

| 验证项 | 公式 | 期望 |
|-------|------|------|
| RAD ratio | RAD/NON vs ASC.prev/ASC.new | 完全匹配 |
| REV adj | REV-NON vs ASC.new-ASC.prev | 完全匹配 |

### 阶段 4: 数据完整性评分
**目标**: 对 50 合约进行数据质量分级

| 等级 | 标准 |
|------|------|
| **A** | ASC+NON+RAD+REV 齐全，Roll Date/Price 验证通过 |
| **B** | NON+RAD，Roll Date 理论+NON 跳变一致 |
| **C** | 仅 NON，依赖理论规则 |
| **D** | 数据异常 (如 ZN 的 10000x) |
| **F** | 数据缺失或损坏 |

### 输出
1. `roll_date_truth.csv` - Roll Date 验证结果 (ASC vs NON vs 理论)
2. `roll_price_truth.csv` - Roll Price 验证结果 (ASC vs NON)
3. `adjustment_validation.csv` - RAD/REV 调整参数验证
4. `data_quality_scores.csv` - 50 合约质量评分

---

## TODO

- [x] Deterministic 50-contract RAD cross-validation (2026-04-13) — 48/50 VERIFIED/CROSS_VALIDATED
- [x] ZN moved to Commodity (ZN = Natural Gas in CLC data)
- [x] 4 damaged contracts: ZH, ZU, US (corrupt), ZN (incomplete)
- [x] CLC roll rules verified against CME official
- [x] 4 合约 RAD_v2 已用交叉验证方法论生成 (REV→roll date→ratio→cumulative×NON)
- [x] Roll rules vs REV 检测验证: ZH(12/年), ZU(12/年), ZN(12/年), US(4/年) 全部一致
- [ ] Run Table 2 & Table 3 backtesting with all 50 contracts
- [ ] DQN training and comparison with baselines
- [ ] Final presentation
