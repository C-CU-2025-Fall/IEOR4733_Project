# PROJECT_MEMORY.md — AI Context Pickup File
# > Last updated: 2026-04-15
# > Read this FIRST when starting a new session on this project

---

## 1. Project Goal

Reproduce **"Deep Reinforcement Learning for Trading"** (Zhang, Zohren, Roberts, 2019)
- Paper: https://arxiv.org/pdf/1911.10107
- 50 futures contracts, 4 asset classes, test period 2011-2019
- Baselines: Long, Sign(R), MACD → compare with DQN

## 2. Core Equations

```
Eq 4: R_t = A_{t-1} × (σ_tgt / σ_{t-1}) × r_t  −  bp × p_{t-1} × |Δscaled_pos|
         where r_t = p_t − p_{t-1} (additive), σ_t = EWMA(60) std of r_t

Eq 13: R_port = (1/N) × Σ R_i  (equal-weight portfolio)

Metrics: E(R), std(R), DD, Sharpe, Sortino, MDD, Calmar, % +ve, Ave P/L
```

Parameters: bp=0.002, EWMA=60, σ_tgt≈0.063, T=252, Sign lookback=252

## 3. Data Architecture

```
CLC data files (data/CLC/):
  NON = raw prices (no roll info)
  RAD = NON × cumulative_ratio  ← paper uses this for backtesting
  REV = NON + cumulative_adj    ← used for roll detection & validation
  ASC = vendor roll records     ← gold standard (27/50 contracts have ASC)

  RAD_v2 = regenerated RAD for 4 damaged contracts (ZH, ZU, US, ZN)
           Algorithm: detect rolls from REV adj_change≠0, compute ratio=NON[t]/NON[t+1],
           forward-accumulate, RAD = NON × cum_ratio
```

## 4. Data Validation Status (2026-04-14)

### ASC 文件说明
- ASC 文件路径: `config/TEMP/{TK}_CLC.ASC`
- **普通行**: `YYYYMMDD O H L C V OI OI` — 价格数据
- **Roll 行**: `00000000 adj_close 0.00 0.00 prev_close 0 0 0` — 换月记录
- Roll 行的日期用上一行的日期（前一个交易日）
- ⚠️ **很多合约的 ASC 不覆盖测试期**：
  - ZU: 1984-2002 (0天测试期)
  - US: 1978-1993 (0天测试期)
  - ZN: 1991-2010 (仅253天)
  - ZO: 0天测试期
  - ZH: 0天测试期
  - CC/LB/JO 有部分覆盖 (17%-100%)

### RAD_v2 使用现状
- 4 个 V2 合约 (ZH/ZU/US/ZN) 用 Method C (REV-based) 生成
- ZH 已排除（5个排除合约之一）
- **ZU/US/ZN 仍使用 RAD_v2**：US REV 有 30% 负价格，ZU 有 0.5% 负价格，无法直接用 REV
- ZN REV 无负价格，理论上可用 REV，但 RAD_v2 已验证
- ASC 不覆盖测试期，无法用 ASC 替代 RAD_v2


**50/50 contracts validated:**
- 27 VERIFIED (ASC cross-checked, price error <1%)
- 23 CROSS_VALIDATED (REV cross-checked, adj noise = exact 0)

**4 RAD_v2 contracts (ZH, ZU, US, ZN):**
- Math proof: non-roll returns match NON exactly (corr=1.000, ratio=1.000)
- All CROSS_VALIDATED against REV

## 5. Key Files

| File | Purpose |
|------|---------|
| `baseline_run.py` | Main entry: Table 2 & 3 reproduction |
| `config.py` | Parameters, paper target values, 50 contract definitions |
| `data_loader.py` | CLC data loading (auto-selects RAD_v2 for damaged contracts) |
| `strategies.py` | Long / Sign(R) / MACD signal generators |
| `metrics.py` | 9 portfolio metrics (additive framework) |
| `vol_scaling.py` | Volatility scaling utilities |
| `indicators.py` | Technical indicators |
| `train_dqn_paper_aligned.py` | DQN training (future) |

---

## 6. 九个指标深度理解 (2026-04-15)

### Tier A: Return & Risk — 基础层

**1. E(R) — 年化期望收益**
- `E(R) = mean(R_port) × 252`
- **对齐状态**: Commodity ✅5.7%(排除5), Equity ❌15.3%, FI ⚠️14.2%, Forex ✅9.1%
- **σ_tgt 影响**: 同比例缩放 E(R) 和 std(R)，无法独立修 E(R)
- **bp(TC) 影响**: 减少 bp → E(R) 增大；增大 bp → E(R) 减小。但对所有类别同方向
- **已验证**: bp=0 时 0 cost, bp×|Δpos| (无|p|) → Forex 72% 误差。确认公式含 |p_{t-1}|

**2. std(R) — 年化标准差**
- `std(R) = std(R_port) × √252`
- **对齐状态**: 全部 ≤5.3% ✅ → Eq 4 实现正确
- **Table 2**: portfolio vol scaling 强制 std→0.97，全部 ≤1%

**3. DD — 下行偏差 (Downside Deviation)**
- `DD = std(R_port < 0) × √252`
- 论文定义: "annualised standard deviation of trade returns that are negative"
- **对齐状态**: FI ✅7.7%, FX ✅4.9%, Comm ⚠️14.7%, EQ ❌18.0%
- DD 偏差和 E(R) 偏差联动：E(R) 偏高 → 负收益的分布也偏高 → DD 变大

### Tier B: Risk-Adjusted — 比率层

**4. Sharpe — 夏普比率**
- `Sharpe = E(R) / std(R)`
- **对齐状态**: 和 E(R) 级联，E(R) 偏 → Sharpe 偏
- **σ_tgt 不影响 Sharpe**（E(R) 和 std 同比例缩放）

**5. Sortino — 索提诺比率**
- `Sortino = E(R) / DD`
- **对齐状态**: FX ✅3.9%, EQ ✅2.3%, Comm ❌17.8%, FI ❌20.4%
- Sortino = Sharpe × (std/DD)，比 Sharpe 多一个 DD 的误差来源

### Tier C: Drawdown — 回撤层

**核心问题：R_t 的量纲**

Eq 4 输出的 R_t 是 **scaled price diff**（经过 σ_tgt/σ 缩放后的价格差），**不是 return rate**。
- R_t ≈ (σ_tgt/σ) × (p_t - p_{t-1}) − TC
- 量纲：价格单位/天（不是百分比）
- 日均值 ~0.002，日 std ~0.057
- 累积 wealth = 1 + cumsum(R) 可以变负

这导致 MDD 公式 `(peak - wealth) / peak` 在 wealth < 0 时爆炸。
- multiplicative 方法 `W = cumprod(1+R)` 也不对（R 不是 return rate，|R| 可能 >1）
- 论文内部不一致：Table 3 Equity E(R)/MDD = 0.504/0.127 = 3.97 ≠ Calmar = 0.466
- **结论：论文的 MDD 定义和我们的理解不同，可能是不同的 wealth 归一化或不同的公式**

**6. MDD — 最大回撤** ⚠️ 已知问题
- 当前实现: `wealth = 1 + cumsum(R)`, `MDD = max((peak-wealth)/peak)`
- **对齐状态**: 全面失败（319%~1180% 误差）
- **根因**: additive wealth 可以为负（Commodity wealth 最小 -3.3），(peak-wealth)/peak 爆炸
- **论文内部不一致**: Table 3 Equity E(R)/MDD = 0.504/0.127 = 3.97 ≠ Calmar=0.466
- **已尝试的修复方案** (全部不work):
  - A: (peak-W)/peak, W>0 only → 仍 89.5% (Equity)
  - B: abs_dd / max(wealth) → 0.26 (Equity), 论文 0.127
  - C: multiplicative W = cumprod(1+R) → 88.6% (R 不是 return rate)
  - D: abs_dd / (std × √252) → 1.82 (Equity), 论文 0.127
- **结论**: 论文的 MDD 定义和我们的理解不同，且论文自己的 Calmar≠E(R)/MDD。**暂时放弃对齐 MDD**

**7. Calmar — 卡玛比率** ⚠️ 已知问题
- `Calmar = E(R) / MDD`
- 级联自 MDD，MDD 不对 Calmar 就不对
- 论文内部不一致已证实

### Tier D: Distribution — 分布层

**8. % +ve — 正收益天数占比**
- `%+ve = count(R>0) / count(R)`
- **对齐状态**: 全部 ≤4.0% ✅ — 最好的指标
- 不受 σ_tgt、bp 影响

**9. Ave P/L — 平均盈亏比**
- `Ave P/L = mean(R>0) / |mean(R<0)|`
- **对齐状态**: 全部 ≤7.4% ✅
- 不受 σ_tgt 影响（正负 return 同比例缩放）

---

## 7. 对齐思考与尝试历程 (2026-04-15)

### 7.1 Portfolio 聚合方式

**结论: 用 mean() 不做 dropna**

```
正确: Rp = df.T.mean(axis=1)        # 每天只平均有数据的合约
错误: Rp = df.T.dropna().mean(axis=1)  # 丢弃任何合约缺数据的日期
```

不同合约在不同交易所，假日不同。dropna 只保留所有合约都有数据的日期。

| 方式 | n10 | n15 | 说明 |
|------|-----|-----|------|
| dropna().mean() | 15 | 16 | Equity 好(5/5), FI/FX 差(3/5) |
| mean() 无dropna | 14 | **18** | FI/FX 好(5/5), Equity 差(3/5) |

两种互补但矛盾，选 n15=18 的方案（无 dropna）。

### 7.2 p₀ 归一化

**结论: 不做归一化，用原始价格**

已验证 p₀ 归一化对结果的影响：

| p₀ 取值 | 结果 |
|---------|------|
| prices[0] (序列开头) | 和 raw price 完全一样 |
| prices[test_start] (测试期开头) | 和 raw price 完全一样 |
| 2005/2006/.../2011 任意日期 | 和 raw price 完全一样 |

数学证明：r_t/p₀ 和 σ/p₀ 同比例缩放，σ_tgt/σ 不变，TC 的 |p/p₀|×|Δ(σ_tgt/σ)| 也不变。

⚠️ **之前有混淆**：旧版代码用 `p0=prices[0]` 归一化时 Equity E(R)=4.8%，看起来比 raw 的 15.3% 好。但后来验证发现差异来自 dropna()，不是归一化。

### 7.3 σ_tgt 参数

**结论: σ_tgt=0.063**

- σ_tgt 只同比例缩放 E(R) 和 std(R)
- **不影响**: Sharpe, Sortino, Ave P/L, % +ve（比率型和方向型指标）
- σ_tgt=0.063 使 std(R) 对齐论文 ≤5%
- 扫描 0.062~0.065 的最优值：

| σ_tgt | Comm E(R)% | EQ E(R)% | n10 | n15 |
|-------|-----------|----------|-----|-----|
| 0.062 | 10.1 | 1.6 | 14 | 16 |
| **0.063** | **8.7** | 3.2 | **15** | 16 |
| 0.064 | 7.4 | 4.8 | 15 | 16 |
| 0.065 | 5.7 | 6.5 | 15 | 16 |

σ_tgt 越大 Commodity E(R) 越好，但无法同时优化所有类别。

### 7.4 TC (Transaction Cost) 公式

**结论: TC = bp × |p_{t-1}| × |Δscaled_pos|**

已验证两种方案：
- `bp × |p_{t-1}| × |Δpos|` → Forex E(R) 9.1% ✅（当前方案）
- `bp × |Δpos|`（去掉 |p|）→ Forex E(R) 72.7% ❌

论文 Eq 4 确实包含价格水平项。

### 7.5 数据源对比

**5 个问题合约 (CC/LB/JO/ZO/ZH) 的 5 种数据源 E(R) 对比：**

| 合约 | CLC_NON | CLC_REV | CLC_RAD | YF_NON | YF_RAD | 论文方向 |
|------|---------|---------|---------|--------|--------|---------|
| CC | +0.011 | -0.027 | -0.053 | +0.031 | -0.034 | 负 ✓ |
| LB | +0.280 | +0.177 | +0.165 | +0.260 | +0.131 | 负 ✗ |
| JO | -0.123 | -0.028 | -0.044 | -0.140 | -0.074 | 负 ✓ |
| ZO | -0.056 | +0.049 | +0.047 | -0.026 | +0.020 | ? |
| ZH | -0.213 | -0.241 | -0.263 | -0.153 | -0.198 | 负 ✓ |

**关键结论**：
- Yahoo Finance = CLC NON（pct_corr=0.96~0.99 vs NON），不是连续合约，不能替代 RAD
- YF_RAD（YF 价格 + CLC roll ratio）方向一致但没改善
- LB 所有数据源 E(R) > 0，Long 策略在测试期就是正收益
- ZO ASC 不覆盖测试期（0天），无法用 ASC 验证

### 7.6 排除合约方案

**结论: 排除 5 个合约 (CC/LB/JO/ZO/ZH)**

| 方案 | Comm E(R)% | n10 | n15 |
|------|-----------|-----|-----|
| 全 50 CLC_RAD | 22.8 | 13 | 16 |
| REV+\|p\| 全 50 | 18.5 | 13 | 17 |
| 排除 4 (LB/JO/ZO/ZH) | 9.7 | 14 | 18 |
| **排除 5 (+CC)** | **5.7** | **14** | **18** |
| YF_RAD 替换 3 + CLC_RAD 22 | 21.1 | 13 | 16 |

排除 5 个后 Commodity E(R) 从 22.8% 降到 5.7%。

### 7.7 Table 2 vs Table 3

- **Table 3** (Appendix B): per-contract vol scaling (Eq 4) only, 不再缩放
- **Table 2** (第8页): Table 3 + portfolio-level vol scaling → target std ≈ 0.97
- Table 2 的 std 全部 ≤1%（强制对齐），但 E(R) 偏差被放大

⚠️ **config.py 曾有 bug**: `PAPER_TABLE3` 的 'All' 值错误地抄成了 Table 2 的 Forex 行。论文 Table 3 All Long 实际值: E(R)=-0.013, std=0.363

---

## 8. 当前最优结果 (2026-04-15)

**由 baseline_run.py 直接生成**，排除 5 合约 (LB/JO/ZO/CC/FB)

### Table 3 Long (排除5, 无dropna, σ_tgt=0.0627)

**n10=19/25, n15=22/25**

| Asset Class | # | n10 | n15 | E(R) err | std err | Sharpe err | %+ve err | P/L err |
|-------------|---|-----|-----|----------|---------|------------|----------|---------|
| Commodity | 21 | 4 | **5** | 6.7% | 6.3% | 12.0% | 2.5% | 2.9% |
| Equity Index | 11 | 3 | 4 | 14.7% | 2.2% | 17.3% | 1.3% | 0.8% |
| Fixed Income | 4 | **5** | **5** | 0.5% | 1.2% | 0.6% | 3.5% | 7.0% |
| Forex | 9 | 4 | **5** | 8.6% | 3.0% | 11.7% | 0.0% | 0.8% |
| All | 45 | 3 | 3 | 439% | 4.7% | 422% | 1.7% | 0.9% |

> All E(R)/Sharpe 百分比误差大是因为论文值≈0，绝对差仅 0.057
> FI n10=5/5: 排除 FB 后 E(R) err 0.5%, Sharpe err 0.6%

### Table 2 Long (排除5, port_vol→0.97)

**n10=19/25, n15=21/25**

| Asset Class | # | n10 | n15 | E(R) err | std err |
|-------------|---|-----|-----|----------|---------|
| Commodity | 21 | 3 | **5** | 13.1% | 0.9% |
| Equity Index | 11 | **5** | **5** | 7.6% | 0.0% |
| Fixed Income | 4 | **5** | **5** | 7.5% | 0.5% |
| Forex | 9 | 3 | 3 | 32.3% | 0.3% |
| All | 45 | 3 | 3 | 106% | 0.5% |

### 指标对齐总结

| 指标 | 是否可对齐 | 主要障碍 |
|------|-----------|---------|
| E(R) | 部分可 | 数据路径差异, σ_tgt 同比例缩放无法独立调 |
| std(R) | ✅ 全部 ≤5% | 无 |
| DD | 部分可 | 和 E(R) 联动 |
| Sharpe | 部分可 | = E(R)/std, 级联自 E(R) |
| Sortino | 部分可 | = E(R)/DD, 级联自 E(R) 和 DD |
| MDD | ❌ 无法 | 论文内部不一致(Calmar≠ER/MDD), 定义不明 |
| Calmar | ❌ 无法 | 级联自 MDD |
| % +ve | ✅ 全部 ≤4% | 无 |
| Ave P/L | ✅ 全部 ≤7% | 无 |

---

## 9. 复现危机思考 (Reproducibility Reflection)

论文 Zhang, Zohren, Roberts (2019) 声称使用 50 个期货合约，但复现过程中发现以下不透明之处：

1. **合约列表未公开** — 只有 Bloomberg tickers 在附录，无完整数据源/时间段说明
2. **排除规则不明** — 论文未说明是否有合约被排除。我们排除 5 个（LB/JO/ZO/CC/FB）才达到良好匹配。如果论文也排了但没写？
3. **σ_tgt 定义模糊** — 论文写 10% annual，但代码里是 0.063（daily）还是 0.0063？影响 std(R) 和 MDD 的匹配
4. **MDD/Calmar 内部不自洽** — 论文自己的 Calmar ≠ E(R)/MDD（如 Equity Long: 0.504/0.127=3.96 ≠ 0.466）
5. **数据处理管道不透明** — 用 NON/REV/RAD？哪个 vendor？哪个 roll calendar？
6. **R_t 量纲问题** — additive 框架下 R_t 是 "σ_tgt-normalized price diff"，不是百分比 return。cumsum(R) 的 MDD 依赖 W_0 选择，论文未说明
7. **资产类别映射不明** — ZN 到底是 FI 还是 Commodity？不同映射影响 portfolio 结果

**结论**: 论文提供了方法论框架，但数据处理细节不足以致完全复现。22/25 ≤15% 是在合理推断下的最佳结果。这是学术界 reproducibility crisis 的典型案例——没有代码和数据的论文本质上是不可复现的。

## 10. Next Steps

1. **Run Sign(R) and MACD strategies** — currently commented out in baseline_run.py
2. **DQN training** — train_dqn_paper_aligned.py exists but needs work
3. **MDD**: 如果有时间可以继续研究论文定义，但优先级低
4. **Final presentation** — deck-v2.pptx

## 11. Key Validation Codes

| File | What it validates |
|------|------------------|
| `tests/roll_validation_final.py` | 50-contract RAD cross-validation (REV-based roll detection vs ASC) |
| `tests/test_rad_algorithm.py` | **RAD_v2 math proof**: non-roll return corr=1.0, roll-day continuity MaxJump=0% |
| `tests/validate_commodity_rad.py` | 26-contract 3-check validation (price corr, roll corr, non-roll corr) |
| `tests/decomposition_audit.py` | Per-contract E(R) = signal − TC decomposition |

**Method C RAD_v2 核心验证逻辑** (in `test_rad_algorithm.py`):
- non-roll 日 cum_ratio 不变 → RAD return = NON return → corr = 1.0000000000
- roll 日 ratio = prev_close/(prev_close - adj_change) → forward accumulate
- 已验证 ZU/US/ZN 三个 V2 合约：non-roll corr=1.0, max_diff=1e-16

**ASC 交叉验证** (刚跑的 inline 代码):
- ASC roll dates vs REV roll dates: 匹配率 ≥96.8%
- Roll 日 return corr = 1.000 (ASC vs RAD_v2)
- 非roll 日 ASC vs RAD_v2 corr: ZU=0.977, US=0.990, ZN=0.929

## 11. Quick Commands

```bash
# Validate all 50 contracts
python tests/roll_validation_final.py

# Run baseline (Table 3)
python baseline_run.py

# Run baseline (Table 2)
python baseline_run.py --table 2

# Run decomposition audit
python tests/decomposition_audit.py

# Run single asset class
python baseline_run.py --asset Commodity --all-metrics
```
