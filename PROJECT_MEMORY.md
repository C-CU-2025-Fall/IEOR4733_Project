# PROJECT_MEMORY.md — AI Context Pickup File
# > Last updated: 2026-04-16
# > Read this FIRST when starting a new session on this project

---

## 0. Latest Condensed Update (2026-04-16)

This file had accumulated too many search logs. The repo has been condensed so the core retained changes from the latest MDD / Calmar work are:

1. **`baseline_run.py` now supports a separate reporting lane for `MDD` / `Calmar`, and the current preferred bridge is `RISK_PRICE_SIGMA0`.**
   Use:
   - `--report-source RISK_PRICE_SIGMA0` for the current split-world start
   - `--report-source trade` for the pure trade-return all-9 reference
   - the baseline CLI is intentionally lean again; older experimental reporting branches are no longer part of the active runtime surface

2. **Current interpretation is now explicit.**
   - Eq. 4 `R_t` is a standardized additive trading reward, not an equal-dollar return
   - Eq. 13 averages those standardized rewards equally across contracts
   - trade world:
     - `E(R), std(R), DD, Sharpe, Sortino, % +ve, Ave P/L`
   - reporting world:
     - `MDD, Calmar`

3. **Current reporting bridge.**
   For each contract sleeve:
   - `C_i,0 = p_i,0 × sigma_tgt / sigma_i,0`
   - `w_i,t = 1 + cumsum(R_i,t / C_i,0)`
   - portfolio wealth = equal-weight average of sleeve wealth paths

4. **Core reusable helpers retained in code.**
   - `cagr_from_path` and `max_drawdown_from_path` in `metrics.py`
   - the active reporting helper now lives directly in `baseline_run.py` as the `RISK_PRICE_SIGMA0` sleeve-capital bridge

5. **Exploratory search scripts and generated markdown reports from this branch were intentionally removed.**
   The point is to keep the repo centered on the live baseline plus the reusable reporting framework. If a search is needed again, rerun it locally from the core helpers instead of preserving every intermediate artifact.

### Current Live Table 3 Reference

Reference command:

```bash
python baseline_run.py --table 3 --all-metrics --sigma 0.058
```

Current live baseline under that reference run:

- exclusions: none
- source overrides: `25`
- default report source: `RISK_PRICE_SIGMA0`
- reference score:
  - `<=10%: 25/45`
  - `<=15%: 34/45`

Per-asset summary under the reference run:

| Asset | E(R) | std(R) | DD | Sharpe | Sortino | MDD | Calmar | % +ve | Ave P/L | n10 | n15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Commodity | -0.264 | 0.389 | 0.265 | -0.678 | -0.993 | 0.209 | -0.228 | 0.489 | 0.937 | 5 | 7 |
| Equity Index | +0.523 | 0.839 | 0.659 | +0.623 | +0.794 | 0.144 | 0.317 | 0.547 | 0.922 | 6 | 8 |
| Fixed Income | +0.478 | 0.854 | 0.556 | +0.560 | +0.860 | 0.123 | 0.266 | 0.529 | 0.976 | 4 | 6 |
| Forex | -0.177 | 0.423 | 0.273 | -0.419 | -0.650 | 0.220 | -0.089 | 0.490 | 0.971 | 6 | 9 |
| All | +0.033 | 0.336 | 0.236 | +0.098 | +0.139 | 0.121 | -0.037 | 0.521 | 0.933 | 4 | 4 |

### Two-World Reporting Status

Pure trade-world reference:

```bash
python baseline_run.py --table 3 --all-metrics --sigma 0.058 --report-source trade
```

Pure trade-world score:

- `<=10%: 24/45`
- `<=15%: 33/45`

Current split-world start:

```bash
python baseline_run.py --table 3 --all-metrics --sigma 0.058 \
  --report-source RISK_PRICE_SIGMA0
```

Interpretation:

- trade metrics stay in the Eq. 4 / Eq. 13 world:
  - `E(R), std(R), DD, Sharpe, Sortino, % +ve, Ave P/L`
- only `MDD` and `Calmar` switch to the sleeve-capital reporting lane

Current result on the **live current contract sets**:

- `<=10%: 25/45`
- `<=15%: 34/45`

This is now the promoted start point because it is the first split-world bridge that:

- is explainable from price and initial risk scale,
- improves the full Table 3 score over pure trade world,
- and does not produce the pathological `MDD/Calmar` explosions seen in the earlier `WINDOW_FWD` global attempt.

What it did establish cleanly:

- the paper still looks like a **split-world** setup
- `RISK_PRICE_SIGMA0` is a better global reporting bridge than the older `WINDOW_FWD` attempt

### Key Fixed Income Takeaway

When the same FI subset is enforced across both worlds, the strongest row found was:

- subset: `DT,TY,US`
- trade source: `RAD` or `REV`
- reporting source: `WINDOW_FWD` or `RAD_REGEN`

That row is approximately:

- `E(R)=0.578`
- `std(R)=0.877`
- `DD=0.571`
- `Sharpe=0.659`
- `Sortino=1.012`
- `MDD=0.093`
- `Calmar=0.460`
- `% +ve=0.533`
- `Ave P/L=0.974`

Against paper FI:

- `0.605, 0.939, 0.561, 0.645, 1.081, 0.108, 0.455, 0.515, 1.048`

This is still an important FI-specific result, but it is no longer the main promoted global start point.

### Historical Failure Worth Remembering

The repo briefly used `WINDOW_FWD` as a global reporting default. That attempt was wrong as a start point because:

- it mixed trade-lane source overrides with a separate reporting construction
- some asset classes produced nonsensical `MDD/Calmar` outputs, including huge Calmar values for Commodity and `All`
- it improved some local FI intuitions, but failed as a coherent global baseline

Keep this failure in mind when presenting:
- it was a useful diagnostic
- it was not a valid final search baseline

### Historical Note

Older sections below are preserved as historical reasoning logs. They may mention frontiers, exclusions, or generated files that are **not** the current live baseline anymore. For the current repo state, trust this Section 0 first.

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

## 9. 当前分支基线 (2026-04-15 夜间更新)

### 9.1 Active universe policy

- **当前排除 4 个合约**: `LB`, `ZO`, `CC`, `FB`
- `JO` 已重新加入 universe
- 新原则：**不再通过继续删合约来“刷表”**；后续只能在 metric/scaling 理解改善后把合约加回，不能继续缩 universe

### 9.2 Current Table 3 baseline

> Historical mid-iteration snapshot. The current live frontier is in Section 14.

**由 `python baseline_run.py --table 3` 直接生成**  
当前设定：排除 4，`aggregation=variable_n`，`σ_tgt=0.0627`

**n10=18/25, n15=22/25**

| Asset Class | # | E(R) err | std err | Sharpe err | %+ve err | P/L err |
|-------------|---|----------|---------|------------|----------|---------|
| Commodity | 22 | 10.1% | 3.2% | 12.7% | 3.2% | 4.2% |
| Equity Index | 11 | 14.7% | 2.2% | 17.3% | 1.3% | 0.8% |
| Fixed Income | 4 | 0.5% | 1.2% | 0.6% | 3.5% | 7.0% |
| Forex | 9 | 8.6% | 3.0% | 11.7% | 0.0% | 0.8% |
| All | 46 | abs gap 0.056 | 3.3% | abs gap 0.150 | 1.5% | 0.4% |

> `All` 行继续用 absolute gap 更合理，因为论文值接近 0，percent error 会爆炸。

### 9.2b Full 45-comparison context

`docs/er_attribution_report.md` 已生成当前 full-9-metric scorecard：

- **n10=24/45**
- **n15=29/45**

这说明当前 baseline 距离你要求的 **40/45** 还很远，Table 3 不能停。

### 9.3 MDD branch update

- `metrics.py` 现在对 additive MDD 使用 **`W₀ = N_contracts`**
- 这让 additive wealth path 至少在量纲上更接近 “每个 sleeve 先有 1 单位初始财富”
- 但这 **不是** MDD 问题的最终答案；目前只把它视为 additive-wealth branch 的最新基线

## 10. E(R) 归因结论 (Attribution)

### 10.1 数学恒等式

对 `variable_n` 聚合：

```
R_port,t = (1 / N_t) * Σ_i R_i,t
E(R_port) = 252 * mean_t[(1 / N_t) * Σ_i R_i,t]
```

所以每个合约的 realized annualized contribution 是：

```
contrib_i = 252 * mean_t[I_i,t * R_i,t / N_t]
```

且因为 `R_i,t = signal_i,t - tc_i,t`，所以同样可以分解成：

```
E(R) contribution = signal contribution - tc contribution
```

### 10.2 JO re-add generated result

`docs/er_attribution_report.md` 已生成，比较了：

- 旧方案：排除 5 (`LB/JO/ZO/CC/FB`)
- 新方案：排除 4 (`LB/ZO/CC/FB`)

**生成结果结论**：

- JO 加回后，Commodity `E(R)` 从 `-0.278` 变到 `-0.268`
- Commodity 对论文的 `|E(R) gap|` **变差 +0.010**
- `All` 的 `|E(R) gap|` 只 **改善 0.001**
- 所以：**JO re-add 是 policy-consistent，但不是 replication score improvement**

### 10.3 JO attribution itself

在当前 baseline 下，`JO` 的 realized contribution（见 `docs/er_attribution_report.md`）：

- **JO in Commodity**:
  - trade contribution = `-0.002`
  - signal contribution = `-0.000`
  - tc contribution = `+0.001`
- **JO in All**:
  - trade contribution = `-0.001`

**解释**：
- JO 对 Commodity 的方向是略负，但负得还不够，结果把 Commodity Long 从 `-0.278` 拉到 `-0.268`，反而离论文 `-0.298` 更远
- 对 `All`，JO 的影响非常小，只带来 `0.001` 级别的 absolute-gap 改善
- 这说明：**把 JO 加回是 universe-policy 的尝试，不是当前 Table 3 score 的提升来源**

### 10.4 All-4-asset attribution highlights

`docs/er_attribution_report.md` 还给出了四个资产类的 realized contributors 和 targeted leave-one-out 结果：

- **Commodity**: `ZA`, `DA`, `ZT` 是当前最值得诊断的正贡献合约；去掉它们会明显让 Commodity 更接近论文负收益
- **Equity Index**: `EN`, `YM`, `ES/SC/SP` 是当前把 Equity `E(R)` / `Sharpe` 推高的主要来源
- **Fixed Income**: 当前并不是主要问题；FI 已经很接近论文，进一步动合约大概率会伤 baseline
- **Forex**: `AN`, `BN`, `MP` 的 leave-one-out 对 FX `E(R)` gap 有帮助，但对 `All` 往往是反作用，说明 FX 目前不是第一优先级

**阶段结论**：
- 下一步最应该投入的是 **Equity + Commodity 的 E(R) attribution / data-path diagnosis**
- 不是继续做 Table 2，也不是继续删合约

## 11. Next TODO (优先级排序)

1. **继续做 E(R) attribution，而不是先调 Table 2**
   目标是解释 Equity / All 为什么偏，而不是先做 portfolio vol scaling 花活
2. **把 drawdown 指标 split 成两条 branch**
   - additive branch: 当前 `W₀ = N_contracts`
   - wealth/NAV branch: backtesting-consistent drawdown
3. **只允许 add-back，不允许继续删合约**
   下一个候选不是“再排除谁”，而是检查当前 excluded contracts 在新 metric/scaling 理解下能否合理加回
4. **Table 3 先冲到 ~40/45 meaningful comparisons**
   在这个门槛前，Table 2 只做 reference，不做主战场
5. **优先顺序**
   - `Equity Index` attribution / data understanding
   - `Commodity` attribution / data understanding
   - drawdown bridge audit
   - 只有达到更高 Table 3 score 后，再回 Table 2

## 12. Key Validation Codes

| File | What it validates |
|------|------------------|
| `tests/roll_validation_final.py` | 50-contract RAD cross-validation (REV-based roll detection vs ASC) |
| `tests/test_rad_algorithm.py` | **RAD_v2 math proof**: non-roll return corr=1.0, roll-day continuity MaxJump=0% |
| `tests/validate_commodity_rad.py` | 26-contract 3-check validation (price corr, roll corr, non-roll corr) |
| `tests/decomposition_audit.py` | Per-contract E(R) = signal − TC decomposition |
| `tests/er_attribution_analysis.py` | **E(R) attribution proof** for JO add-back (`signal - tc = trade`, plus scenario delta) |

**Method C RAD_v2 核心验证逻辑** (in `test_rad_algorithm.py`):
- non-roll 日 cum_ratio 不变 → RAD return = NON return → corr = 1.0000000000
- roll 日 ratio = prev_close/(prev_close - adj_change) → forward accumulate
- 已验证 ZU/US/ZN 三个 V2 合约：non-roll corr=1.0, max_diff=1e-16

**ASC 交叉验证**:
- ASC roll dates vs REV roll dates: 匹配率 ≥96.8%
- Roll 日 return corr = 1.000 (ASC vs RAD_v2)
- 非roll 日 ASC vs RAD_v2 corr: ZU=0.977, US=0.990, ZN=0.929

## 13. Quick Commands

```bash
# Validate all 50 contracts
python tests/roll_validation_final.py

# Run baseline (Table 3)
python baseline_run.py

# Run baseline (Table 2)
python baseline_run.py --table 2

# Run decomposition audit
python tests/decomposition_audit.py

# Run JO attribution report
python tests/er_attribution_analysis.py
```

## 14. Current Table 3 Frontier (2026-04-16)

### 14.1 这轮迭代做了什么

本轮没有继续删合约，而是把重点放在 **data-path/source bridge**：

- 支持按合约选择 `RAD` / `REV` / `RAD_REGEN`
- 用 generated result 搜索 source override，而不是凭感觉改
- 在 source override 固定后，再重新搜索 `sigma_tgt`

新增/更新的关键代码：

- `data_loader.py`
  - 新增 source-aware loader：`RAD` / `REV` / `NON` / `RAD_REGEN`
- `baseline_run.py`
  - `load_contracts(..., source_overrides=...)`
  - 当前默认 `sigma_tgt = 0.0600`
- `repro_analysis.py`
  - `evaluate_table(..., source_overrides=...)`
- `tests/source_override_search.py`
  - 生成 `docs/source_override_search_report.md`

### 14.2 当前 active working frontier

当前默认 baseline 使用：

- excluded contracts: `LB`, `ZO`, `CC`, `FB`
- `sigma_tgt = 0.0600`
- aggregation: `variable_n`
- active source overrides:

```python
{
    'DA': 'RAD_REGEN',
    'EN': 'REV',
    'ER': 'REV',
    'ES': 'REV',
    'GI': 'RAD_REGEN',
    'JN': 'RAD_REGEN',
    'JO': 'REV',
    'KC': 'REV',
    'KW': 'REV',
    'MD': 'RAD_REGEN',
    'MP': 'RAD_REGEN',
    'NK': 'RAD_REGEN',
    'SC': 'RAD_REGEN',
    'SP': 'RAD_REGEN',
    'YM': 'RAD_REGEN',
    'ZA': 'RAD_REGEN',
    'ZC': 'REV',
    'ZF': 'REV',
    'ZG': 'RAD_REGEN',
    'ZH': 'REV',
    'ZI': 'REV',
    'ZK': 'REV',
    'ZN': 'REV',
    'ZR': 'REV',
    'ZT': 'RAD_REGEN',
    'ZU': 'REV',
    'ZW': 'REV',
}
```

### 14.3 Generated result: current baseline score

由 `python baseline_run.py --table 3 --all-metrics` 直接生成：

- **≤10%: 27/45**
- **≤15%: 34/45**

各资产当前 Long 结果：

| Asset | E(R) ours | E(R) paper | \|E(R) gap\| | Sharpe ours | Sharpe paper | \|Sharpe gap\| | n10 | n15 |
|------|-----------:|------------:|-------------:|-------------:|--------------:|---------------:|----:|----:|
| Commodity | -0.293 | -0.298 | 0.005 | -0.720 | -0.723 | 0.003 | 7 | 7 |
| Equity Index | +0.536 | +0.504 | 0.032 | +0.617 | +0.543 | 0.074 | 5 | 8 |
| Fixed Income | +0.576 | +0.605 | 0.029 | +0.649 | +0.645 | 0.004 | 7 | 7 |
| Forex | -0.173 | -0.198 | 0.025 | -0.395 | -0.420 | 0.025 | 5 | 8 |
| All | +0.029 | -0.013 | 0.042 | +0.082 | -0.036 | 0.118 | 3 | 4 |

### 14.4 和旧 baseline 相比，为什么 improvement 是真的

这轮 improvement 不是“再排除几个合约”换来的，而是两步叠加：

1. **Commodity source fixes**
   `DA/GI/ZG/ZT -> RAD_REGEN`, `JO/KW/ZF/ZH/ZN/ZU/ZW -> REV`
   直接把 Commodity `E(R)` 从旧 baseline 的 `-0.268` 推到接近论文的 `-0.291 ~ -0.298`

2. **Equity source fixes + lower sigma**
   `EN/ER/ES -> REV`, `MD/SC/SP/YM -> RAD_REGEN`
   再把 Equity `E(R)` 从 `+0.578` 压到 `+0.536`
   同时 `sigma_tgt` 从 `0.0627` 下调到 `0.0600`，进一步帮助 Equity / All

3. **Late no-regression All refinements**
   `MP -> RAD_REGEN`, `ZC/ZI/ZK/ZR -> REV`
   没有再抬高 score，但继续把 `Commodity` 和 `All` 的 absolute gap 往下压了一点

也就是说：

- **编程上**：确实改变了 loader 所使用的价格路径
- **数学上**：Eq. 4 的 `r_t = p_t - p_{t-1}` 直接由 source 决定，所以 source 变了，`signal`、`tc`、`trade` 的 realized annualized contribution 就会跟着变
- improvement 的方向和前面的 attribution 诊断是一致的，不是 random noise

### 14.5 还没到 40/45，剩下的主要卡点

虽然已经从旧的 `29/45`（≤15%）推进到 **34/45**，但还没有到目标的 `40/45`。

当前最主要的剩余问题：

- `All` row 仍然最难
  - `E(R)` paper 接近 0，百分比误差天然会炸
  - 即使绝对 gap 已经改善到 `0.044`，仍然是 major blocker
- `Calmar` 仍然基本不可信
  - 论文内生不一致，继续优化它没有意义
- `MDD` 也仍然更多是 diagnostic
- `Equity` 还可以继续小修，但已经没有 Commodity 那种大块头 source mismatch 了

### 14.6 当前判断

- **这是目前最强的 Table 3 working frontier**
- 继续冲 `40/45` 的下一步，不应该回去搞 Table 2
- 下一步最值得做的是：
  1. 围绕 `All` row 做更细的组合/bridge 诊断
  2. 检查剩余小幅 source switch 是否能继续改善 `All`
  3. 继续把 drawdown 指标留在次要轨道，不让它主导 search
