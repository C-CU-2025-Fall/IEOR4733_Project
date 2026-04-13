# 🔧 MDD/DD 修复长期计划

**创建日期**: 2026-04-12  
**目标**: 解决 MDD (122.7% 平均误差) 和 DD (15.6% 平均误差) 的系统性偏差  
**优先级**: 高 — 影响论文核心指标复现

---

## 📋 问题概述

### 当前状态

| 指标 | ≤15% | 平均误差 | 最大误差 | 状态 |
|------|------|---------|---------|------|
| **DD** | 6/12 | 15.6% | 28.7% | ⚠️ 部分偏差 |
| **MDD** | **1/12** | **122.7%** | **553.2%** | ❌ 严重偏差 |

### 异常最严重的案例

| Asset | Strategy | Our MDD | Paper MDD | Error |
|-------|----------|---------|-----------|-------|
| Fixed Income | MACD | 0.810 | 0.124 | **553.2%** ❌ |
| Fixed Income | Sign(R) | 0.635 | 0.165 | **284.8%** ❌ |
| Forex | Sign(R) | 0.406 | 0.170 | **138.8%** ❌ |
| Forex | MACD | 0.382 | 0.156 | **144.9%** ❌ |

---

## 🎯 阶段一：Paper 公式深度梳理 (Week 1)

### Task 1.1: 重读 Paper Section 4.4
**目标**: 逐字分析 MDD/DD 定义

**检查清单**:
- [ ] MDD 是否使用 additive wealth 还是 multiplicative wealth?
- [ ] DD 的 MAR (Minimum Acceptable Return) 是什么？(0? risk-free rate?)
- [ ] 是否有补充材料 (Supplementary Material) 提供计算细节？
- [ ] 引用文献 [27] Lim et al. 如何定义这些指标？

**输出**: `docs/paper_metrics_analysis.md`

### Task 1.2: 查阅引用文献 [27]
**文献**: Lim, Kim, Kim 等 — "Deep Momentum Networks"

**检查清单**:
- [ ] 获取论文全文
- [ ] 查找 Downside Deviation 定义
- [ ] 查找 Maximum Drawdown 定义
- [ ] 确认是否使用 logarithmic returns 或 simple returns

**输出**: `docs/lim_et_al_metrics.md`

### Task 1.3: 行业标准对比
**目标**: 确认 CFA/学术标准定义

**检查清单**:
- [ ] CFA Institute 对 DD 的定义 (MAR=0 vs MAR=rf)
- [ ] CFA Institute 对 MDD 的定义 (peak-to-trough vs 其他)
- [ ] 期货/量化交易领域的惯例

**输出**: `docs/industry_standard_metrics.md`

---

## 🔬 阶段二：计算方法实验 (Week 2)

### Task 2.1: 实现多种 MDD 计算方法

**方法 A: 当前实现 (additive wealth)**
```python
cumret = np.cumsum(R_eq)
wealth = N * W0 + cumret
peak = np.maximum.accumulate(wealth)
mdd = max((peak - wealth) / peak)
```

**方法 B: Multiplicative wealth**
```python
wealth = np.cumprod(1 + R_eq)
peak = np.maximum.accumulate(wealth)
mdd = max((peak - wealth) / peak)
```

**方法 C: Log returns**
```python
log_r = np.log(1 + R_eq)
cum_log = np.cumsum(log_r)
wealth = np.exp(cum_log)
# 然后计算 MDD
```

**方法 D: 不使用 N×W0 缩放**
```python
cumret = np.cumsum(R_eq)
peak = np.maximum.accumulate(cumret)
trough = cumret - peak
mdd = max(-trough)  # 绝对值 MDD
```

**方法 E: Portfolio-level scaling**
```python
# 先计算 portfolio return，再应用 vol scaling
# 而非 per-contract scaling
```

**输出**: `tests/mdd_methods_comparison.py`

### Task 2.2: 实现多种 DD 计算方法

**方法 A: 当前实现 (zero-target LPM(2))**
```python
shortfall = np.minimum(R_eq, 0.0)
dd = np.sqrt(np.mean(shortfall ** 2)) * np.sqrt(252)
```

**方法 B: 仅负收益的 std**
```python
neg_returns = R_eq[R_eq < 0]
dd = np.std(neg_returns, ddof=0) * np.sqrt(252)
```

**方法 C: MAR = risk-free rate**
```python
MAR = risk_free_rate / 252
shortfall = np.minimum(R_eq - MAR, 0.0)
dd = np.sqrt(np.mean(shortfall ** 2)) * np.sqrt(252)
```

**方法 D: LPM(1) 而非 LPM(2)**
```python
shortfall = np.maximum(-R_eq, 0.0)
dd = np.mean(shortfall) * 252  # 一阶矩
```

**输出**: `tests/dd_methods_comparison.py`

### Task 2.3: 网格测试所有组合

**测试矩阵**:
| MDD 方法 | DD 方法 | 预期结果 |
|---------|--------|---------|
| A (additive) | A (zero-LPM2) | 当前结果 (baseline) |
| B (multiplicative) | A | 测试 |
| A | B (neg-std) | 测试 |
| B | B | 测试 |
| ... | ... | ... |

**输出**: `tests/results/mdd_dd_grid_search.csv`

---

## 📊 阶段三：数据一致性验证 (Week 3)

### Task 3.1: 单合约 vs 组合 MDD

**假设**: Paper 可能先计算单合约 MDD，再平均

**实验**:
```python
# 方法 1: 组合回报 → MDD (当前)
portfolio_return = mean(R_i)
mdd_portfolio = MDD(portfolio_return)

# 方法 2: 单合约 MDD → 平均
mdd_i = [MDD(R_i) for i in contracts]
mdd_avg = mean(mdd_i)
```

**输出**: `tests/mdd_aggregation_test.py`

### Task 3.2: Vol Scaling 前后 MDD 对比

**假设**: Paper 的 MDD 可能是 un-scaled 的

**实验**:
```python
# 方法 1: scaled returns → MDD (当前)
R_scaled = vol_scale(R_raw)
mdd = MDD(R_scaled)

# 方法 2: raw returns → MDD → scale
mdd_raw = MDD(R_raw)
mdd_scaled = mdd_raw * sigma_target / sigma_raw
```

**输出**: `tests/mdd_scaling_test.py`

### Task 3.3: 财富路径可视化

**目标**: 直观对比我们的 wealth path 与 Paper 预期

**输出**: 
- `figures/wealth_path_comparison_EqIdx.png`
- `figures/wealth_path_comparison_FixedInc.png`
- `figures/wealth_path_comparison_Forex.png`

---

## 🧪 阶段四：反向工程验证 (Week 4)

### Task 4.1: 使用 Paper 值反推

**思路**: 从 Paper 的 MDD 值反推可能的计算方法

**方法**:
1. 假设 Paper MDD = 0.124 (Fixed Income MACD)
2. 尝试找到 wealth path 使得 MDD = 0.124
3. 反推 initial wealth 或 scaling factor

**输出**: `tests/mdd_reverse_engineering.py`

### Task 4.2: 敏感性分析

**变量**:
- Initial wealth (N×W0, W0=1, W0=0.1, etc.)
- Scaling method (per-contract vs portfolio)
- Return type (additive vs multiplicative)

**输出**: `tests/mdd_sensitivity_analysis.py`

### Task 4.3: 边界案例测试

**测试用例**:
- 全正收益 → MDD 应为 0
- 全负收益 → MDD 应为 1 (100%)
- 单一大回撤 → 验证计算正确

**输出**: `tests/mdd_edge_cases.py`

---

## 📝 阶段五：文档与决策 (Week 5)

### Task 5.1: 撰写技术报告

**内容**:
- 所有测试方法的结果对比
- 最接近 Paper 的计算方法
- 剩余偏差的解释 (数据差异？)

**输出**: `docs/mdd_dd_final_report.md`

### Task 5.2: 更新代码

**行动**:
- [ ] 采用最佳计算方法更新 `metrics.py`
- [ ] 添加详细注释说明选择理由
- [ ] 添加引用 (Paper 章节、文献 [27]、行业标准)

**输出**: `metrics.py` v2.0

### Task 5.3: 论文中的说明

**如需保留偏差**:
- 在论文 Methodology 章节说明计算差异
- 在 Results 章节讨论可能的原因
- 作为 Limitation 提及

**输出**: `deck.md` 更新

---

## 📅 时间表

| Week | 阶段 | 关键交付物 |
|------|------|-----------|
| **Week 1** | Paper 公式梳理 | `paper_metrics_analysis.md` |
| **Week 2** | 计算方法实验 | `mdd_dd_methods_comparison.py` |
| **Week 3** | 数据一致性验证 | `mdd_aggregation_test.py` + figures |
| **Week 4** | 反向工程验证 | `mdd_reverse_engineering.py` |
| **Week 5** | 文档与决策 | `mdd_dd_final_report.md` + 代码更新 |

---

## ✅ 成功标准

| 标准 | 目标 | 当前 |
|------|------|------|
| MDD ≤15% 误差 | ≥8/12 | 1/12 ❌ |
| MDD 平均误差 | <30% | 122.7% ❌ |
| DD ≤15% 误差 | ≥10/12 | 6/12 ⚠️ |
| DD 平均误差 | <20% | 15.6% ⚠️ |

---

## 🔗 相关文件

- `metrics.py` — 当前实现
- `docs/replication_status.md` — 复现程度评估
- `tests/results/full_baseline_table3.csv` — 当前结果
- Paper: Zhang, Zohren, Roberts (2019) Section 4.4
- Reference [27]: Lim et al. (Deep Momentum Networks)

---

## 💡 可能的根本原因 (假设)

1. **Wealth 定义不同**: Paper 使用 multiplicative，我们使用 additive
2. **Scaling 时机不同**: Paper 先计算 MDD 再 scaling，我们先 scaling 再计算
3. **Aggregation 不同**: Paper 平均单合约 MDD，我们计算组合 MDD
4. **MAR 设置不同**: DD 使用 rf 而非 0 作为 MAR
5. **数据差异**: CLC 2026 vs Paper 2019 数据源不同导致真实 MDD 不同

---

**下一步行动**: 从 Task 1.1 开始，重读 Paper Section 4.4 并做详细笔记
