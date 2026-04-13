# MDD/DD 诊断报告

**日期**: 2026-04-12  
**状态**: DD 已修复，MDD 仍有偏差

---

## ✅ DD 修复 (已完成)

### Paper 定义 (Section 4.4)
> "annualised **standard deviation of trade returns that are negative**"

### 修复前 (错误)
```python
# Zero-target LPM(2) - 这不是 Paper 的定义
shortfall = np.minimum(R_eq, 0.0)
dd = np.sqrt(np.mean(shortfall ** 2)) * np.sqrt(T)
```

### 修复后 (正确)
```python
# Std of negative returns only - Paper 的定义
neg_returns = R_eq[R_eq < 0]
dd = np.std(neg_returns, ddof=0) * np.sqrt(T)
```

### 修复效果

| Asset | Strategy | Before | After | Paper | Err Before | Err After |
|-------|----------|--------|-------|-------|------------|-----------|
| Commodity | Long | 0.294 | **0.259** | 0.258 | 14.0% | **0.4%** ✅ |
| Equity Index | Long | 0.695 | 0.736 | 0.606 | 14.7% | 21.5% ⚠️ |
| Fixed Income | Long | 0.570 | **0.528** | 0.561 | 1.6% | **5.9%** ✅ |
| Forex | Long | 0.336 | **0.299** | 0.285 | 17.9% | **4.9%** ✅ |

**DD ≤15%**: 6/12 → **7/12** 改善！

---

## ❌ MDD 偏差 (待解决)

### 当前状态

| Asset | N | Our MDD | Paper MDD | Ratio | Error |
|-------|---|---------|-----------|-------|-------|
| Commodity | 24 | 0.102 | 0.248 | 0.41x | 58.9% ❌ |
| Equity Index | 11 | 0.113 | 0.127 | 0.89x | 11.0% ✅ |
| Fixed Income | 6 | 0.207 | 0.108 | 1.92x | 91.7% ❌ |
| Forex | 9 | 0.321 | 0.219 | 1.47x | 46.6% ❌ |

### 当前实现
```python
cumret = np.cumsum(R_eq)
wealth = n_contracts * w0 + cumret  # N × 1.0 + cumret
peak = np.maximum.accumulate(wealth)
mdd = float(np.max((peak - wealth) / peak))
```

### 已排除的假设

| 假设 | 验证结果 |
|------|---------|
| Table 3 用 portfolio-level vol scaling | ❌ Paper 明确说 Table 3 **没有** portfolio-level scaling |
| Paper 用 multiplicative wealth | ❌ 我们已确认 additive 框架正确 (std 匹配) |
| Paper 用 per-contract MDD 然后平均 | ⚠️ 需要进一步验证 |
| Paper 用不同的 initial wealth | ⚠️ 需要进一步验证 |
| 数据源差异 (CLC 2026 vs Paper 2019) | ⚠️ 可能，但无法解释系统性模式 |

### 观察到的模式

**MDD Ratio vs N 的关系**:
- N 越大 (Commodity 24) → Our MDD < Paper MDD (0.41x)
- N 中等 (Equity Index 11) → Our MDD ≈ Paper MDD (0.89x)
- N 较小 (Fixed Income 6, Forex 9) → Our MDD > Paper MDD (1.92x, 1.47x)

**这说明**: 我们的 MDD 计算对 N 敏感，Paper 的不敏感 (或敏感度不同)。

---

## 🔍 待验证假设

### 假设 1: Paper 使用不同的 wealth 公式

**可能性**: Paper 可能用 `wealth = 1 + cumret` (固定 initial wealth，不依赖 N)

**预测**: 
- 如果 `wealth = 1 + cumret`，MDD 会更大 (cumret 波动不被 N 稀释)
- 但这会使所有资产的 MDD 都变大，与观察不符

**验证方法**: 尝试 `wealth = 1 + cumret` 并对比结果

### 假设 2: Paper 使用 per-contract MDD 然后平均

**可能性**: Paper 可能先计算每个合约的 MDD，然后等权平均

**预测**:
- Per-contract MDD 不受 N 影响
- 平均 MDD ≈ 典型合约的 MDD

**验证方法**: 实现 per-contract MDD 并平均

### 假设 3: Paper 的 MDD 基于 cumulative return 而非 wealth

**可能性**: Paper 可能用 `max_drawdown(cumret) / |mean(cumret)|` 或其他变体

**预测**: 需要查看 Paper 补充材料或联系作者

### 假设 4: 数据差异导致真实 MDD 不同

**可能性**: CLC 2026 数据与 Paper 2019 数据不同，导致真实 MDD 不同

**预测**: 
- Equity Index 数据最稳定 → MDD 匹配最好 (11%)
- Fixed Income 数据变化大 → MDD 偏差最大 (91.7%)

**验证方法**: 检查各资产类别的数据质量等级与 MDD 误差的相关性

---

## 📋 下一步行动

### 高优先级 (本周)
1. **实现 per-contract MDD 方法** 并测试
2. **测试固定 initial wealth** (`wealth = 1 + cumret`)
3. **检查数据质量与 MDD 误差的相关性**

### 中优先级 (下周)
4. **重读 Paper Section 4.4 和补充材料** 寻找 MDD 定义细节
5. **查阅文献 [27] Lim et al.** 确认 MDD 定义
6. **联系作者** (如果上述方法都失败)

### 低优先级
7. **在论文中说明 MDD 计算差异** (作为 Limitation)

---

## 📊 当前总体复现程度

| 指标 | ≤15% | 平均误差 | 状态 |
|------|------|---------|------|
| E(R) | 4/12 | ~100% | ⚠️ 数据差异 |
| std(R) | **12/12** | **<5%** | ✅ 完美 |
| DD | **7/12** | **~15%** | ✅ 改善 |
| Sharpe | 4/12 | ~100% | ⚠️ 受 E(R) 影响 |
| MDD | **1/12** | **122.7%** | ❌ 严重偏差 |
| %+ve | **12/12** | **<5%** | ✅ 完美 |
| Ave P/L | **12/12** | **<10%** | ✅ 完美 |

**综合**: ~65% 指标 ≤15% 误差

---

## 🔗 相关文件
- `metrics.py` — 当前实现 (DD 已修复)
- `docs/mdd_dd_fix_plan.md` — 长期修复计划
- `tests/results/full_baseline_table3.csv` — 当前结果
