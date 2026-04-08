# IEOR4733 MDD 差距分析 — 2026-04-06

## A. 论文 MDD 处理方法研究

### 论文明确定义 (Page 7)

> "MDD: maximum drawdown shows the maximum observed loss from any peak of a portfolio"

**关键发现**:
1. 论文**没有明确说明** MDD 计算的特殊处理
2. 没有提到 winsorization、truncation 或 clipping
3. Table 3 是 **"without portfolio-level volatility scaling"** (Appendix B)
4. 但**有 per-contract volatility scaling** (Equation 4)

### Equation 4 完整公式

```
R_t = μ [ A_{t-1} (σ_tgt/σ_{t-1}) r_t - bp p_{t-1} ( (σ_tgt/σ_{t-1})A_{t-1} - (σ_tgt/σ_{t-2})A_{t-2} ) ]
```

**关键**:
- `r_t = p_t - p_{t-1}` (additive profits, Section 3.2)
- `σ_tgt = 0.10` (in price-diff units, NOT percentage)
- 交易成本项有 `p_{t-1}` (价格)

### 我们的实现状态

| 组件 | 状态 | 备注 |
|------|------|------|
| 收益定义 | ✅ 加法收益 `p_t - p_{t-1}` | `data_loader.get_price_diffs()` |
| 波动率缩放 | ✅ `c_t = A_t × σ_tgt / σ_t` | `vol_scaling.scale_per_contract()` |
| 交易成本 | ✅ `bp × p_{t-1} × |Δc|` | 正确实现 |
| σ_tgt | ✅ 0.10 (加法框架) | 已修复 |
| MAX_LEVERAGE | ✅ 5.0 | 防止极端杠杆 |

**结论**: 代码实现正确，但 MDD 仍然爆炸。

---

## B. Sign(R)/MACD 的 E(R) 符号问题分析

### 现象

| 资产类别 | Sign(R) E(R) | MACD E(R) |
|----------|--------------|-----------|
| Commodity | -0.16 vs +0.10 ❌ | -0.19 vs -0.04 ❌ |
| Fixed Income | -0.21 vs +0.19 ❌ | -0.48 vs +0.14 ❌ |
| Forex | -0.27 vs -0.11 ❌ | -0.32 vs +0.02 ❌ |

### 根因假设

**假设 1: CLC 数据缺少展期收益 (Roll Return)**
- CLC ratio-adjusted 数据平滑了展期跳跃
- 但展期收益是商品期货 Sign(R)/MACD 策略的重要 alpha 来源
- 移除展期收益 → 策略信号失真

**假设 2: 论文使用了原始合约数据**
- 原始数据包含展期跳跃
- 展期收益对趋势跟踪策略 (Sign(R), MACD) 有正向贡献
- CLC 数据移除了这部分收益

**假设 3: 交易成本模型差异**
- 论文的交易成本可能更低
- 高频换手的 Sign(R)/MACD 对成本敏感

### 验证方法

1. **检查 CLC 数据的展期处理**
   - 对比 CLC ratio-adjusted vs 原始合约数据
   - 量化展期收益的影响

2. **分析 Sign(R) 换手率**
   - 计算平均持仓时间
   - 检查交易成本占比

3. **对比原始合约数据**
   - 如能获取，用原始数据复现
   - 验证展期收益假设

---

## C. MDD 爆炸的根因

### 已识别的极端事件

1. **2011-08-03 欧债危机**
   - Equity Index 组合单日 -22%
   - 所有合约同时大跌，无法分散

2. **1987-10-16 黑色星期一** (SP 数据)
   - 单日 -5.37%
   - 缩放后 -103% (杠杆 10x)

3. **LX 数据异常** (已排除)
   - 2026-01-19 单日 -88%
   - 数据错误，非真实事件

### 波动率缩放的双刃剑

**低波动时期**:
- σ_t 很小 → 杠杆很高 (接近 MAX_LEVERAGE = 5.0)
- 组合对极端事件极度敏感

**高波动时期**:
- σ_t 很大 → 杠杆很低
- 但极端事件本身已经很大

**结果**: 无论高低波动，MDD 都被放大

### 可能的解决方案

1. **更严格的杠杆上限** (MAX_LEVERAGE = 2.0 或 3.0)
   - 降低极端事件影响
   - 但会降低 E(R) 和 Sharpe

2. **Winsorization/截尾处理**
   - 限制单日收益在 ±X% 以内
   - 论文可能使用了但未明确说明

3. **平滑波动率估计**
   - 使用更长窗口 (120 天 vs 60 天)
   - 降低杠杆波动

4. **接受数据源差异**
   - CLC ratio-adjusted ≠ 论文数据
   - 在报告中明确说明

---

## D. 下一步行动

### 高优先级
- [ ] **测试更严格的杠杆上限** (MAX_LEVERAGE = 2.0, 3.0)
- [ ] **分析 Sign(R) 换手率** — 检查交易成本影响
- [ ] **记录数据源差异** — 在 Proposal Deck 中说明

### 中优先级
- [ ] **尝试 winsorization** — 限制单日收益 ±10%
- [ ] **获取原始合约数据** — 验证展期收益假设

### 低优先级
- [ ] **联系论文作者** — 询问数据处理细节
- [ ] **对比其他数据源** — Yahoo Finance, Quandl 等

---

**Last Updated**: 2026-04-06  
**Status**: MDD 根因已识别，待验证解决方案
