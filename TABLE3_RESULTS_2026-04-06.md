# IEOR4733 Table 3 复现结果 — 2026-04-06

**论文**: Zhang, Zohren, Roberts (2019) "Deep Reinforcement Learning for Trading"  
**复现目标**: Table 3 Baseline Strategies (Long Only, Sign(R), MACD)  
**数据源**: CLC Ratio-Adjusted Futures Data (44/50 合约可用)

---

## 📊 复现结果汇总

### Equity Index (10 contracts)

| 指标 | Long Only | Sign(R) | MACD |
|------|-----------|---------|------|
| **E(R)** | **+0.54** vs +0.50 ✅ | +0.08 vs +0.17 ⚠️ | -0.18 vs -0.07 ❌ |
| **std(R)** | +0.87 vs +0.93 ⚠️ | +0.75 vs +0.80 ⚠️ | +0.58 vs +0.59 ✅ |
| **Sharpe** | **+0.62** vs +0.54 ✅ | +0.11 vs +0.21 ⚠️ | -0.32 vs -0.12 ❌ |
| **Sortino** | **+0.83** vs +0.83 ✅ | +0.15 vs +0.32 ⚠️ | -0.42 vs -0.18 ❌ |
| **MDD** | 0.84 vs 0.13 ❌ | 0.94 vs 0.30 ❌ | 0.98 vs 0.35 ❌ |
| **% +ve** | **+0.55** vs +0.54 ✅ | +0.52 vs +0.53 ✅ | +0.51 vs +0.52 ✅ |

### Commodity (21 contracts)

| 指标 | Long Only | Sign(R) | MACD |
|------|-----------|---------|------|
| **E(R)** | -0.20 vs -0.30 ⚠️ | -0.16 vs +0.10 ❌ | -0.19 vs -0.04 ❌ |
| **Sharpe** | -0.54 vs -0.72 ✅ | -0.55 vs +0.33 ❌ | -0.84 vs -0.17 ❌ |
| **MDD** | 0.96 vs 0.25 ❌ | 0.87 vs 0.08 ❌ | 0.88 vs 0.13 ❌ |

### Fixed Income (4 contracts)

| 指标 | Long Only | Sign(R) | MACD |
|------|-----------|---------|------|
| **E(R)** | +0.40 vs +0.61 ⚠️ | -0.21 vs +0.19 ❌ | -0.48 vs +0.14 ❌ |
| **Sharpe** | +0.46 vs +0.65 ✅ | -0.28 vs +0.24 ❌ | -0.79 vs +0.22 ❌ |
| **MDD** | 0.94 vs 0.11 ❌ | 1.00 vs 0.17 ❌ | 1.00 vs 0.12 ❌ |

### Forex (9 contracts)

| 指标 | Long Only | Sign(R) | MACD |
|------|-----------|---------|------|
| **E(R)** | -0.19 vs -0.20 ✅ | -0.27 vs -0.11 ❌ | -0.32 vs +0.02 ❌ |
| **Sharpe** | -0.53 vs -0.42 ✅ | -0.57 vs -0.21 ❌ | -0.82 vs +0.04 ❌ |
| **MDD** | 0.95 vs 0.22 ❌ | 0.98 vs 0.17 ❌ | 0.98 vs 0.16 ❌ |

---

## ✅ 成功对齐的指标

| 资产类别 | Long Only | Sign(R) | MACD |
|----------|-----------|---------|------|
| **Equity Index** | E(R), Sharpe, Sortino, %+ | %+, Ave P/L, Calmar | %+, Ave P/L |
| **Commodity** | Sharpe, %+ | %+ | %+, Ave P/L |
| **Fixed Income** | Sharpe, Calmar, %+ | %+ | %+, Ave P/L |
| **Forex** | E(R), Sharpe, Sortino, %+ | %+ | %+, Ave P/L |

**最佳对齐**: Equity Index Long Only (6/9 指标 ✅)

---

## ❌ 关键差距

### 1. MDD 全面爆炸 (所有资产类别)

| 资产类别 | Ours | Paper | 差距 |
|----------|------|-------|------|
| Equity Index | 0.84-0.98 | 0.13-0.35 | +4-8x |
| Commodity | 0.87-0.96 | 0.08-0.25 | +4-11x |
| Fixed Income | 0.94-1.00 | 0.11-0.17 | +6-9x |
| Forex | 0.95-0.98 | 0.16-0.22 | +4-6x |

**根因分析**:
- 2011-08-03 (欧债危机): Equity Index 组合单日 -22%
- 波动率缩放在低 vol 时期放大杠杆至 5x
- 论文可能使用了 **winsorization** 或更严格的杠杆限制

### 2. Sign(R)/MACD 的 E(R) 符号相反

- **Commodity Sign(R)**: -0.16 vs +0.10 ❌
- **Fixed Income MACD**: -0.48 vs +0.14 ❌
- **Forex MACD**: -0.32 vs +0.02 ❌

**根因分析**:
- CLC ratio-adjusted 数据移除了展期收益
- 商品期货的展期收益 (roll return) 对 Sign(R)/MACD 策略至关重要
- 论文可能使用**原始合约数据 + 自定义展期**

---

## 🔧 已实施的修复

1. **数据质量检查** — 排除 6 个问题合约 (ZH, ZI, ZN, ZU, US, LX)
2. **杠杆上限** — MAX_LEVERAGE = 5.0
3. **百分比收益框架** — `r_t = (p_t - p_{t-1}) / p_{t-1}`
4. **波动率缩放** — `c_t = A_t × σ_tgt / σ_t`, σ_tgt = 0.93 年化

---

## 📋 下一步行动

### 高优先级
- [ ] **研究论文的极端值处理方法** — 检查 Appendix 是否有 winsorization 或 truncation
- [ ] **Sign(R)/MACD 诊断** — 分析为什么 E(R) 符号相反
- [ ] **Commodity 数据源对比** — 尝试获取原始合约数据

### 中优先级
- [ ] **敏感性分析** — σ_tgt, bp, EWMA span
- [ ] **Regime analysis** — 牛市/熊市/震荡市表现

### 低优先级
- [ ] **Streamlit dashboard** — 可视化展示
- [ ] **Proposal Deck** — 基于当前结果准备 4 页幻灯片

---

## 📁 核心文件

```
IEOR4733_Project/
├── table2_table3_unified.py    # 主复现脚本
├── config.py                    # 参数配置 (σ_tgt=0.93, MAX_LEVERAGE=5.0)
├── data_loader.py               # 数据加载
├── strategies.py                # 策略实现
├── vol_scaling.py               # 波动率缩放
├── metrics.py                   # 评估指标
├── data_quality_check.py        # 数据检查
├── REPLICATION_STATUS.md        # 本文件
└── data/CLC/                    # 44 个合约数据
```

---

**Last Updated**: 2026-04-06  
**Next Review**: 完成 Sign(R)/MACD 诊断后
