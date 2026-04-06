# IEOR4733 论文复现 — Table 2/3 基线策略

**论文**: Zhang, Zohren, Roberts (2019) "Deep Reinforcement Learning for Trading"  
**复现目标**: Table 2 & Table 3 基线策略（Long Only, Sign(R), MACD）  
**当前日期**: 2026-04-06  
**Deadline**: 期末展示

---

## 📊 复现状态总览

| 资产类别 | Long Only | Sign(R) | MACD | 状态 |
|----------|-----------|---------|------|------|
| **Commodity (21)** | ✅ | ⏳ | ⏳ | 数据问题已解决 |
| **Equity Index (11)** | ✅ | ⏳ | ⏳ | 部分对齐 |
| **Fixed Income (4)** | ⏳ | ⏳ | ⏳ | 待运行 |
| **Forex (9)** | ⏳ | ⏳ | ⏳ | 待运行 |

---

## ✅ 已完成

### 1. 数据层
- ✅ CLC ratio-adjusted 数据集成 (45/50 合约可用)
- ✅ 数据质量检查脚本 (`data_quality_check.py`)
- ✅ 排除 5 个问题合约 (ZH, ZI, ZN, ZU, US)
- ✅ 日期对齐组合构建

### 2. 核心框架
- ✅ 百分比收益框架 (`r_t = (p_t - p_{t-1}) / p_{t-1}`)
- ✅ 波动率缩放 (Eq. 4): `c_t = A_t × σ_tgt / σ_t`
- ✅ 交易成本: `bp × |Δc|` (20 bps)
- ✅ 9 个评估指标 (`metrics.py`)

### 3. 策略实现
- ✅ Long Only: 永远持有 +1
- ✅ Sign(R): 252 天累计收益符号
- ✅ MACD: 三组时间尺度 (8,24), (16,48), (32,96)

### 4. 论文理解
- ✅ 公式 4 (reward function)
- ✅ 公式 13 (portfolio construction)
- ✅ Table 2 vs Table 3 区别
- ✅ 波动率缩放机制

---

## 🔍 关键发现

### 数据源差异
- **CLC ratio-adjusted 数据移除了展期收益**
- 论文可能使用**原始合约数据 + 展期效应**
- Commodity 组合结构性亏损 87%（缺失 roll return）

### 当前对齐情况 (Equity Index, 2026-04-06)
| 指标 | Ours | Paper | 差异 |
|------|------|-------|------|
| E(R) | +0.54 | +0.50 | +7% ✅ |
| Sharpe | +0.62 | +0.54 | +15% ✅ |
| Sortino | +0.83 | +0.83 | 0% ✅ |
| MDD | 0.84 | 0.13 | ❌ **极端事件放大** |

**MDD 差距根因**:
- 2011-08-03 (欧债危机): 组合单日 -22%
- 波动率缩放在低 vol 时期放大杠杆 (5-10x)
- 论文可能使用了 **winsorization** 或 **杠杆上限更严格**

---

## ⏳ 待完成

### 高优先级
- [ ] **完整运行所有资产类别** — Fixed Income, Forex
- [ ] **Sign(R) 分析** — 换手率、交易成本分解
- [ ] **MACD 分析** — 信号分布、时间尺度影响
- [ ] **数据源对比** — CLC vs 原始合约（如能获取）

### 中优先级
- [ ] **敏感性分析** — σ_tgt, bp, EWMA span
- [ ] **稳健性检查** — 不同测试期、滚动窗口
- [ ] **Regime analysis** — 牛市/熊市/震荡市表现

### 低优先级
- [ ] **Streamlit dashboard** — 可视化展示
- [ ] **DQN 复现** — 论文主体 RL 算法

---

## 📚 论文阅读清单

### 必读（核心）
- [x] Zhang et al. (2019) — 主论文
- [ ] Baz et al. (2015) — MACD 信号定义 [4]
- [ ] Lim et al. (2019) — 波动率缩放 [27]

### 选读（扩展）
- [ ] 论文引用列表中的其他 DRL 交易论文

---

## 📁 核心文件

```
IEOR4733_Project/
├── table2_table3_unified.py    # 主复现脚本
├── config.py                    # 参数配置
├── data_loader.py               # 数据加载
├── strategies.py                # 策略实现
├── vol_scaling.py               # 波动率缩放
├── metrics.py                   # 评估指标
├── requirements.md              # 课程要求（已恢复）
├── data_quality_check.py        # 数据检查
├── DATA_QUALITY_FINAL_REPORT.md # 数据报告
└── data/CLC/                    # 45 个合约数据
```

---

## 🎯 下一步行动

1. **运行完整复现** — 所有资产类别 Table 3
2. **分析差距** — 逐个指标对比论文
3. **记录发现** — 更新本文件
4. **准备 Proposal Deck** — 基于复现结果

---

**Last Updated**: 2026-04-06
