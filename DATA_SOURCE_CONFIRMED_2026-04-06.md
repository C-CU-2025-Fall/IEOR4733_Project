# 论文数据源确认 — 2026-04-06

## 论文明确定义 (Section 4.1, Page 6)

> "We use data on **50 ratio-adjusted continuous futures contracts** from the **Pinnacle Data Corp CLC Database** [10]. Our dataset ranges from 2005 to 2019, and consists of a variety of asset classes including commodity, equity index, fixed income and FX."

**测试期**: 2011-2019 (Section 4.1)

**训练方案**: 每 5 年重新训练，用截至当时的所有数据优化参数，然后固定参数运行下一个 5 年 out-of-sample

---

## 论文 50 合约列表 (Appendix A)

### Commodities (25)
CC, DA, GI, JO, KC, KW, LB, NR, SB, ZA, ZC, ZF, ZG, **ZH**, **ZI**, ZK, ZL, **ZN**, ZO, ZP, ZR, ZT, **ZU**, ZW, ZZ

### Equity Indexes (11)
CA, EN, ER, ES, **LX**, MD, SC, SP, XU, XX, YM

### Fixed Income (5)
DT, FB, TY, UB, **US**

### Forex (9)
AN, BN, CN, DX, FN, JN, MP, NK, SN

---

## 数据质量对比

| 合约 | 论文 | 我们的 CLC 数据 | 问题 |
|------|------|----------------|------|
| **LX** (FTSE 100) | ✅ 正常 | ❌ 2026-01-19 单日 -88% | 数据异常 |
| **ZN** (Natural Gas) | ✅ 正常 | ❌ 价格比率 10451x | 展期调整错误 |
| **ZI** (Silver) | ✅ 正常 | ❌ 最大跳空 4676% | 展期调整错误 |
| **ZH** (Heating Oil) | ✅ 正常 | ❌ 测试期无数据 | 数据缺失 |
| **ZU** (Crude Oil) | ✅ 正常 | ❌ 测试期无数据 | 数据缺失 |
| **US** (T-Bonds) | ✅ 正常 | ❌ 测试期无数据 | 数据缺失 |

**可用合约**: 44/50 (88%)

---

## 关键结论

### 1. 数据源一致
- 论文：**CLC ratio-adjusted continuous contracts**
- 我们：**CLC ratio-adjusted continuous contracts** ✅

### 2. 数据质量差异
- 论文的 CLC 数据是**干净的**
- 我们的 CLC 数据有**严重质量问题**

### 3. MDD 差距的根因
- **不是**方法论差异
- **是**数据质量问题
- LX 的 -88% 暴跌、ZN 的 10451x 价格比率等异常值导致 MDD 爆炸

### 4. Sign(R)/MACD 表现差的根因
- 数据质量差 → 策略信号失真
- 特别是 ZN、ZI 等商品期货的异常值影响 MACD 信号

---

## 解决方案

### 选项 1: 重新下载 CLC 数据 (推荐)
- 联系 Pinnacle Data 确认数据质量
- 重新下载 50 个合约的 ratio-adjusted 数据
- 验证数据完整性（无极端跳跃）

### 选项 2: 排除问题合约
- 当前已排除 6 个问题合约
- 在报告中明确说明数据质量限制
- 接受 44/50 合约的复现结果

### 选项 3: 使用原始合约数据
- 获取单个合约的原始数据
- 自己实现展期逻辑（rolling）
- 工作量较大，但可控制质量

---

## 下一步行动

1. **验证 CLC 数据源** — 检查下载/处理流程是否有问题
2. **联系 Pinnacle Data** — 确认数据质量问题
3. **在 Proposal Deck 中说明** — 数据质量限制及影响

---

**Last Updated**: 2026-04-06  
**Status**: 数据源已确认（CLC ratio-adjusted），但质量有差异
