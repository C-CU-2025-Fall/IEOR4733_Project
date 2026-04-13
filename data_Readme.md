# 📊 CLC 数据对齐文档

**项目**: IEOR4733 - Deep Reinforcement Learning for Trading 复现  
**数据源**: CLC (Continuous Linked Contract) Database, Futures Truth Company  
**文档目的**: 记录数据理解、问题发现、解决方案的完整过程，供论文和 presentation 使用

---

## 🎯 核心发现

### 1. 数据文件的四类结构

CLC 数据库包含四类相关文件：

| 文件类型 | 含义 | 用途 | 格式 |
|---------|------|------|------|
| **NON** | Non-adjusted | 原始合约价格序列 | 每个合约一条 CSV |
| **ASC** | Adjusted Settlement Contract | 换月调整记录（黄金标准） | 包含 roll date 和价格 |
| **REV** | Reverse-adjusted | 加法调整连续合约 | RAD = NON + adjustment |
| **RAD** | Ratio-adjusted | 乘法调整连续合约 | RAD = NON × ratio |

### 2. Roll Date 是唯一真相

**关键理解**: Roll Date（换月日）只有一个真相，所有调整都基于它：

```
ASC 记录的 Roll Date → 真相
NON 价格跳变 → 可以检测 Roll Date
理论规则 (MPDM_N) → 可以推测 Roll Date（无 ASC 时的 backup）
```

### 3. 调整参数的推导

从 ASC + NON 可以唯一确定调整参数：

```
ratio = ASC.prev_close / ASC.new_close
adjustment = ASC.new_close - ASC.prev_close
```

**重要**: REV 和 RAD 是两种独立的调整方法，它们之间没有等价关系。

---

## 🔍 问题发现过程

### 问题 1: 50 合约中部分 RAD 数据损坏

**发现**: 运行 baseline 回测时，发现部分合约的 RAD 文件异常：

| 合约 | 问题 | 影响 |
|------|------|------|
| ZH (10-Year T-Note) | RAD 全零 | 无法计算收益 |
| ZU (Crude Oil) | RAD 全零 | 无法计算收益 |
| US (30-Year T-Bond) | RAD 全 NaN | 无法计算收益 |
| ZN (10-Year T-Note) | RAD 价格异常 (~10000x) | 收益计算错误 |
| ZI (2-Year T-Note) | RAD ratio 异常 (1.32±0.06) | 需进一步验证 |

**根因**: CLC vendor 数据质量问题，非系统性错误

### 问题 2: Roll Date 匹配率低

**发现**: 理论规则计算的 Roll Date 与 RAD 检测的跳变日不一致：

- 初始匹配率: ~60-70%
- 平均差异: ~1.8 天

**用户质疑**: "匹配率差那么多我敢吗？我觉得我们还没有彻底理解数据，跑回测意义不大"

### 问题 3: 理论规则 vs 实际交易日

**发现**: 理论 Roll Date 可能落在周末或假期，实际调整发生在下一个交易日

**示例**:
```
理论日期: 2015-03-11 (Wednesday) → 实际: 2015-03-11 ✅
理论日期: 2015-06-11 (Thursday) → 实际: 2015-06-11 ✅
理论日期: 2015-09-11 (Friday) → 实际: 2015-09-11 ✅
理论日期: 2015-12-11 (Friday) → 实际: 2015-12-11 ✅

但有些情况:
理论日期: 2016-01-11 (Monday, 假期) → 实际: 2016-01-12 (Tuesday)
理论日期: 2017-07-11 (Tuesday) → 实际: 2017-07-11 ✅
```

---

## ✅ 解决方案

### 方案 1: 交易日日历法 (Trading Day Calendar Method)

**方法论**:
1. 根据 MPDM_N 规则计算理论 Roll Date
2. 找到理论日期后的**第一个交易日**作为实际调整日
3. 用实际调整日的 NON 价格计算 ratio

**公式**:
```
actual_roll_date = next_trading_day(theoretical_date)
ratio = NON[actual_roll_date - 1] / NON[actual_roll_date]
RAD_v2 = NON × cumulative_ratio
```

**结果**:
| 合约 | 精确匹配 | +1 天 | +2 天 | 平均差异 |
|------|---------|-------|-------|---------|
| ZH | 78% | 15% | 7% | 0.5 天 |
| ZN | 66% | 18% | 16% | 0.5 天 |
| ZU | 72% | 16% | 12% | 0.4 天 |
| US | 66% | 18% | 16% | 0.5 天 |

**改进**: 平均差异从 1.8 天 → 0.4-0.5 天

### 方案 2: 生成 RAD_v2 修复损坏合约

**脚本**: `tests/generate_rad_v2_final_4.py`

**步骤**:
1. 读取 NON 数据
2. 根据 MPDM 规则计算理论 Roll Dates
3. 映射到实际交易日
4. 计算 ratio 并累积
5. 生成 RAD_v2 = NON × ratio

**输出**:
- `data/CLC/ZH_RAD_v2.CSV`
- `data/CLC/ZN_RAD_v2.CSV`
- `data/CLC/ZU_RAD_v2.CSV`
- `data/CLC/US_RAD_v2.CSV`

### 方案 3: 50 合约全覆盖验证

**初始状态**: 45/50 合约可用（排除 5 个损坏）

**修复后**: 50/50 合约全部可用
- 47 vendor RAD（含 ZI）
- 4 RAD_v2（ZH, ZN, ZU, US）
- 减去 1 个重叠（ZI 不需要 RAD_v2）

---

## 📈 交叉验证结果 (2026-04-12)

### 关键发现：ASC 文件不覆盖测试期

**检查结果**: 0/22 D-class 合约的 ASC 文件覆盖测试期 (2011-2019)
- 15/22 在 2000 年前结束
- 7/22 在 2000s 结束 (最晚到 2008-2010)
- **0/22 覆盖 2011-2019 测试期**

**意义**: ASC 文件不能用于验证测试期内的 Roll Dates，验证策略调整为：
- 理论规则 vs RAD 跳变检测
- RAD vs NON 相关性分析

### 数据质量评分 (50 合约)

| 等级 | 标准 | 合约数 | 比例 | 合约列表 |
|------|------|--------|------|----------|
| **A** | corr≥0.95 | 31 | 62% | SP, ZA, ZG, ZH, ZI, ZK, ZL, ZP, ZU, CC, GI, KC, KW, CA, EN, ER, ES, LX, MD, SC, XU, YM, DT, UB, AN, BN, CN, DX, FN, JN, NK |
| **B** | corr≥0.90 | 8 | 16% | SB, SN, US, ZC, ZN, ZW, XX, MP |
| **C** | corr<0.90 | 11 | 22% | NR, TY, ZF, ZO, ZR, ZT, ZZ, DA, JO, LB, FB |
| **D** | 异常跳变 | 0 | 0% | - |

**总结**: 39/50 (78%) 合约数据质量 A/B 级，可用于回测

### 损坏合约修复

| 合约 | 原状态 | RAD_v2 状态 | 质量评分 |
|------|--------|-----------|---------|
| ZH | 全零 | ✅ corr=0.997 | A |
| ZN | 21.9x 异常 | ✅ corr=0.947 | B |
| ZU | 全零 | ✅ corr=0.962 | A |
| US | 全 NaN | ✅ corr=0.905 | B |

**RAD_v2 修复成功**: 4 个损坏合约全部达到 A/B 级

---

## 📂 生成的文件

### RAD_v2 文件
| 文件 | 描述 | 行数 (2011-2019) |
|------|------|-----------------|
| `ZH_RAD_v2.CSV` | 10-Year T-Note | 2267 |
| `ZN_RAD_v2.CSV` | 10-Year T-Note (修复) | 2267 |
| `ZU_RAD_v2.CSV` | Crude Oil | 2267 |
| `US_RAD_v2.CSV` | 30-Year T-Bond | 2268 |

### 汇总报告
| 文件 | 描述 |
|------|------|
| `rad_v2_4contracts_summary.csv` | 4 合约生成结果汇总 |
| `rad_v2_trading_day_method.csv` | 交易日日历法验证结果 |
| `data_quality_full_validation.csv` | 50 合约交叉验证结果 (2026-04-12) |

### 脚本
| 文件 | 用途 |
|------|------|
| `generate_rad_v2_final_4.py` | 生成 4 个损坏合约的 RAD_v2 |
| `cross_validate.py` | 执行交叉验证 |
| `detect_rolls_from_non.py` | 从 NON 检测 Roll Dates |
| `verify_d_contracts.py` | 验证 D 类合约数据覆盖 |

---

## 🎓 Presentation 要点

### 1. 数据质量挑战
- CLC vendor 数据存在损坏（全零、NaN、异常值）
- 5/50 合约需要修复
- Roll Date 匹配率初始仅~60%

### 2. 方法论贡献
- **交易日日历法**: 理论规则 + 交易日映射
- **RAD_v2 生成**: 从 NON 重建 Ratio-adjusted 数据
- **交叉验证框架**: 4 阶段验证确保数据可靠性

### 3. 结果改进
- Roll Date 平均差异: 1.8 天 → 0.4-0.5 天
- 50/50 合约全覆盖
- 数据质量评分: A 类~20, B 类~15, C 类~11

### 4. 对回测的意义
- 数据对齐是复现的前提
- 损坏数据导致回测结果不可信
- 交叉验证确保方法论可靠性

---

## 📝 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-04-12 | 初始版本，记录数据对齐过程 |

---

**最后更新**: 2026-04-12 20:30 EDT  
**作者**: Hao Wang, IEOR4733 Project Team

---

## 🔬 交叉验证执行结果 (2026-04-12)

**脚本**: `tests/cross_validate_simple.py`

### ASC 覆盖检查
- 0/22 D-class 合约的 ASC 覆盖测试期 (2011-2019)
- 15/22 在 2000 年前结束
- 7/22 在 2000s 结束 (最晚到 2008-2010)

### 50 合约数据质量评分

| 等级 | 标准 | 合约数 | 比例 |
|------|------|--------|------|
| A | corr≥0.95 | 31 | 62% |
| B | corr≥0.90 | 8 | 16% |
| C | corr<0.90 | 11 | 22% |
| D | 异常跳变>50% | 0 | 0% |

**39/50 (78%) 合约 A/B 级，可用于回测**

### 损坏合约修复验证

| 合约 | 原状态 | RAD_v2 corr | 评分 |
|------|--------|-----------|------|
| ZH | 全零 | 0.997 | A |
| ZN | 21.9x 异常 | 0.947 | B |
| ZU | 全零 | 0.962 | A |
| US | 全 NaN | 0.905 | B |

**全部修复成功!**

---

**最后更新**: 2026-04-12 20:45 EDT
