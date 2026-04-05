# IEOR4733_Project - Deep Reinforcement Learning for Trading

复现论文：**"Deep Reinforcement Learning for Trading"** by Zhang, Zohren, and Roberts (Oxford, 2019)

📄 **Paper Link:** [https://arxiv.org/pdf/1911.10107](https://arxiv.org/pdf/1911.10107)

---

## 📊 数据说明

### CLC 官方数据（核心数据源）

`data/CLC/` 目录包含 95 个期货合约的 Pinnacle CLC 数据（论文官方数据源），文件格式为 `{TICKER}_RAD.CSV`。

> ⚠️ **重要**: `data/CLC/` 已通过 `.gitignore` 排除，**提交代码时不要提交这些数据文件**。

**资产类别分布:**
- **Commodity** (商品): 22 个合约 (CC, DA, GI, JO, KC, KW, LB, NR, SB, ZA, ZC, ZF, ZG, ZI, ZK, ZL, ZN, ZO, ZP, ZR, ZT, ZW, ZZ)
- **Equity Index** (股指期货): 11 个合约 (CA, EN, ER, ES, LX, MD, SC, SP, XU, XX, YM)
- **Fixed Income** (固定收益): 4 个合约 (DT, FB, TY, UB)
- **Forex** (外汇): 9 个合约 (AN, BN, CN, DX, FN, JN, MP, NK, SN)

**排除的合约:** ZH (Heating Oil Electronic), ZU (Crude Oil Electronic), US (T-Bonds Composite) - 这些合约在 2011-2019 期间无有效数据。

### 辅助数据

- `data/risk_free_rate.csv` - 无风险利率数据 (DTB3)
- `data/index_data.csv` - VIX 指数数据

---

## 🔧 核心脚本

### 训练与评估

| 脚本 | 说明 |
|------|------|
| `table2_table3_unified.py` | **主脚本**: 统一复现论文 Table 2 & Table 3，支持多种模式 |
| `table2_baselines.py` | Table 2 基线策略复现（Long-only, Sign, MACD） |
| `train_dqn_paper_aligned.py` | DQN 训练脚本（对齐论文参数） |

### 工具模块

| 脚本 | 说明 |
|------|------|
| `indicators.py` | 技术指标计算 (MACD, RSI, 波动率等) |
| `paper_components.py` | 论文核心组件实现 |

---

## 🚀 使用方法

### 运行 Table 3 (原始信号，无波动率缩放)

```bash
python table2_table3_unified.py
```

### 运行 Table 3 + 单个合约波动率缩放

```bash
python table2_table3_unified.py --per-contract
```

### 运行 Table 2 (完整两层波动率缩放)

```bash
python table2_table3_unified.py --per-contract --portfolio
```

### 自定义目标波动率

```bash
python table2_table3_unified.py --sigma-tgt 0.15
```

---

## 📁 项目结构

```
IEOR4733_Project/
├── data/
│   ├── CLC/              # 95 个期货合约 CLC 数据 (*_RAD.CSV)
│   ├── risk_free_rate.csv
│   └── index_data.csv
├── table2_table3_unified.py    # 主脚本
├── table2_baselines.py         # 基线策略
├── train_dqn_paper_aligned.py  # DQN 训练
├── indicators.py               # 技术指标
├── paper_components.py         # 论文组件
├── methodology.md              # 方法论说明
├── EXECUTION_PLAN.md           # 执行计划
└── README.md                   # 本文件
```

---

## 📝 关键参数

根据论文 Table 1:

- **交易成本**: 20 bps (`BP = 0.0020`)
- **目标波动率**: σ_tgt = 0.10 (年化 10%)
- **波动率估计窗口**: 60 天 EWMA
- **交易日**: 252 天/年
- **投资组合目标波动率**: 0.97 (Table 2)

---

## 📋 待办事项

### Phase 2: Post-Mid Term

- [ ] 完成 DQN 训练并对比基线结果
- [ ] 进行稳健性检验
- [ ] 准备最终演示文稿
- [ ] 完成代码文档和可复现性检查

---

## 📚 相关文档

- [methodology.md](methodology.md) - 详细方法论说明
- [EXECUTION_PLAN.md](EXECUTION_PLAN.md) - 项目执行计划
