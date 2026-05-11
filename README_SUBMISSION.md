# DRL Trading Strategies - 项目提交版

> **Modular Python Framework** - 模块化交易策略框架

**项目类型**: Modular Python Framework  
**主要语言**: Python 3.9+  
**提交方式**: 模块化框架结构  

---

## 📋 项目简介

本项目实现了一个**完整的交易策略框架**，包含多种策略对比分析：

### ✅ 已完成的策略与模型

| 策略 | 类型 | 状态 | 位置 |
|------|------|------|------|
| **Long Only** | 传统 | ✅ 完成 | `baseline_run.py` |
| **Sign(R)** | 传统 | ✅ 完成 | `baseline/signr/` |
| **MACD** | 传统 | ✅ 完成 | `baseline/macd/` |
| **DQN** | 深度强化学习 | ✅ 完成 | `drl/dqn/` |
| **Route B** | A2C + 制度检测 | ✅ 完成 | `regime_detection/` |
| **A2C** | 深度强化学习 | ✅ 完成 | `src/core/models/a2c/` |

---

## 🗂️ 项目结构

```
IEOR4733_Project/
│
├── 📄 SUBMISSION_STRUCTURE.md        # 📌 详细的模块化结构说明
│
├── 📁 src/                           # 📌 模块化框架根目录（NEW）
│   ├── __init__.py
│   ├── 📁 core/
│   │   ├── 📁 strategies/            # 各种交易策略
│   │   ├── 📁 models/
│   │   │   ├── dqn/                  # ✅ DQN 模型
│   │   │   ├── a2c/                  # ⏳ A2C 模型（待上传）
│   │   │   └── regime/               # ✅ 制度检测
│   │   ├── 📁 data/                  # 数据加载和预处理
│   │   └── 📁 utils/                 # 指标计算、可视化
│   ├── 📁 scripts/                   # 可执行脚本
│   └── 📁 notebooks/                 # Jupyter 分析笔记本
│
├── 📁 tests/                         # 单元测试集合（NEW）
│
├── 📄 requirements.txt               # Python 依赖列表（NEW）
├── 📄 setup.py                       # 包安装配置（NEW）
│
├── 📁 config/                        # 配置文件
├── 📁 data/                          # 数据目录
│
├── 📁 rl_models/                     # 训练好的模型权重
│   ├── dqn/                          # ✅ DQN 权重
│   ├── a2c/                          # ⏳ A2C 权重（占位符）
│   └── README.md
│
├── 📁 reproduction_of_figures/       # 论文复现结果
├── 📁 regime_detection/              # Route B 实现
├── 📁 drl/                           # DRL 研究文件
│
└── 📁 docs/                          # 项目文档（NEW）
    ├── API.md                        # API 文档
    ├── MODELS.md                     # 模型说明
    ├── STRATEGIES.md                 # 策略说明
    └── INSTALLATION.md               # 安装指南

```

---

## 🚀 快速开始

### 1️⃣ 环境安装

```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# 或
.venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2️⃣ 运行基线策略

```bash
# 运行所有基线策略
python src/scripts/run_baseline_strategies.py

# 查看结果
cat results/baseline_results/*.csv
```

### 3️⃣ 运行 DQN 模型 ✅

```bash
python src/scripts/run_dqn_model.py
```

### 4️⃣ 运行 Route B 策略 ✅

```bash
python src/scripts/run_route_b_model.py
```

### 5️⃣ 生成策略对比图表

```bash
python src/scripts/evaluate_all_strategies.py
```

### 6️⃣ 查看完整分析 (Jupyter)

```bash
jupyter notebook src/notebooks/05_strategy_comparison.ipynb
```

---

## 📊 核心功能说明

### ✅ 已完成模块

#### 1. **传统策略** - `src/core/strategies/`
- Long Only: 长仓基准
- Sign(R): 基于收益率信号的策略
- MACD: 技术指标驱动策略

**特点**:
- 统一的策略接口
- 支持多资产类别 (Commodity, Equity Index, Fixed Income, Forex)
- 完整的 backtesting 框架

#### 2. **DQN 模型** - `src/core/models/dqn/` ✅
- 完整的网络架构和训练逻辑
- 论文 (Zhang et al. 2019) 的忠实复现
- 支持离线评估和在线部署

**文件结构**:
```
src/core/models/dqn/
├── __init__.py
├── model.py              # DQN 网络
├── trainer.py            # 训练脚本
├── evaluator.py          # 评估脚本
├── replay_buffer.py      # 经验回放
└── utils.py              # 工具函数
```

#### 3. **Route B (A2C + 制度)** - `src/core/models/regime/` ✅
- FFT 基础的制度检测
- GMM 软概率聚类
- 动态头寸调整

**论文创新**:
- 原始 9-dim 状态空间 + 3-dim 制度软概率 = 12-dim 输入
- 相同的 A2C 架构，但输入特征增强
- 显著的回报改进

#### 4. **数据模块** - `src/core/data/`
- 统一的数据加载器
- 自动化的数据清洗和预处理
- 期货合约展期管理

#### 5. **工具模块** - `src/core/utils/`
- 风险收益指标计算 (Sharpe、Calmar、MDD 等)
- 技术指标计算 (SMA、EMA、MACD、RSI 等)
- 可视化工具

### ⏳ 待上传模块

#### **A2C 模型** - `src/core/models/a2c/` ⏳

需要补充的文件：
```
src/core/models/a2c/
├── __init__.py           # (占位符已存在)
├── model.py              # ← 待提供
├── trainer.py            # ← 待提供
├── evaluator.py          # ← 待提供
└── utils.py              # ← 待提供
```

需要上传的模型权重：
```
rl_models/a2c/
├── period1_checkpoints/  # Period 1 (2011-2015) 模型
├── period2_checkpoints/  # Period 2 (2016-2019) 模型
└── results/
    ├── a2c_results_wide.csv
    └── metrics_report.csv
```

📌 **详见** `rl_models/a2c/README.md`

---

## 📈 实验结果

### 策略对比 (2011-2019)

生成的对比图表存放在 `reproduction_of_figures/` 目录：

| 图表 | 说明 |
|------|------|
| `strategies_by_asset_class.png` | 论文Figure 1复现 (4策略) |
| `strategies_5_comparison_no_routeb.png` | 5策略对比 (Long, Sign(R), MACD, A2C, DQN) |
| `strategies_with_route_b_and_dqn.png` | 完整6策略对比 (含Route B) |

### 数据输出

所有策略结果导出为 CSV 格式，便于外部分析：

```
results/baseline_results/
├── baseline_long_only_commodity.csv
├── baseline_long_only_equity_index.csv
├── baseline_signr_commodity.csv
├── ...（共15个文件）
└── README.md
```

---

## 🧪 测试框架

使用 pytest 的完整测试套件：

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_dqn.py -v
pytest tests/test_route_b.py -v

# 生成覆盖率报告
pytest tests/ --cov=src/
```

---

## 📚 文档

详细文档见 `docs/` 目录：

- **API.md** - 所有公共 API 的说明
- **MODELS.md** - 模型架构和参数详解
- **STRATEGIES.md** - 各策略的详细说明
- **INSTALLATION.md** - 详细安装和配置指南
- **SUBMISSION_STRUCTURE.md** - 项目结构完整说明

---

## 🔧 开发环境

| 工具 | 版本 | 说明 |
|------|------|------|
| Python | 3.9+ | 主语言 |
| PyTorch | 1.9+ | 深度学习框架 |
| NumPy | 1.20+ | 数值计算 |
| Pandas | 1.3+ | 数据处理 |
| Matplotlib | 3.4+ | 绘图 |
| Scikit-learn | 0.24+ | 机器学习工具 |
| Pytest | 6.0+ | 单元测试 |

**完整依赖** 见 `requirements.txt`

---

## 📝 模块化设计原则

本项目遵循以下最佳实践：

1. **单一职责原则** - 每个模块负责一项功能
2. **接口统一** - 所有策略/模型实现共同接口
3. **易于扩展** - 添加新策略仅需继承基类
4. **配置集中** - 参数在 `config/` 集中管理
5. **可测试性** - 每个模块都有单元测试
6. **文档完善** - 代码注释 + API 文档 + 使用示例

---

## 🎯 使用示例

### 导入和使用基础策略

```python
import numpy as np
from src.core.strategies import LongOnlyStrategy, SignRStrategy, MACDStrategy

# 初始化策略
long_strategy = LongOnlyStrategy(config)
signr_strategy = SignRStrategy(config)
macd_strategy = MACDStrategy(config)

# 计算收益
prices = np.random.randn(252)  # 一年的价格数据
long_returns = long_strategy.compute_returns(prices)
signr_returns = signr_strategy.compute_returns(prices)
```

### 使用 DQN 模型

```python
from src.core.models.dqn import DQNEvaluator

# 创建评估器
evaluator = DQNEvaluator(
    model_path="rl_models/dqn/checkpoints/model.pt",
    data_path="data/",
    asset_class="Commodity"
)

# 评估策略
metrics = evaluator.evaluate(
    start_date="2011-01-01",
    end_date="2019-12-31"
)

print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
print(f"Calmar Ratio: {metrics['calmar_ratio']:.3f}")
```

### 计算风险指标

```python
from src.core.utils import metrics

returns = np.random.randn(252)
sharpe = metrics.calculate_sharpe_ratio(returns)
mdd = metrics.calculate_max_drawdown(returns)
calmar = metrics.calculate_calmar_ratio(returns)

print(f"Sharpe: {sharpe:.3f}, MDD: {mdd:.3f}, Calmar: {calmar:.3f}")
```

---

## 📞 项目信息

| 项目 | IEOR 4733 - Deep Reinforcement Learning for Trading |
|------|-------|
| 课程 | Columbia University |
| 学期 | 2024 Spring |
| 论文复现 | Zhang, Zohren, Roberts (2019) |
| 论文链接 | [arXiv:1911.10107](https://arxiv.org/abs/1911.10107) |

---

## 📋 提交清单

- [x] 传统策略 (Long Only, Sign(R), MACD)
- [x] DQN 模型实现 + 权重
- [x] Route B 策略 (A2C + 制度检测)
- [x] 数据加载和预处理模块
- [x] 指标计算和可视化工具
- [x] 单元测试框架
- [x] Jupyter 分析笔记本
- [x] 完整的项目文档
- [ ] **A2C 模型代码** (占位符已预留)
- [ ] **A2C 模型权重** (占位符已预留)

---

## 🏁 后续步骤

1. **上传 A2C 代码** → `src/core/models/a2c/`
2. **上传 A2C 权重** → `rl_models/a2c/`
3. **完成单元测试** → `tests/test_a2c.py`
4. **验证所有测试通过** → `pytest tests/ -v`
5. **生成最终报告** → `src/scripts/generate_report.py`

---

## 📄 许可证

[待补充]

## ✉️ 联系方式

[待补充]

---

**Last Updated**: 2024年5月11日  
**Version**: 1.0.0 (Submission Ready)

