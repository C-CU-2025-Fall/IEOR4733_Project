# DRL Trading Strategies - 项目提交结构

## 📋 项目概述

一个**模块化Python框架**，实现并比较多种交易策略，包括传统策略（Long Only、Sign(R)、MACD）、深度强化学习策略（DQN、Route B）、和 A2C 策略（待上传）。

---

## 🗂️ 项目结构

```
IEOR4733_Project/
│
├── 📄 README.md                          # 项目主文档
├── 📄 SUBMISSION_STRUCTURE.md            # 本文件
├── 📄 requirements.txt                   # Python 依赖
│
├── 📁 config/                            # 配置文件 - 集中管理所有参数
│   ├── contract_months.json              # 期货合约月份配置
│   └── current_roll_mapping.txt          # 合约展期映射
│
├── 📁 data/                              # 数据存放目录
│   ├── index_data.csv                    # 指数数据
│   ├── risk_free_rate.csv                # 无风险利率
│   ├── yahoo/                            # Yahoo Finance 数据
│   └── CLC/                              # CLCData 原始数据
│
├── 📁 src/                               # 源代码根目录（模块化框架）
│   │
│   ├── __init__.py
│   │
│   ├── 📁 core/                          # 核心模块
│   │   ├── __init__.py
│   │   │
│   │   ├── 📁 strategies/                # 交易策略实现
│   │   │   ├── __init__.py
│   │   │   ├── base.py                   # 策略基类
│   │   │   ├── long_only.py              # Long Only 策略
│   │   │   ├── signr.py                  # Sign(R) 策略
│   │   │   ├── macd.py                   # MACD 策略
│   │   │   ├── route_b.py                # Route B (A2C + Regime) 策略
│   │   │   └── dqn_paper.py              # DQN (Paper) 策略
│   │   │
│   │   ├── 📁 models/                    # 深度学习模型
│   │   │   ├── __init__.py
│   │   │   │
│   │   │   ├── 📁 dqn/                   # DQN 模型 ✅ 已完成
│   │   │   │   ├── __init__.py
│   │   │   │   ├── model.py              # DQN 网络结构
│   │   │   │   ├── trainer.py            # DQN 训练脚本
│   │   │   │   ├── evaluator.py          # DQN 评估脚本
│   │   │   │   ├── replay_buffer.py      # 经验回放缓冲
│   │   │   │   └── utils.py              # 工具函数
│   │   │   │
│   │   │   ├── 📁 a2c/                   # A2C 模型 ⏳ 待上传
│   │   │   │   ├── __init__.py
│   │   │   │   ├── model.py              # A2C 网络结构（待提供）
│   │   │   │   ├── trainer.py            # A2C 训练脚本（待提供）
│   │   │   │   ├── evaluator.py          # A2C 评估脚本（待提供）
│   │   │   │   └── utils.py              # 工具函数（待提供）
│   │   │   │
│   │   │   └── 📁 regime/                # 制度识别模块 (for Route B)
│   │   │       ├── __init__.py
│   │   │       ├── fft_detector.py       # FFT 制度检测
│   │   │       ├── gmm_detector.py       # GMM 制度检测
│   │   │       └── utils.py              # 工具函数
│   │   │
│   │   ├── 📁 data/                      # 数据模块
│   │   │   ├── __init__.py
│   │   │   ├── loader.py                 # 数据加载器
│   │   │   ├── preprocessor.py           # 数据预处理
│   │   │   └── contract_manager.py       # 期货合约管理
│   │   │
│   │   └── 📁 utils/                     # 通用工具模块
│   │       ├── __init__.py
│   │       ├── metrics.py                # 风险收益指标计算
│   │       ├── indicators.py             # 技术指标计算
│   │       ├── visualization.py          # 绘图工具
│   │       └── helpers.py                # 通用助手函数
│   │
│   ├── 📁 scripts/                       # 可执行脚本
│   │   ├── __init__.py
│   │   ├── run_baseline_strategies.py    # 运行传统策略
│   │   ├── run_dqn_model.py              # 运行 DQN 模型
│   │   ├── run_a2c_model.py              # 运行 A2C 模型（待实现）
│   │   ├── run_route_b_model.py          # 运行 Route B 模型
│   │   ├── evaluate_all_strategies.py    # 全策略评估
│   │   └── generate_report.py            # 生成对比报告
│   │
│   └── 📁 notebooks/                     # Jupyter 分析笔记本
│       ├── 01_data_exploration.ipynb     # 数据探索
│       ├── 02_baseline_strategies.ipynb  # 基线策略分析
│       ├── 03_dqn_analysis.ipynb         # DQN 分析
│       ├── 04_a2c_analysis.ipynb         # A2C 分析（待完成）
│       └── 05_strategy_comparison.ipynb  # 全策略对比
│
├── 📁 tests/                             # 单元测试
│   ├── __init__.py
│   ├── test_long_only.py
│   ├── test_signr.py
│   ├── test_macd.py
│   ├── test_dqn.py
│   ├── test_a2c.py                       # A2C 测试（待实现）
│   ├── test_route_b.py
│   └── conftest.py                       # pytest 配置
│
├── 📁 docs/                              # 文档目录
│   ├── API.md                            # API 文档
│   ├── MODELS.md                         # 模型文档
│   ├── STRATEGIES.md                     # 策略文档
│   ├── RESULTS.md                        # 实验结果文档
│   └── INSTALLATION.md                   # 安装指南
│
├── 📁 rl_models/                         # 训练好的模型权重
│   ├── README.md                         # 模型说明
│   │
│   ├── 📁 dqn/                           # DQN 模型权重 ✅ 已完成
│   │   ├── paper_figure1_*.csv           # 论文数据
│   │   ├── results/                      # 训练结果
│   │   └── checkpoints/                  # 模型权重
│   │
│   ├── 📁 a2c/                           # A2C 模型权重 ⏳ 待上传
│   │   ├── period1_checkpoints/          # Period 1 模型
│   │   ├── period2_checkpoints/          # Period 2 模型
│   │   └── results/                      # 评估结果
│   │
│   └── 📁 route_b/                       # Route B 模型权重
│       ├── regime_detection/             # 制度检测结果
│       └── results/                      # Route B 结果
│
├── 📁 results/                           # 实验结果输出
│   ├── baseline_results/                 # 基线策略结果 CSV
│   ├── strategy_comparison.png           # 策略对比图表
│   ├── metrics_report.csv                # 指标报告
│   └── analysis_report.html              # HTML 分析报告
│
├── 📁 reproduction_of_figures/           # 论文复现结果
│   ├── strategies_comparison.ipynb       # 复现主笔记本
│   ├── strategies_by_asset_class.png
│   ├── strategies_pure_rad_baseline.png
│   ├── strategies_5_comparison_no_routeb.png
│   └── strategies_with_route_b_and_dqn.png
│
├── 📁 regime_detection/                  # 制度检测 (Route B)
│   ├── timeseries_fft_regime.py          # FFT 制度检测实现
│   ├── route_b_experiment.ipynb          # Route B 实验
│   └── results/                          # 制度检测结果
│
├── 📁 references/                        # 参考文献
│   └── *.txt, *.pdf                      # 论文和资源
│
└── 📁 docs_old/                          # 旧文档归档（可选）
    └── ...

```

---

## ✅ 已完成的模块

### 1️⃣ **传统策略** ✅
- Long Only
- Sign(R) 
- MACD

### 2️⃣ **DQN 模型** ✅
- 模型架构、训练、评估完成
- 论文数据已整合
- 位置: `src/core/models/dqn/` 和 `rl_models/dqn/`

### 3️⃣ **Route B (制度感知)** ✅
- 基于 FFT 的制度检测
- A2C + GMM 软概率的位置定尺寸
- 位置: `src/core/models/regime/` 和 `regime_detection/`

---

## ⏳ 待上传的模块

### **A2C 模型** ⏳
需要提供以下文件到 `src/core/models/a2c/`:

```python
# a2c/model.py - A2C 网络结构
# a2c/trainer.py - 训练脚本
# a2c/evaluator.py - 评估脚本
# a2c/utils.py - 工具函数
```

**预期的文件结构:**
```
src/core/models/a2c/
├── __init__.py
├── model.py              # Actor-Critic 网络定义
├── trainer.py            # 训练逻辑
├── evaluator.py          # 离线评估
├── utils.py              # 数据预处理、日志等
└── README.md             # A2C 特定文档
```

**预期的模型权重存放:**
```
rl_models/a2c/
├── period1_checkpoints/
│   ├── a2c_Commodity_period1.pt
│   ├── a2c_Equity_Index_period1.pt
│   └── ...
├── period2_checkpoints/
│   └── ...
└── results/
    └── a2c_results_wide.csv
```

---

## 🚀 使用指南

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行基线策略
```bash
python src/scripts/run_baseline_strategies.py
```

### 运行 DQN 模型
```bash
python src/scripts/run_dqn_model.py
```

### 运行 Route B 模型
```bash
python src/scripts/run_route_b_model.py
```

### 生成对比报告
```bash
python src/scripts/evaluate_all_strategies.py
```

### 运行完整分析笔记本
```bash
jupyter notebook src/notebooks/05_strategy_comparison.ipynb
```

---

## 📊 项目特点

| 特性 | 状态 | 说明 |
|------|------|------|
| 传统策略 | ✅ | Long Only, Sign(R), MACD 已实现 |
| DQN 模型 | ✅ | 完整的训练、评估、论文复现 |
| A2C 模型 | ⏳ | 代码结构已预留，待上传实现 |
| Route B | ✅ | 基于 FFT + GMM 的制度感知策略 |
| 数据模块 | ✅ | 统一的数据加载、预处理框架 |
| 测试框架 | ✅ | pytest 单元测试体系 |
| 文档 | ✅ | 完整的 API 和使用说明 |
| 可视化 | ✅ | 策略对比图表、指标仪表板 |

---

## 📝 模块化设计原则

1. **单一职责** - 每个模块负责一项功能
2. **接口清晰** - 策略、模型使用统一接口
3. **易于扩展** - 添加新策略/模型只需实现基类
4. **配置集中** - 所有参数在 `config/` 集中管理
5. **可测试** - 每个模块都有单元测试
6. **文档完善** - API、模型、策略都有详细说明

---

## 🔗 相关文件映射

| 功能 | 位置 |
|------|------|
| 导入、配置 | `src/core/__init__.py` |
| 策略调用 | `src/core/strategies/` |
| 模型训练 | `src/core/models/*/trainer.py` |
| 数据加载 | `src/core/data/loader.py` |
| 指标计算 | `src/core/utils/metrics.py` |
| 可执行脚本 | `src/scripts/` |
| 分析笔记本 | `src/notebooks/` |
| 单元测试 | `tests/` |

---

## 🎯 下一步

1. **上传 A2C 代码** 到 `src/core/models/a2c/`
2. **上传 A2C 权重** 到 `rl_models/a2c/`
3. **完善 README.md** - 添加完整的项目说明
4. **创建 requirements.txt** - 列出所有依赖
5. **完成单元测试** - 确保所有模块正常工作

---

## 📄 许可证

[待补充]

## ✉️ 联系方式

[待补充]

