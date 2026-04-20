# IEOR4733_Project —— 面向交易的深度强化学习

本仓库是对 Zhang、Zohren、Roberts (2019) 的复现，并已收缩为当前这几个核心部分：

- 一个保留全部股票指数 / 外汇合约的在线基线版本
- 一个可达到 `41/45` 的实验性调整上界版本
- 一条用于解释为什么干净同规则设定仍然停留在 `40+/45` 以下的报告口径审计线

论文链接：[arXiv PDF](https://arxiv.org/pdf/1911.10107)

> 如果你是在继续之前的工作，请先阅读 [PROJECT_MEMORY.md](PROJECT_MEMORY.md)。

## 快速开始

```bash
pip install numpy pandas yfinance

# 在线基线：包含全部 50 个合约
python baseline_run.py --table 3 --all-metrics --sigma 0.058

# 在清洗后数据源原则下重建历史基线
python tests/historical_36x_rebuild_search.py

# 自迭代的 reporting / Calmar 对齐审计
python tests/calmar_alignment_iteration.py

# 枚举干净同规则前沿与 40+ 实验性前沿
python tests/frontier_40plus_enumeration.py

# 一行复现实验性保留上界 41/45
python tests/run_legacy_41.py

# 检查在重新纳入股票指数后，基于 Yahoo 的 ES/EN 路径是否有帮助
python tests/equity_yf_rad_regen_probe.py
```

## 保留版本

### 版本表

| 版本 | 一行命令 | `<=10 /45` | `<=15 /45` | 是否完整保留股票指数 / 外汇 | 是否同规则 | 说明 |
| --- | --- | ---: | ---: | --- | --- | --- |
| 在线基线 | `python baseline_run.py --table 3 --all-metrics --sigma 0.058` | 25 | 31 | 是 | 是 | 当前默认运行版本 |
| 干净同规则最大值 | `python tests/frontier_40plus_enumeration.py` | 29 | 34 | 是 | 是 | 当前原则下最好的干净解释 |
| 更干净的实验性回退版本 | `python tests/run_legacy_41.py` + `JO -> RAD` probe | 35 | 40 | 否 | 否 | 保留旧版上界形状，但移除了 `JO_REV` |
| 实验性上界 | `python tests/run_legacy_41.py` | 36 | 41 | 否 | 否 | 排除 `EN, ES, FB, ZA, ZO`；股票指数单独使用 `risk_price_non` |

### 论文目标值（Table 3 Long）

| 资产类别 | `E(R)` | `std(R)` | `DD` | `Sharpe` | `Sortino` | `MDD` | `Calmar` | `% +ve` | `Ave P/L` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 商品 | -0.298 | 0.412 | 0.258 | -0.723 | -1.152 | 0.248 | -0.130 | 0.473 | 0.987 |
| 股票指数 | 0.504 | 0.928 | 0.606 | 0.543 | 0.831 | 0.127 | 0.466 | 0.541 | 0.928 |
| 固定收益 | 0.605 | 0.939 | 0.561 | 0.645 | 1.081 | 0.108 | 0.455 | 0.515 | 1.048 |
| 外汇 | -0.198 | 0.472 | 0.285 | -0.420 | -0.696 | 0.219 | -0.101 | 0.491 | 0.966 |
| 全部 | -0.013 | 0.363 | 0.230 | -0.036 | -0.057 | 0.037 | -0.009 | 0.519 | 0.919 |

### 9 指标对齐表

下面给出当前保留的 Table 3 Long 与论文目标值之间的对齐结果。

#### 在线基线

| 资产类别 | `<=15 /9` | `E(R)` | `std(R)` | `DD` | `Sharpe` | `Sortino` | `MDD` | `Calmar` | `% +ve` | `Ave P/L` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 商品 | 6/9 | -0.232 | 0.374 | 0.256 | -0.621 | -0.907 | 0.626 | -0.121 | 0.491 | 0.938 |
| 股票指数 | 6/9 | 0.526 | 0.839 | 0.660 | 0.627 | 0.798 | 0.149 | 0.344 | 0.548 | 0.919 |
| 固定收益 | 6/9 | 0.471 | 0.854 | 0.556 | 0.552 | 0.847 | 0.123 | 0.267 | 0.529 | 0.975 |
| 外汇 | 9/9 | -0.173 | 0.423 | 0.273 | -0.409 | -0.635 | 0.220 | -0.090 | 0.490 | 0.972 |
| 全部 | 4/9 | 0.037 | 0.331 | 0.232 | 0.111 | 0.157 | 0.259 | -0.056 | 0.522 | 0.933 |

在 `<=15%` 阈值下的主要未命中项：

- 商品：`E(R)`、`Sortino`、`MDD`
- 股票指数：`Sharpe`、`MDD`、`Calmar`
- 固定收益：`E(R)`、`Sortino`、`Calmar`
- 外汇：无
- 全部：`E(R)`、`Sharpe`、`Sortino`、`MDD`、`Calmar`

#### 干净同规则最大值

| 资产类别 | `<=15 /9` | `E(R)` | `std(R)` | `DD` | `Sharpe` | `Sortino` | `MDD` | `Calmar` | `% +ve` | `Ave P/L` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 商品 | 4/9 | -0.198 | 0.377 | 0.258 | -0.525 | -0.768 | 0.431 | -0.097 | 0.494 | 0.940 |
| 股票指数 | 7/9 | 0.523 | 0.839 | 0.659 | 0.624 | 0.794 | 0.146 | 0.324 | 0.547 | 0.920 |
| 固定收益 | 9/9 | 0.555 | 0.859 | 0.570 | 0.647 | 0.975 | 0.111 | 0.414 | 0.534 | 0.969 |
| 外汇 | 9/9 | -0.173 | 0.423 | 0.273 | -0.409 | -0.635 | 0.220 | -0.090 | 0.490 | 0.972 |
| 全部 | 5/9 | 0.075 | 0.342 | 0.244 | 0.219 | 0.306 | 0.196 | -0.004 | 0.528 | 0.926 |

在 `<=15%` 阈值下的主要未命中项：

- 商品：`E(R)`、`Sharpe`、`Sortino`、`MDD`、`Calmar`
- 股票指数：`Calmar`
- 固定收益：无
- 外汇：无
- 全部：`E(R)`、`Sharpe`、`Sortino`、`MDD`

#### 实验性上界（`41/45`）

| 资产类别 | `<=15 /9` | `E(R)` | `std(R)` | `DD` | `Sharpe` | `Sortino` | `MDD` | `Calmar` | `% +ve` | `Ave P/L` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 商品 | 8/9 | -0.298 | 0.380 | 0.259 | -0.784 | -1.150 | 0.220 | 0.180 | 0.479 | 0.958 |
| 股票指数 | 8/9 | 0.470 | 0.833 | 0.645 | 0.564 | 0.728 | 0.126 | 0.331 | 0.546 | 0.915 |
| 固定收益 | 9/9 | 0.555 | 0.859 | 0.570 | 0.647 | 0.975 | 0.111 | 0.414 | 0.534 | 0.969 |
| 外汇 | 9/9 | -0.173 | 0.423 | 0.273 | -0.409 | -0.635 | 0.220 | -0.109 | 0.490 | 0.972 |
| 全部 | 7/9 | -0.013 | 0.327 | 0.228 | -0.038 | -0.055 | 0.125 | 0.300 | 0.515 | 0.934 |

在 `<=15%` 阈值下的主要未命中项：

- 商品：`Calmar`
- 股票指数：`Calmar`
- 固定收益：无
- 外汇：无
- 全部：`MDD`、`Calmar`

#### 更干净的实验性回退版本（`40/45`）

如果你希望在维持 `40+` 的同时减少一个不干净的 `REV` 依赖，这是当前最接近的保留回退版本：

- 从 `41/45` 上界版本出发
- 只做这一项修改：
  - `JO: REV -> RAD`

| 资产类别 | `<=15 /9` | `E(R)` | `std(R)` | `DD` | `Sharpe` | `Sortino` | `MDD` | `Calmar` | `% +ve` | `Ave P/L` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 商品 | 8/9 | -0.294 | 0.378 | 0.258 | -0.778 | -1.141 | 0.231 | 0.133 | 0.476 | 0.969 |
| 股票指数 | 8/9 | 0.470 | 0.833 | 0.645 | 0.564 | 0.728 | 0.126 | 0.331 | 0.546 | 0.915 |
| 固定收益 | 9/9 | 0.555 | 0.859 | 0.570 | 0.647 | 0.975 | 0.111 | 0.414 | 0.534 | 0.969 |
| 外汇 | 9/9 | -0.173 | 0.423 | 0.273 | -0.409 | -0.635 | 0.220 | -0.109 | 0.490 | 0.972 |
| 全部 | 6/9 | -0.011 | 0.327 | 0.228 | -0.034 | -0.049 | 0.130 | 0.254 | 0.515 | 0.938 |

在 `<=15%` 阈值下的主要未命中项：

- 商品：`Calmar`
- 股票指数：`Calmar`
- 固定收益：无
- 外汇：无
- 全部：`E(R)`、`MDD`、`Calmar`

### 1. 在线基线

这是当前的基线版本，**保留全部股票指数 / 外汇合约**，并且仍然是默认运行版本：

```bash
python baseline_run.py --table 3 --all-metrics --sigma 0.058
```

当前得分：

- `<=10: 25/45`
- `<=15: 31/45`

特点：

- exclusions：无
- reporting bridge：`RISK_PRICE_SIGMA0`
- 在线 CLI 默认的 reporting numerator 仍然是 `wealth_cagr`
- 这是当前“全部合约都保留”的最稳妥参考点

### 2. 干净同规则搜索上限

这是当前保留下来的最佳**干净解释**路线：

```bash
python tests/frontier_40plus_enumeration.py
```

当前干净同规则上限：

- `<=10: 29/45`
- `<=15: 34/45`

其解释含义是：

- 一个全局统一的 reporting rule
- 一个全局统一的 numerator
- 没有按资产类别单独覆盖 reporting
- 不重新启用对负价格敏感的 `REV`

### 3. 实验性调整上界

这是当前保留下来的**得分优先**调整版本，目前可以达到 `41/45`：

```bash
python tests/run_legacy_41.py
```

代表性的 `41/45` 设定：

- family：`legacy experimental upper bound`
- exclusions：`FB, ZA, ZO, EN, ES`
- 股票指数单独的 reporting：`risk_price_non`
- reporting extraction：
  - `annual_mean_sleeve`，或
  - `wealth_cagr`
- aggregation：`contract_equal_path`

当前得分：

- `<=10: 36/45`
- `<=15: 41/45`

这个版本被有意保留为：

- **实验性上界**
- 不是主推的主要解释版本

## 核心数据问题

这些是当前仓库在所有保留实验之后得到的仓库级结论。

### 1. 负价格 `REV` 不能在公式 4 中被当作有效数据源使用

公式 4 在交易成本项中使用的是原始价格水平：

$$
R_t = A_{t-1}\frac{\sigma_{tgt}}{\sigma_{t-1}}r_t - bp \cdot p_{t-1}\cdot |\Delta scaled\_pos|
$$

因此，如果 `p_{t-1} < 0`：

- 交易成本在经济意义上会失效
- 报告口径中的资本锚定也会失效

这就是为什么当前仓库会谨慎对待这些合约：

- `CC`
- `LB`
- `JO`
- `ZH`
- `ZO`

### 2. Yahoo Finance 的行为更像 `NON`，而不是调整后的连续价格

保留下来的本地 Yahoo 探针结果表明：

- Yahoo 约等于 `CLC NON`
- Yahoo 不能替代 `RAD`
- Yahoo 不能替代负价格 `REV`

对 `ES/EN` 而言，本地 Yahoo 映射是：

- `ES ↔ ES=F`
- `EN ↔ NQ=F`

即使构建了基于 Yahoo 的 `YF_RAD_REGEN` 路径，在重新加入 `EN/ES` 后，结果仍然无法恢复到 `40+/45`。

参考：

- [docs/equity_yf_rad_regen_probe.md](docs/equity_yf_rad_regen_probe.md)

### 3. 干净同规则设定仍然停留在 `40+/45` 以下

在保留的 reporting-world 迭代之后：

- 最强的干净同规则版本仍低于 `40+/45`
- 当前干净上限是 `34/45`
- 当前在线基线是 `31/45`

所以目前：

- **这个仓库里不存在干净同规则的 40+ 前沿**
- 唯一可复现的 `40+` 情况都属于实验性上界家族

### 4. Reporting 诊断：`MDD` 对齐了，但 numerator 不对

保留下来的 Calmar 对齐循环得出的结论是：

- 在商品数据清理后，`MDD` 已经比较接近
- 当前剩下的主要不匹配点是**报告口径中的年化收益 numerator**
- 在保留下来的同路径审计中，最佳候选是：
  - `annual_mean_simple`

参考：

- [docs/calmar_alignment_iteration.md](docs/calmar_alignment_iteration.md)

## 保留的核心文件

仓库已经被裁剪。当前保留的关键探索文件有：

- [tests/historical_36x_rebuild_search.py](tests/historical_36x_rebuild_search.py)
- [tests/calmar_alignment_iteration.py](tests/calmar_alignment_iteration.py)
- [tests/frontier_40plus_enumeration.py](tests/frontier_40plus_enumeration.py)
- [tests/equity_yf_rad_regen_probe.py](tests/equity_yf_rad_regen_probe.py)

以及对应保留的报告：

- [docs/historical_36x_rebuild_search.md](docs/historical_36x_rebuild_search.md)
- [docs/calmar_alignment_iteration.md](docs/calmar_alignment_iteration.md)
- [docs/frontier_40plus_enumeration.md](docs/frontier_40plus_enumeration.md)
- [docs/equity_yf_rad_regen_probe.md](docs/equity_yf_rad_regen_probe.md)

更早的一次性搜索产物，在其结论被合并进 [PROJECT_MEMORY.md](PROJECT_MEMORY.md) 后已经删除。

## 最小可工作解释

- **Trade world** 负责：
  - `E(R), std(R), DD, Sharpe, Sortino, % +ve, Ave P/L`
- **Reporting world** 负责：
  - `MDD, Calmar`
- 当前 reporting bridge：
  - `RISK_PRICE_SIGMA0`
- 当前干净 reporting 的结论：
  - `MDD` 是可用的
  - `Calmar` 仍然对定义敏感
  - `annual_mean_simple` 是当前保留下来的最佳同路径 numerator 候选项

## 当前建议

建议按下面顺序使用这个仓库：

1. 先用在线基线，作为“全部合约都保留”的参考版本
2. 再看干净同规则前沿，作为可解释的搜索上限
3. 最后仅把实验性 `41/45` 前沿当作上界参考

如果你之后还要继续研究，建议从这里开始：

- [PROJECT_MEMORY.md](PROJECT_MEMORY.md)
- 然后看 [docs/frontier_40plus_enumeration.md](docs/frontier_40plus_enumeration.md)