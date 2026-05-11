# IEOR4733_Project — Deep Reinforcement Learning for Trading

Reproduction of Zhang, Zohren, Roberts (2019), with the repo condensed to the current core:

- one **live baseline** that keeps all Equity / Forex contracts
- one **experimental adjusted upper bound** that reaches `41/45`
- one **reporting-world audit line** explaining why clean same-rule still stalls below `40+/45`

Paper: [arXiv PDF](https://arxiv.org/pdf/1911.10107)

> Read [PROJECT_MEMORY.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/PROJECT_MEMORY.md) first if you are resuming work.

## 🎓 部署应用 - 交易策略模拟平台

> **新增功能!** 现已支持完整的 Streamlit Web 应用，满足教师要求的交互式部署。

### 启动 Streamlit 应用

```bash
# 方法 1: 使用启动脚本
chmod +x run_app.sh
./run_app.sh

# 方法 2: 手动启动
streamlit run src/app/main.py
```

访问: **http://localhost:8501**

### ✨ 应用功能

| 功能 | 说明 |
|------|------|
| **清洁数据管道** | 自动从 `data/CLC/` 加载、验证、清洁数据 |
| **回测引擎** | 支持 Long Only, Sign(R), MACD 三种策略 |
| **交易成本模型** | 自动计算滚动成本、头寸调整成本 |
| **性能仪表板** | 关键指标: 收益率、Sharpe 比、最大回撤 |
| **风险分析** | 波动率、最大回撤对比、可视化图表 |
| **参数优化** | 敏感性分析界面，实时计算参数影响 |

📖 **详细使用说明**: 请参考 [STREAMLIT_GUIDE.md](STREAMLIT_GUIDE.md)

---

## Quick Start

```bash
pip install numpy pandas yfinance

# Live baseline: all 50 contracts included
python baseline_run.py --table 3 --all-metrics --sigma 0.058

# Historical baseline rebuild under the cleaned source doctrine
python tests/historical_36x_rebuild_search.py

# Self-iterating reporting / Calmar alignment audit
python tests/calmar_alignment_iteration.py

# Enumerate clean same-rule vs 40+ experimental frontiers
python tests/frontier_40plus_enumeration.py

# One-line reproduction of the retained 41/45 experimental upper bound
python tests/run_legacy_41.py

# Long
python tests/run_legacy_41.py

# MACD
python baseline/macd/run_legacy_41_macd.py

# Sign(R)
python baseline/signr/run_legacy_41_signr.py

# 生成以上三种策略的table3格式的表格：
python generate_table3_comparison.py #快速查看版本，输出到控制台
python generate_table3_markdown.py #生成markdown文件

# Probe whether Yahoo-based ES/EN paths help when putting Equity back
python tests/equity_yf_rad_regen_probe.py

# ============================================================================
# 深度强化学习模型 (DQN, PG, A2C) 新增功能
# ============================================================================

# 训练所有 RL 模型 (DQN + PG + A2C)
python rl_models/train_all_rl_models.py

# 训练单个模型
python rl_models/train_all_rl_models.py dqn    # DQN: Fixed Q-targets + Double DQN
python rl_models/train_all_rl_models.py pg     # PG:  Policy Gradient (Monte Carlo)
python rl_models/train_all_rl_models.py a2c    # A2C: Advantage Actor-Critic

# 回测已训练的 RL 模型
python rl_models/evaluate_rl_models.py

# 详细文档（包含算法解释、超参数、性能指标等）
# 参考 rl_models/RL_MODELS_GUIDE.md
```

## 深度强化学习模型 (新增)

已实现论文《Deep Reinforcement Learning for Trading》中的三个核心 RL 算法：

### DQN (Deep Q-Network)
- **特点**: 价值函数方法，使用 Fixed Q-targets + Double DQN 增强稳定性
- **超参数**: lr=0.0001, γ=0.3, batch_size=64, memory=5000, τ=1000
- **网络**: LSTM [64, 32] 层，Leaky-ReLU 激活

### PG (Policy Gradient)
- **特点**: 直接学习策略，基于 Monte Carlo 轨迹采样
- **超参数**: lr_actor=0.0001, γ=0.3
- **网络**: LSTM [64, 32] 层，Softmax 输出动作概率

### A2C (Advantage Actor-Critic)
- **特点**: 混合方法，结合 Actor（策略）和 Critic（价值）网络，实时更新
- **超参数**: lr_critic=0.001, lr_actor=0.0001, γ=0.3, batch_size=128
- **网络**: 双 LSTM [64, 32] 架构，分离 Actor 和 Critic

**详细说明请参考** [RL_MODELS_GUIDE.md](rl_models/RL_MODELS_GUIDE.md)

## Preserved Versions

### Version Table

| Version | One-line Command | `<=10 /45` | `<=15 /45` | Equity / Forex fully kept? | Same-rule? | Notes |
| --- | --- | ---: | ---: | --- | --- | --- |
| Live baseline | `python baseline_run.py --table 3 --all-metrics --sigma 0.058` | 25 | 31 | Yes | Yes | current default runtime |
| Clean same-rule max | `python tests/frontier_40plus_enumeration.py` | 29 | 34 | Yes | Yes | best clean interpretation under current doctrine |
| Cleaner experimental fallback | `python tests/run_legacy_41.py` + `JO -> RAD` probe | 35 | 40 | No | No | keeps the legacy upper-bound shape but removes `JO_REV` |
| Experimental upper bound | `python tests/run_legacy_41.py` | 36 | 41 | No | No | excludes `EN, ES, FB, ZA, ZO`; Equity-only `risk_price_non` |

### Paper Target (Table 3 Long)

| Asset | `E(R)` | `std(R)` | `DD` | `Sharpe` | `Sortino` | `MDD` | `Calmar` | `% +ve` | `Ave P/L` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Commodity | -0.298 | 0.412 | 0.258 | -0.723 | -1.152 | 0.248 | -0.130 | 0.473 | 0.987 |
| Equity Index | 0.504 | 0.928 | 0.606 | 0.543 | 0.831 | 0.127 | 0.466 | 0.541 | 0.928 |
| Fixed Income | 0.605 | 0.939 | 0.561 | 0.645 | 1.081 | 0.108 | 0.455 | 0.515 | 1.048 |
| Forex | -0.198 | 0.472 | 0.285 | -0.420 | -0.696 | 0.219 | -0.101 | 0.491 | 0.966 |
| All | -0.013 | 0.363 | 0.230 | -0.036 | -0.057 | 0.037 | -0.009 | 0.519 | 0.919 |

### 9-Metric Alignment Tables

Below are the current retained Table 3 Long alignment tables against the paper targets.

#### Live Baseline

| Asset | `<=15 /9` | `E(R)` | `std(R)` | `DD` | `Sharpe` | `Sortino` | `MDD` | `Calmar` | `% +ve` | `Ave P/L` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Commodity | 6/9 | -0.232 | 0.374 | 0.256 | -0.621 | -0.907 | 0.626 | -0.121 | 0.491 | 0.938 |
| Equity Index | 6/9 | 0.526 | 0.839 | 0.660 | 0.627 | 0.798 | 0.149 | 0.344 | 0.548 | 0.919 |
| Fixed Income | 6/9 | 0.471 | 0.854 | 0.556 | 0.552 | 0.847 | 0.123 | 0.267 | 0.529 | 0.975 |
| Forex | 9/9 | -0.173 | 0.423 | 0.273 | -0.409 | -0.635 | 0.220 | -0.090 | 0.490 | 0.972 |
| All | 4/9 | 0.037 | 0.331 | 0.232 | 0.111 | 0.157 | 0.259 | -0.056 | 0.522 | 0.933 |

Main misses at `<=15%`:
- Commodity: `E(R)`, `Sortino`, `MDD`
- Equity Index: `Sharpe`, `MDD`, `Calmar`
- Fixed Income: `E(R)`, `Sortino`, `Calmar`
- Forex: none
- All: `E(R)`, `Sharpe`, `Sortino`, `MDD`, `Calmar`

#### Clean Same-Rule Max

| Asset | `<=15 /9` | `E(R)` | `std(R)` | `DD` | `Sharpe` | `Sortino` | `MDD` | `Calmar` | `% +ve` | `Ave P/L` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Commodity | 4/9 | -0.198 | 0.377 | 0.258 | -0.525 | -0.768 | 0.431 | -0.097 | 0.494 | 0.940 |
| Equity Index | 7/9 | 0.523 | 0.839 | 0.659 | 0.624 | 0.794 | 0.146 | 0.324 | 0.547 | 0.920 |
| Fixed Income | 9/9 | 0.555 | 0.859 | 0.570 | 0.647 | 0.975 | 0.111 | 0.414 | 0.534 | 0.969 |
| Forex | 9/9 | -0.173 | 0.423 | 0.273 | -0.409 | -0.635 | 0.220 | -0.090 | 0.490 | 0.972 |
| All | 5/9 | 0.075 | 0.342 | 0.244 | 0.219 | 0.306 | 0.196 | -0.004 | 0.528 | 0.926 |

Main misses at `<=15%`:
- Commodity: `E(R)`, `Sharpe`, `Sortino`, `MDD`, `Calmar`
- Equity Index: `Calmar`
- Fixed Income: none
- Forex: none
- All: `E(R)`, `Sharpe`, `Sortino`, `MDD`

#### Experimental Upper Bound (`41/45`)

| Asset | `<=15 /9` | `E(R)` | `std(R)` | `DD` | `Sharpe` | `Sortino` | `MDD` | `Calmar` | `% +ve` | `Ave P/L` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Commodity | 8/9 | -0.298 | 0.380 | 0.259 | -0.784 | -1.150 | 0.220 | 0.180 | 0.479 | 0.958 |
| Equity Index | 8/9 | 0.470 | 0.833 | 0.645 | 0.564 | 0.728 | 0.126 | 0.331 | 0.546 | 0.915 |
| Fixed Income | 9/9 | 0.555 | 0.859 | 0.570 | 0.647 | 0.975 | 0.111 | 0.414 | 0.534 | 0.969 |
| Forex | 9/9 | -0.173 | 0.423 | 0.273 | -0.409 | -0.635 | 0.220 | -0.109 | 0.490 | 0.972 |
| All | 7/9 | -0.013 | 0.327 | 0.228 | -0.038 | -0.055 | 0.125 | 0.300 | 0.515 | 0.934 |

Main misses at `<=15%`:
- Commodity: `Calmar`
- Equity Index: `Calmar`
- Fixed Income: none
- Forex: none
- All: `MDD`, `Calmar`

#### Cleaner Experimental Fallback (`40/45`)

This is the closest retained fallback if you want to reduce one dirty `REV` dependency while staying at `40+`:
- start from the `41/45` upper bound
- change only:
  - `JO: REV -> RAD`

| Asset | `<=15 /9` | `E(R)` | `std(R)` | `DD` | `Sharpe` | `Sortino` | `MDD` | `Calmar` | `% +ve` | `Ave P/L` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Commodity | 8/9 | -0.294 | 0.378 | 0.258 | -0.778 | -1.141 | 0.231 | 0.133 | 0.476 | 0.969 |
| Equity Index | 8/9 | 0.470 | 0.833 | 0.645 | 0.564 | 0.728 | 0.126 | 0.331 | 0.546 | 0.915 |
| Fixed Income | 9/9 | 0.555 | 0.859 | 0.570 | 0.647 | 0.975 | 0.111 | 0.414 | 0.534 | 0.969 |
| Forex | 9/9 | -0.173 | 0.423 | 0.273 | -0.409 | -0.635 | 0.220 | -0.109 | 0.490 | 0.972 |
| All | 6/9 | -0.011 | 0.327 | 0.228 | -0.034 | -0.049 | 0.130 | 0.254 | 0.515 | 0.938 |

Main misses at `<=15%`:
- Commodity: `Calmar`
- Equity Index: `Calmar`
- Fixed Income: none
- Forex: none
- All: `E(R)`, `MDD`, `Calmar`

### 1. Live Baseline

This is the baseline version that **keeps all Equity / Forex contracts** and remains the default runtime:

```bash
python baseline_run.py --table 3 --all-metrics --sigma 0.058
```

Current score:
- `<=10: 25/45`
- `<=15: 31/45`

Properties:
- exclusions: none
- reporting bridge: `RISK_PRICE_SIGMA0`
- default reporting numerator in the live CLI is still `wealth_cagr`
- this is the safest “all contracts retained” reference point

### 2. Clean Same-Rule Search Ceiling

This is the best retained **clean interpretation** line:

```bash
python tests/frontier_40plus_enumeration.py
```

Current clean same-rule max:
- `<=10: 29/45`
- `<=15: 34/45`

Interpretation:
- one global reporting rule
- one global numerator
- no asset-specific reporting override
- no negative-price-sensitive `REV` comeback

### 3. Experimental Adjusted Upper Bound

This is the retained **score-first adjusted version** that currently reaches `41/45`:

```bash
python tests/run_legacy_41.py
```

Representative `41/45` case:
- family: `legacy experimental upper bound`
- exclusions: `FB, ZA, ZO, EN, ES`
- Equity-only reporting: `risk_price_non`
- reporting extraction:
  - `annual_mean_sleeve`, or
  - `wealth_cagr`
- aggregation: `contract_equal_path`

Current score:
- `<=10: 36/45`
- `<=15: 41/45`

This version is intentionally preserved as:
- **experimental upper bound**
- not the promoted main interpretation

## Core Data Problems

These are the current repo-level conclusions after all retained sessions.

### 1. Negative-price `REV` cannot be treated as an active source in Eq. 4

Eq. 4 uses raw price level in the transaction-cost term:

\[
R_t = A_{t-1}\frac{\sigma_{tgt}}{\sigma_{t-1}}r_t - bp \cdot p_{t-1}\cdot |\Delta scaled\_pos|
\]

So if `p_{t-1} < 0`:
- transaction cost becomes economically invalid
- reporting capital anchor also becomes invalid

This is why the repo now treats these contracts cautiously:
- `CC`
- `LB`
- `JO`
- `ZH`
- `ZO`

### 2. Yahoo Finance behaves like `NON`, not like adjusted continuous prices

The retained local Yahoo probes show:
- Yahoo ≈ `CLC NON`
- Yahoo is **not** a replacement for `RAD`
- Yahoo is **not** a replacement for negative-price `REV`

For `ES/EN`, local Yahoo mapping is:
- `ES ↔ ES=F`
- `EN ↔ NQ=F`

And even after building Yahoo-based `YF_RAD_REGEN` paths, putting `EN/ES` back still does **not** recover `40+/45`.

Reference:
- [docs/equity_yf_rad_regen_probe.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/docs/equity_yf_rad_regen_probe.md)

### 3. Clean same-rule still stalls below `40+/45`

After the retained reporting-world iteration:
- the strongest clean same-rule line is still below `40+/45`
- the current clean ceiling is `34/45`
- the current live baseline is `31/45`

So at the moment:
- **there is no clean same-rule 40+ frontier in this repo**
- the only reproducible `40+` cases are experimental upper-bound families

### 4. Reporting diagnosis: `MDD aligned, numerator wrong`

The retained Calmar alignment loop concluded:
- `MDD` is relatively close after Commodity cleanup
- the main remaining mismatch is the **reporting annual return numerator**
- the best same-path winner in the retained audit is:
  - `annual_mean_simple`

Reference:
- [docs/calmar_alignment_iteration.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/docs/calmar_alignment_iteration.md)

## Retained Core Files

The repo was condensed. The key retained exploration files are:

- [tests/historical_36x_rebuild_search.py](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/tests/historical_36x_rebuild_search.py)
- [tests/calmar_alignment_iteration.py](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/tests/calmar_alignment_iteration.py)
- [tests/frontier_40plus_enumeration.py](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/tests/frontier_40plus_enumeration.py)
- [tests/equity_yf_rad_regen_probe.py](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/tests/equity_yf_rad_regen_probe.py)

And their retained reports:

- [docs/historical_36x_rebuild_search.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/docs/historical_36x_rebuild_search.md)
- [docs/calmar_alignment_iteration.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/docs/calmar_alignment_iteration.md)
- [docs/frontier_40plus_enumeration.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/docs/frontier_40plus_enumeration.md)
- [docs/equity_yf_rad_regen_probe.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/docs/equity_yf_rad_regen_probe.md)

Older one-off search artifacts were removed after their conclusions were merged into [PROJECT_MEMORY.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/PROJECT_MEMORY.md).

## Minimal Working Interpretation

- **Trade world** owns:
  - `E(R), std(R), DD, Sharpe, Sortino, % +ve, Ave P/L`
- **Reporting world** owns:
  - `MDD, Calmar`
- current reporting bridge:
  - `RISK_PRICE_SIGMA0`
- current clean reporting takeaway:
  - `MDD` is usable
  - `Calmar` is still definition-sensitive
  - `annual_mean_simple` is the best retained same-path numerator candidate

## Current Recommendation

Use the repo in this order:

1. live baseline for “all contracts retained” reference
2. clean same-rule frontier for interpretable search ceiling
3. experimental `41/45` frontier only as upper bound

If you need to continue research later, start from:
- [PROJECT_MEMORY.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/PROJECT_MEMORY.md)
- then [docs/frontier_40plus_enumeration.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/docs/frontier_40plus_enumeration.md)
