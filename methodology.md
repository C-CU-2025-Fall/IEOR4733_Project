# 📋 论文方法论 - Deep Reinforcement Learning for Trading

**论文**: Zhang, Zohren, Roberts (2019) - Oxford  
**arXiv**: https://arxiv.org/pdf/1911.10107

> **⚠️ 复现时必须严格对照本文档！**

---

## 当前实现状态 (2026-04-06)

### 方程4 实现（已完成）

```
R_t = A_{t-1} × (σ_tgt / σ_{t-1}) × r_t − bp × p_{t-1} × |Δ(scaled position)|
```

| 变量 | 定义 | 实现 | 量纲 |
|------|------|------|------|
| `p_t` | `Close[t] / Close[0]` (p0-normalized) | ✅ | 无量纲 |
| `r_t` | `p_t - p_{t-1}` (additive on normalized) | ✅ | 无量纲 |
| `σ_t` | `EWMA(span=60) std of r_t` | ✅ | 无量纲 |
| `σ_tgt` | `0.10 / √252 ≈ 0.0063` (daily, = 10% annual) | ✅ | 无量纲 |
| `A_t` | 策略 position ∈ {-1, 0, +1} | ✅ | 无量纲 |
| `bp` | `0.0020` (20 bps) | ✅ | 无量纲 |
| `cost` | `bp × p_{t-1} × |Δ(scaled_pos)|` | ✅ | 无量纲 |

### 指标计算（metrics.py，唯一来源）

| 指标 | 公式 | 备注 |
|------|------|------|
| E(R) | `mean(R) × 252` | 年化 |
| std(R) | `std(R) × √252` | 年化 |
| DD | `√mean(min(0,R)²) × √252` | 标准下方差 |
| Sharpe | `E(R) / std(R)` | |
| Sortino | `E(R) / DD` | |
| MDD | **rolling 252-day max** on NAV | max across all 1-year windows |
| Calmar | `E(R) / MDD` | |
| % +ve | `count(R>0) / count(R)` | |
| Ave P/L | `mean(R>0) / |mean(R<0)|` | |

### Portfolio 构建

```
每个合约: NAV_i = 100 × cumprod(1 + R_t^i)
总 NAV: NAV_total = Σ NAV_i
Portfolio returns: NAV_total.pct_change()
```

### Table 2 vs Table 3

| | Table 3 | Table 2 |
|---|---|---|
| Per-contract vol scaling | ✅ (Equation 4) | ❌ |
| Portfolio 构建 | NAV | NAV |
| Portfolio-level vol scaling | ❌ | ✅ `σ_tgt / σ_portfolio × R_portfolio` |
| σ_tgt | 10% annual (≈0.0063 daily) | same |

---

## 对齐结果 (σ_tgt_annual=10%, Equity Index)

### Table 3 (per-contract vol scaling)
| | E(R) | std | DD | Sharpe | Sortino | MDD | Calmar | %+ve | Ave P/L |
|---|---|---|---|---|---|---|---|---|---|
| Long ours | +0.057 | +0.094 | +0.070 | +0.607 | +0.812 | +0.163 | +0.351 | +0.552 | +0.902 |
| paper | +0.504 | +0.928 | +0.606 | +0.543 | +0.831 | +0.127 | +0.466 | +0.541 | +0.928 |
| | ❌88 | ❌89 | ❌88 | ✅11.8 | ✅2.3 | ⚠️28 | ⚠️24 | ✅2.0 | ✅2.8 |

### Table 2 (portfolio-level vol scaling)
| | E(R) | std | DD | Sharpe | Sortino | MDD | Calmar | %+ve | Ave P/L |
|---|---|---|---|---|---|---|---|---|---|
| Long ours | +0.089 | +0.098 | +0.071 | +0.909 | +1.262 | +0.132 | +0.676 | +0.550 | +0.951 |
| paper | +0.668 | +0.970 | +0.606 | +0.688 | +1.102 | +0.132 | +0.509 | +0.542 | +0.948 |
| | ❌86 | ❌89 | ❌88 | ❌32 | ✅14.5 | **✅0.0** | ❌32 | ✅1.5 | ✅0.3 |

### 已知问题
- std/E(R)/DD 绝对值偏低 ~90%（σ_tgt 的 scale 不对，论文的 std ≈ 0.97 annual）
- 比率指标（Sharpe/Sortino/%+ve/Ave P/L）对齐好
- Table 2 Long MDD = 0.132 精确匹配论文 ✅
- Sign(R)/MACD E(R) 方向/幅度仍有问题

### 待确定
- 论文 std ≈ 0.97 annual 对应 σ_tgt_daily ≈ 0.061（而非 0.0063）
- σ_tgt=0.064 时 std ≈ 0.97 但 MDD 崩到 0.9+
- 可能需要 σ_tgt ≈ 0.061 daily（对应 97% annual），MDD 用不同定义

---

## 代码结构

```
IEOR4733_Project/
├── config.py              # 参数（BP, σ_tgt, 资产分类, paper值）
├── data_loader.py         # CLC 数据加载
├── strategies.py          # Long/Sign(R)/MACD 信号
├── vol_scaling.py         # EWMA vol 计算
├── metrics.py             # 9 指标（唯一来源！）
├── baseline_backtest.py   # Table 2/3 主运行脚本
└── methodology.md         # 本文件
```

**规则**：
- 所有指标计算统一用 `metrics.py`，禁止在其他文件重复实现
- 参数统一在 `config.py`，不要 hardcode

---

## 1. 数据集 (Section 4.1)

| 项目 | 论文要求 |
|------|----------|
| **数据源** | Pinnacle Data Corp CLC Database |
| **合约数** | 50 个期货合约 |
| **时间范围** | 2005-2019 |
| **资产类别** | Commodity (23), Equity Index (11), Fixed Income (4), Forex (9) |
| **测试期** | 2011-2019 |

### CLC 数据质量
- 45/50 可用（排除 ZH/ZU/US 全零/NaN，ZI/ZN 严重跳变）
- LX 排除（2026-01-19 异常单日跌 88%）

---

## 2. 状态空间 (Section 3.1)

**时间窗口**: 60 天

**8 个特征**:
1. 归一化收盘价
2. 21 天收益率 (vol-adjusted)
3. 42 天收益率
4. 63 天收益率
5. 252 天收益率
6. MACD 指标 (Eq 3)
7. RSI (30 天)
8. 波动率 (60 天 EWMA)

**输出形状**: `(60, 8)` → LSTM 输入

---

## 3. 动作空间

| 模型 | 动作空间 |
|------|----------|
| DQN | `{-1, 0, 1}` 离散 |
| PG | `{-1, 0, 1}` 离散 |
| A2C | `[-1, 1]` 连续 |

---

## 4. 奖励函数 (公式 4)

```
R_t = A_{t-1} × (σ_tgt/σ_{t-1}) × r_t − bp × p_{t-1} × |A_{t-1}×σ_tgt/σ_{t-1} − A_{t-2}×σ_tgt/σ_{t-2}|
```

其中:
- `r_t = p_t - p_{t-1}` (additive profits on normalized prices)
- `σ_t = EWMA(60) std of r_t`
- `σ_tgt = 10% annual → 0.0063 daily`
- `bp = 0.0020`

---

## 5. 网络架构 (Section 4.3)

| 组件 | 规格 |
|------|------|
| 类型 | 两层 LSTM |
| 第一层 | 64 单元 |
| 第二层 | 32 单元 |
| 激活 | Leaky-ReLU (slope=0.01) |

---

## 6. 超参数 (Table 1)

### DQN
| 参数 | 值 |
|------|-----|
| α_critic | 0.0001 |
| Optimiser | Adam |
| Batch size | 64 |
| γ | 0.3 |
| bp | 0.0020 |
| Memory size | 5000 |
| τ (target update) | 1000 |

### A2C ⚠️
| 参数 | 值 |
|------|-----|
| α_critic | **0.001** |
| α_actor | 0.0001 |
| Batch size | 128 |
| γ | 0.3 |

---

## 7. 训练方式

- 按资产类别分组训练（4 个模型）
- 每 5 年重新训练
- 等权组合: `R_port = (1/N) Σ R_i`
