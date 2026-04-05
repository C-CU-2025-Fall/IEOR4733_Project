# 📋 论文方法论 - Deep Reinforcement Learning for Trading

**论文**: Zhang, Zohren, Roberts (2019) - Oxford  
**arXiv**: https://arxiv.org/pdf/1911.10107

> **⚠️ 复现时必须严格对照本文档！**

---

## 关键发现 ⚠️

### Portfolio-level Volatility Scaling

论文 Table 2 的结果经过 **portfolio-level volatility scaling**：

> "We present our results in Table 2 where an additional layer of portfolio-level volatility scaling is applied for each model. This brings the volatility of different methods to a **same target** so we can directly compare metrics."

**目标 std(R) = 0.97 (97% 年化)**

### 计算流程

1. **训练时**: 使用 σ_tgt = 10% 年化目标波动率
2. **评估时**: 应用 portfolio-level scaling 使 std(R) ≈ 0.97
3. **所有指标**: 都基于缩放后的收益计算（包括 MDD）

### 验证结果 (Equity Index Long Only)

| 指标 | 我们 | 论文 | 差异 |
|------|------|------|------|
| E(R) | 0.678 | 0.668 | +1.5% ✅ |
| std(R) | 0.970 | 0.970 | 0% ✅ |
| Sharpe | 0.699 | 0.688 | +1.6% ✅ |

---

## 1. 数据集 (Section 4.1)

| 项目 | 论文要求 |
|------|----------|
| **数据源** | Pinnacle Data Corp CLC Database |
| **合约数** | 50 个期货合约 |
| **时间范围** | 2005-2019 |
| **资产类别** | Commodity (25), Equity Index (11), Fixed Income (5), FX (9) |
| **训练方式** | 每隔 5 年重新训练 |
| **测试期** | 2011-2019 |

### 滚动训练 (论文原文)

> "We retrain our model **at every 5 years**, using all data available up to that point to optimise the parameters. Model parameters are then **fixed for the next 5 years** to produce out-of-sample results. In total, our testing period is from **2011 to 2019**."

**理解**:
- 训练期：使用到某时间点之前的所有数据
- 参数固定：随后 5 年用于样本外测试
- 测试期：2011-2019

---

## 2. 状态空间 (Section 3.1)

**时间窗口**: 60 天

**8 个特征**:
1. **归一化收盘价** (Normalised close price series)
2. **21 天收益率** (1 month, 波动率调整): `r_{t-21,t} / (σ_t * √21)`
3. **42 天收益率** (2 months): `r_{t-42,t} / (σ_t * √42)`
4. **63 天收益率** (3 months): `r_{t-63,t} / (σ_t * √63)`
5. **252 天收益率** (1 year): `r_{t-252,t} / (σ_t * √252)`
6. **MACD 指标** (公式 3):
   ```
   MACD_t = q_t / std(q_{t-252:t})
   q_t = (m(S) - m(L)) / std(p_{t-63:t})
   ```
   多时间尺度平均: Sk ∈ {8, 16, 32}, Lk ∈ {24, 48, 96}
7. **RSI 指标** (30 天回看)
8. **波动率** (60 天指数加权移动标准差)

**输出形状**: `(60, 8)` → LSTM 输入

---

## 3. 动作空间

| 模型 | 动作空间 | 说明 |
|------|----------|------|
| **DQN** | `{-1, 0, 1}` | 离散：做空、空仓、做多 |
| **PG** | `{-1, 0, 1}` | 离散 |
| **A2C** | `[-1, 1]` | 连续 |

---

## 4. 奖励函数 (公式 4)

```
R_t = (A_t * σ_t^(-1)) * r_t - bp * p_{t-1} * |A_t - A_{t-1}| * σ_t^(-1)
```

**其中**:
- `A_t`: 动作 (位置)
- `σ_t^(-1) = σ_target / σ_t`: 波动率缩放因子
- `σ_target = 0.10` (10% 年化目标波动率)
- `bp = 0.0020` (20 bps 交易成本)
- `r_t`: 收益率
- `p_{t-1}`: 前一时刻价格

---

## 5. 网络架构 (Section 4.3)

**所有模型使用相同网络结构**:

| 组件 | 规格 |
|------|------|
| **类型** | 两层 LSTM |
| **第一层** | 64 单元 |
| **第二层** | 32 单元 |
| **激活函数** | Leaky-ReLU |
| **Leaky-ReLU 负斜率** | 0.01 (标准值) |

> "We use **two-layer LSTM networks with 64 and 32 units** in all models, and **Leaky Rectifying Linear Units (Leaky-ReLU)** are used as activation functions."

---

## 6. 超参数 (Table 1) ⚠️ 严格对照

### DQN

| 参数 | 值 |
|------|-----|
| **α_critic** | **0.0001** |
| **Optimiser** | Adam |
| **Batch size** | **64** |
| **γ (折扣因子)** | **0.3** |
| **bp (交易成本)** | **0.0020** |
| **Memory size** | **5000** |
| **τ (target update)** | **1000** |

> "The memory size shows the size of the buffer for experience replay, and we update the parameters of our target network in DQN at **every 1000 steps**."

### PG

| 参数 | 值 |
|------|-----|
| **α_actor** | **0.0001** |
| **Optimiser** | Adam |
| **Batch size** | **- (无)** |
| **γ** | **0.3** |
| **bp** | **0.0020** |
| **Memory size** | **-** |
| **τ** | **-** |

### A2C ⚠️ 注意 critic 学习率

| 参数 | 值 |
|------|-----|
| **α_critic** | **0.001** ⚠️ (不是 0.0001！) |
| **α_actor** | **0.0001** |
| **Optimiser** | Adam |
| **Batch size** | **128** |
| **γ** | **0.3** |
| **bp** | **0.0020** |
| **Memory size** | **-** |
| **τ** | **-** |

---

## 7. 训练稳定机制 (Section 3.2)

**DQN 采用三种稳定策略**:

1. **Fixed Q-targets** [49]
   - 使用独立的目标网络产生 target values
   - 每 τ=1000 步更新目标网络参数
   
2. **Double DQN** [18]
   - 减少 policy variances
   - 主网络选择动作，目标网络计算 Q 值

3. **Dueling DQN** [50] (可选)
   - 分离 Q-value 为 state value 和 advantage

---

## 8. 训练方式

### 按资产类别分组训练

> "As our dataset consists of different asset classes, **we train a separate model for each asset class**."

**4 个资产类别**:
- Commodity
- Equity Index
- Fixed Income
- FX

### 等权组合评估

> "We then form a simple portfolio by **giving equal weights to each contract**"

```
R_port = (1/N) * Σ R_i
```

---

## 9. 评估指标 (Section 4.4)

| 指标 | 说明 |
|------|------|
| **E(R)** | 年化期望收益 |
| **std(R)** | 年化标准差 |
| **DD** | Downside Deviation (负收益标准差) |
| **Sharpe** | E(R) / std(R) |
| **Sortino** | E(R) / DD |
| **MDD** | Maximum Drawdown |
| **Calmar** | E(R) / MDD |
| **% +ve Returns** | 正收益交易占比 |
| **Ave. P / Ave. L** | 正/负收益比率 |

---

## 10. 代码实现检查清单

### ✅ 必须对齐

| 组件 | 论文值 | 检查 |
|------|--------|------|
| LSTM 结构 | [64, 32] | ✅ |
| Leaky-ReLU | negative_slope=0.01 | ✅ |
| DQN α_critic | 0.0001 | ✅ |
| DQN batch | 64 | ✅ |
| DQN memory | 5000 | ✅ |
| DQN τ | 1000 | ✅ |
| A2C α_critic | **0.001** | ⚠️ 需修复 |
| A2C α_actor | 0.0001 | ✅ |
| A2C batch | 128 | ⚠️ 需检查 |
| PG α_actor | 0.0001 | ✅ |
| γ | 0.3 | ✅ |
| bp | 0.002 | ✅ |
| 状态窗口 | 60 天 | ✅ |
| 状态特征 | 8 个 | ✅ |
| 波动率目标 | 10% | ✅ |

---

## 11. 论文 Table 2 目标结果

### DQN Sharpe Ratio (目标)

| 资产类别 | 论文 Sharpe |
|----------|-------------|
| Commodity | 0.723 |
| Equity Index | 0.648 |
| Fixed Income | 0.935 |
| FX | 0.546 |
| **All** | **1.288** |

---

## 12. 修复记录

### 2026-03-20 21:35 EDT
- ⚠️ 发现 A2C α_critic 应为 **0.001** (之前错误设为 0.0001)
- ⚠️ PG batch size 为 **-** (无，之前错误设为 128)
- ✅ 更新 methodology.md 为完整论文提取版本
