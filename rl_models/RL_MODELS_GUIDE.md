# 深度强化学习模型实现指南

## 概述

实现了论文《Deep Reinforcement Learning for Trading》中的三个核心 RL 算法：
- **DQN** (Deep Q-Network): Fixed Q-targets + Double DQN
- **PG** (Policy Gradient): Monte Carlo 策略梯度
- **A2C** (Advantage Actor-Critic): 实时优势演员-评论家

## 文件结构

```
rl_models/
├── train_dqn_paper_aligned.py      # DQN 模型训练
├── train_pg_paper_aligned.py       # PG 模型训练
├── train_a2c_paper_aligned.py      # A2C 模型训练
├── train_all_rl_models.py          # 统一训练脚本
├── evaluate_rl_models.py           # 回测和评估
├── RL_MODELS_GUIDE.md              # 本文档
└── __init__.py                     # Python 包初始化文件
```

## 快速开始

### 方式 1: 训练所有模型（从项目根目录）

```bash
python rl_models/train_all_rl_models.py
```

输出：
- `models_dqn_paper_YYYYMMDD_HHMMSS.pkl` - DQN 模型
- `models_pg_paper_YYYYMMDD_HHMMSS.pkl` - PG 模型  
- `models_a2c_paper_YYYYMMDD_HHMMSS.pkl` - A2C 模型

### 方式 2: 训练单个模型

```bash
# 仅训练 DQN
python rl_models/train_all_rl_models.py dqn

# 仅训练 PG
python rl_models/train_all_rl_models.py pg

# 仅训练 A2C
python rl_models/train_all_rl_models.py a2c
```

### 方式 3: 回测已训练的模型

```bash
python rl_models/evaluate_rl_models.py
```

## 模型详细说明

### 1. DQN (Deep Q-Network)

**原理**：价值函数方法，学习状态-动作对的最优价值。

**关键特性**：
- ✅ **Fixed Q-targets**: 使用目标网络防止自举不稳定
- ✅ **Double DQN**: 主网络选动作，目标网络计算 Q 值，降低过估计
- ✅ **Experience Replay**: 5000 容量的回放缓冲区打破样本相关性
- ✅ **Gradient Clipping**: 梯度裁剪防止爆炸

**超参数** (论文 Table 1):
```python
LR = 0.0001              # 学习率
GAMMA = 0.3              # 折扣因子
BATCH_SIZE = 64          # 批量大小
MEMORY_SIZE = 5000       # 经验回放缓冲区大小
TAU = 1000               # 目标网络更新频率
BP = 0.0020              # 交易成本 (20 bps)
```

**网络架构**:
```
输入 (8 维特征)
    ↓
LSTM (64 单元)
    ↓
LSTM (32 单元)
    ↓
Dense (3 输出: 3 个动作的 Q 值)
```

**训练流程**:
1. 重放缓冲区采样小批量
2. 双 DQN 目标计算
3. MSE 损失优化
4. 每 1000 步更新目标网络

**时间复杂度**: O(batch_size × seq_len)

### 2. PG (Policy Gradient)

**原理**：直接学习策略，最大化期望累积奖励。

**关键特性**：
- ✅ **Monte Carlo 采样**: 基于完整回合的累积回报计算梯度
- ✅ **轨迹采集**: 收集完整回合后进行策略更新
- ✅ **回报标准化**: 降低方差，加速收敛

**超参数** (论文 Table 1):
```python
LR_ACTOR = 0.0001       # Actor 学习率
GAMMA = 0.3             # 折扣因子
BP = 0.0020             # 交易成本 (20 bps)
```

**网络架构** (Actor):
```
输入 (8 维特征)
    ↓
LSTM (64 单元)
    ↓
LSTM (32 单元)
    ↓
Dense (3 输出: 3 个动作的概率)
    ↓
Softmax
```

**训练流程**:
1. 从策略采样完整回合
2. 计算 Monte Carlo 回报: G_t = Σ γ^k R_{t+k}
3. 标准化回报降低方差
4. 计算 log 概率的加权和: J(θ) = E[log π(a|s) × G_t]
5. 策略梯度优化

**优势**:
- 适合连续动作空间
- 收敛更稳定
- 梯度无偏

**劣势**:
- 高方差（需要大量样本）
- 仅在回合结束时更新（慢）

### 3. A2C (Advantage Actor-Critic)

**原理**：结合价值函数（评论家）和策略函数（演员）的混合方法。

**关键特性**：
- ✅ **Advantage Function**: A(s,a) = R + γV(s') - V(s)
- ✅ **实时更新**: 每步更新（不等回合结束）
- ✅ **双网络架构**: 分离 Actor（策略）和 Critic（价值）
- ✅ **同步更新**: Actor 和 Critic 并行优化

**超参数** (论文 Table 1):
```python
LR_CRITIC = 0.001       # Critic 学习率
LR_ACTOR = 0.0001       # Actor 学习率
GAMMA = 0.3             # 折扣因子
BATCH_SIZE = 128        # 批量大小
BP = 0.0020             # 交易成本 (20 bps)
```

**网络架构**:
```
Actor Network (策略):        Critic Network (价值):
输入 (8 维特征)              输入 (8 维特征)
    ↓                           ↓
LSTM (64 单元)              LSTM (64 单元)
    ↓                           ↓
LSTM (32 单元)              LSTM (32 单元)
    ↓                           ↓
Dense (3 输出)              Dense (1 输出: V(s))
```

**训练流程**:
1. **Critic 更新**: 学习状态价值 V(s)
   - TD 目标: R + γV(s')
   - 损失: MSE(V(s), R + γV(s'))
2. **优势计算**: A(s,a) = R + γV(s') - V(s)
3. **Actor 更新**: 最大化 log π(a|s) × A(s,a)
   - 损失: -log π(a|s) × A(s,a)
4. 实时重复（每步或小批量）

**优势**:
- 低方差（通过 Critic 建立基线）
- 实时更新（快速收敛）
- 策略和价值分离（灵活）

**劣势**:
- 两个网络（参数多）
- 需要更小心的学习率调整

## 特征工程

所有模型使用 8 维特征向量：

1. **平均收益率** (mean return)
2. **波动率** (volatility)
3. **偏度** (skewness) - 收益分布的对称性
4. **峰度** (kurtosis) - 收益分布的尖峰性
5. **最大回撤** (max drawdown)
6. **夏普比** (Sharpe ratio) - 风险调整回报
7. **Sortino 比** (Sortino ratio) - 下行风险调整回报
8. **正收益率比例** (pct positive returns)

计算窗口：100 日回溯窗口

## 数据处理

### 训练期 (2011-01-01 to 2015-12-31)

- 4 个资产类别
- 36 个合约（部分缺失）
- 采样频率：日收益率
- 特征：标准化后的 8 维向量

### 测试期 (2016-01-01 to 2019-12-31)

- 用于模型评估（待实现）
- 计算与规则策略相同的 9 个指标

## 与规则策略的对比

| 策略 | 方法 | 更新频率 | 参数 |
|------|------|---------|------|
| Long | 简单持有 | - | σ = 0.058 |
| MACD | 动量指标 | 日 | 参数枚举 |
| Sign(R) | 历史回报符号 | 日 | 365 日窗口 |
| **DQN** | **Q 学习** | **每批** | **~4000** |
| **PG** | **策略梯度** | **每回合** | **~4000** |
| **A2C** | **演员-评论家** | **每步** | **~8000** |

## 当前实现状态

### ✅ 已完成

- [x] DQN 模型实现（Fixed Q-targets + Double DQN）
- [x] PG 模型实现（Monte Carlo）
- [x] A2C 模型实现（实时优势演员-评论家）
- [x] LSTM 特征提取
- [x] 统一训练脚本
- [x] 模型保存和加载机制
- [x] 训练历史记录
- [x] RL 模型文件夹组织

### ⏳ 待实现

- [ ] 集成回测系统与现有 baseline_run.py
- [ ] 测试期 (2016-2019) 推理和评估
- [ ] 计算 9 个指标并与规则策略对比
- [ ] Table 3 扩展版本（加入 DQN/PG/A2C）
- [ ] 超参数敏感性分析
- [ ] 转移学习（跨资产类别）
- [ ] 在线学习和模型更新

## 性能指标

### 计划计算的 9 个指标

对于每个资产类别和策略：

1. **E(R)** - 期望收益率 (年化)
2. **std(R)** - 收益标准差 (年化)
3. **Drawdown Duration** - 最大回撤周期
4. **Sharpe** - 夏普比
5. **Sortino** - Sortino 比
6. **MDD** - 最大回撤
7. **Calmar** - Calmar 比 (年化回报 / MDD)
8. **% +ve** - 正收益率百分比
9. **Ave P/L** - 平均单笔交易损益

## 故障排除

### 问题：模型训练缓慢

**解决方案**：
- 检查是否使用 GPU: `torch.cuda.is_available()`
- 减少 episodes 数量进行测试
- 增加 batch_size

### 问题：加载数据失败

**解决方案**：
- 确保数据文件在 `config/TEMP/` 或 `data/CLC/` 目录
- 文件格式应为 CSV (Date, Open, High, Low, Close, Volume)
- 检查日期范围 (2011-01-01 to 2015-12-31)

### 问题：模型保存失败

**解决方案**：
- 检查磁盘空间
- 确保有写入权限
- 模型文件较大 (~50MB per asset class)

### 问题：导入错误 ModuleNotFoundError

**解决方案**：
- 确保从项目根目录运行脚本：`python rl_models/train_*.py`
- 不要从 rl_models/ 目录直接运行

## 扩展建议

1. **多步学习**: 使用 n-step returns 替代 1-step TD
2. **优先经验回放**: 给高误差转移更高采样概率
3. **参数化噪声**: 在网络权重中加入噪声进行探索
4. **多资产联合学习**: 跨资产类别共享表示
5. **元学习**: 快速适应新资产

## 参考文献

原论文：Zhang, Z., Zohren, S., & Roberts, S. (2019)  
"Deep Reinforcement Learning for Trading"  
*NeurIPS 2019 Workshop on Applications of Machine Learning to Algorithmic Finance*

## 许可

遵循原项目许可证。

---

**最后更新**: 2024年4月  
**维护者**: AI Assistant  
**位置**: `rl_models/` 文件夹
