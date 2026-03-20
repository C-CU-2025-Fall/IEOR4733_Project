# 📊 论文复现差异分析报告

**分析时间**: 2026-03-20 15:57 EDT  
**对比对象**: 论文PDF vs 我们的代码实现

---

## 🔴 关键差异（Critical Gaps）

### 1. 状态空间（State Space）- ❌ 严重不匹配

#### 论文要求（第4页）
```
State Space:
- 过去60个观察值 (past 60 observations)
- 特征包括:
  1. 归一化收盘价序列 (Normalised close price series)
  2. 不同周期收益率:
     - 过去1个月 (21天)
     - 过去2个月 (42天)
     - 过去3个月 (63天)
     - 过去1年 (252天)
  3. MACD指标 (公式3)
     - Sk ∈ {8, 16, 32}
     - Lk ∈ {24, 48, 96}
  4. RSI指标 (30天回溯)
```

#### 我们的实现
```python
def _obs(self):
    obs = np.zeros(16, dtype=np.float32)
    if self.step_idx >= 20:
        obs[0] = np.mean(self.returns[self.step_idx-20:self.step_idx])  # 20天平均收益
        obs[1] = np.std(self.returns[self.step_idx-20:self.step_idx])   # 20天波动率
        obs[2] = self.returns[self.step_idx-1]                          # 昨日收益
    return obs
```

**差异**:
- ❌ 只用20天数据，论文要求60天
- ❌ 缺少归一化收盘价序列
- ❌ 缺少多周期收益率（1月/2月/3月/1年）
- ❌ **缺少MACD指标**
- ❌ **缺少RSI指标**
- ❌ 状态维度只有3个有效特征，论文要求多个特征×60时间步

**影响**: ⭐⭐⭐⭐⭐ **非常严重** - 这是核心方法论差异

---

### 2. 奖励函数（Reward Function）- ⚠️ 部分匹配

#### 论文要求（第4页，公式4）
```
R_t = λ * A_{t-1} * (σ_tgt / σ_{t-1}) * r_t 
    - bp * |p_{t-1} * (σ_tgt / σ_{t-1}) * A_{t-1} 
           - p_{t-2} * (σ_tgt / σ_{t-1}) * A_{t-2}|

其中:
- λ = 1 (固定值)
- σ_tgt = 波动率目标 (volatility target)
- σ_{t-1} = 事前波动率估计 (60天指数加权移动标准差)
- r_t = 收益率
- bp = 交易成本率 (0.0020 = 20bps)
```

**关键点**: 论文使用了**波动率缩放** (volatility scaling)

#### 我们的实现
```python
def step(self, action):
    action = float(np.clip(action, -1, 1))
    cost = abs(action - self.last_action) * 0.002  # 20bps
    ret = self.returns[self.step_idx + 1]
    reward = action * ret - cost
```

**差异**:
- ⚠️ **缺少波动率缩放** (volatility scaling)
- ⚠️ 交易成本计算简化（没有考虑价格项）
- ✅ 基础奖励结构正确 (action * return - cost)

**影响**: ⭐⭐⭐⭐ **严重** - 波动率缩放是论文的核心创新之一

---

### 3. 训练周期（Training Period）- ❌ 不匹配

#### 论文要求（第6页）
```
数据集: 2005-2019
训练策略: 每5年重新训练
- 2011年测试: 用2005-2010训练
- 2016年测试: 用2011-2015训练
测试期: 2011-2019
```

#### 我们的实现
```python
train = df[(df['Date'] >= '2011-01-03') & (df['Date'] <= '2015-12-31')]
```

**差异**:
- ❌ 训练期: 2011-2015 (论文要求2005-2010用于2011测试)
- ❌ 没有滚动重新训练机制
- ❌ 测试期: 2016-2019 (论文要求2011-2019)

**影响**: ⭐⭐⭐⭐ **严重** - 数据时间窗口不匹配

---

### 4. 数据集（Dataset）- ⚠️ 部分匹配

#### 论文要求（第6页）
```
- 50个期货合约
- 数据来源: Pinnacle Data Corp CLC Database
- 4个资产类别:
  - Commodity (商品)
  - Equity Index (股票指数)
  - Fixed Income (固定收益)
  - FX (外汇)
```

#### 我们的实现
```python
CONTRACTS_BY_CLASS = {
    'Commodity': ['CL=F', 'GC=F', ...],  # 13个合约
    'Equity Index': ['ES=F', 'NQ=F', 'YM=F'],  # 3个合约
    'Fixed Income': ['ZN=F', 'ZB=F', ...],  # 5个合约
    'FX': ['6E=F', '6J=F', ...]  # 9个合约
}
# 总计: 30个合约
```

**差异**:
- ⚠️ 合约数: 30 vs 50 (60%覆盖)
- ⚠️ 数据源: Yahoo Finance vs Pinnacle CLC

**影响**: ⭐⭐⭐ **中等** - 数据限制但方法论可对齐

---

## 🟡 中等差异（Moderate Gaps）

### 5. LSTM输入形状 - ⚠️ 不匹配

#### 论文要求（第6页）
```
Two-layer LSTM networks with 64 and 32 units
输入: (batch, 60时间步，特征数)
```

#### 我们的实现
```python
class LSTMNetwork(nn.Module):
    def __init__(self, input_dim, hidden_sizes=[64, 32], output_dim=3):
        self.lstm1 = nn.LSTM(input_dim, hidden_sizes[0], batch_first=True)
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        
# 实际使用:
state_t = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
# 形状: (1, 1, 16) - 只有1个时间步！
```

**差异**:
- ❌ 输入只有1个时间步，论文要求60个时间步
- ❌ 没有利用LSTM的序列建模能力

**影响**: ⭐⭐⭐⭐ **严重** - LSTM变成了普通前馈网络

---

### 6. 超参数（Hyperparameters）- ✅ 基本匹配

#### 论文 Table 1
| Parameter | DQN | 我们 |
|-----------|-----|------|
| α_critic | 0.0001 | 0.0001 ✅ |
| Batch size | 64 | - ❌ |
| γ | 0.3 | 0.3 ✅ |
| bp | 0.0020 | 0.002 ✅ |
| Memory size | 5000 | - ❌ |
| τ | 1000 | - ❌ |

**差异**:
- ✅ 学习率对齐
- ✅ 折扣因子对齐
- ✅ 交易成本对齐
- ❌ 缺少experience replay buffer
- ❌ 缺少target network更新机制

**影响**: ⭐⭐⭐ **中等** - 核心超参数对齐，但训练机制简化

---

## 🟢 匹配良好（Good Match）

### 7. 网络架构 - ✅ 匹配
- ✅ 两层LSTM [64, 32]
- ✅ Leaky-ReLU激活函数
- ✅ 按资产类别训练（4个模型）

### 8. 动作空间 - ✅ 匹配
- ✅ 离散动作空间 {-1, 0, 1}
- ✅ 对应 short/neutral/long

### 9. 评估指标 - ✅ 匹配
- ✅ Sharpe Ratio
- ✅ 10个完整指标实现 (calc_all_metrics.py)

---

## 📋 修复优先级

### 🔴 高优先级（必须修复）

1. **状态空间重构**
   - 添加60天时间窗口
   - 实现MACD指标
   - 实现RSI指标
   - 添加多周期收益率

2. **奖励函数修正**
   - 添加波动率缩放
   - 使用60天指数加权移动标准差
   - 修正交易成本计算

3. **LSTM输入形状**
   - 从(1, 16)改为(60, 特征数)
   - 利用序列建模能力

4. **训练周期调整**
   - 使用正确的训练/测试分割
   - 实现滚动重新训练机制

### 🟡 中优先级（建议修复）

5. **经验回放机制**
   - 添加replay buffer (5000)
   - 实现target network (τ=1000)

6. **数据扩展**
   - 尽可能获取更多合约数据

### 🟢 低优先级（可选）

7. **文档完善**
   - 记录所有差异
   - 说明简化原因

---

## 📊 对齐度评估

| 组件 | 对齐度 | 说明 |
|------|--------|------|
| 状态空间 | 20% | ❌ 缺少MACD/RSI/多周期 |
| 奖励函数 | 50% | ⚠️ 缺少波动率缩放 |
| 网络架构 | 90% | ✅ LSTM [64,32]正确 |
| 超参数 | 70% | ⚠️ 缺少replay buffer |
| 训练周期 | 30% | ❌ 时间窗口错误 |
| 数据集 | 60% | ⚠️ 30/50合约 |
| 评估指标 | 100% | ✅ 完整实现 |

**总体对齐度**: **~60%**

---

## 💡 建议

### 短期（1-2天）
1. 实现正确的状态空间（MACD+RSI+多周期）
2. 添加波动率缩放到奖励函数
3. 修正LSTM输入形状为(60, 特征数)

### 中期（3-5天）
4. 调整训练/测试时间窗口
5. 添加experience replay机制
6. 重新训练并对比结果

### 长期
7. 获取更多数据（接近50合约）
8. 实现完整的滚动训练机制

---

## 🎯 对结果的影响

### 当前结果
- Equity Index DQN: 0.972 (超越论文0.648)
- 其他类别不如论文

### 可能原因
- **简化的状态空间** 可能导致过拟合
- **缺少波动率缩放** 可能影响风险调整收益
- **训练期不同** 导致市场特征不同

### 修复后预期
- 结果可能更接近论文
- Equity Index可能不会超越论文（当前可能是过拟合）
- 其他类别可能改善

---

**分析完成时间**: 2026-03-20 15:57 EDT  
**建议**: 优先修复状态空间和奖励函数，然后重新训练对比
