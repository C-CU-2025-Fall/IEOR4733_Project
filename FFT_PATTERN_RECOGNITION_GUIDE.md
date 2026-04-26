# FFT Pattern Recognition 详解
## graph_model_fall2025 中的 fft_pattern_model_soft_regime_weights 工作原理

---

## 📊 整体架构

```
训练期日回报数据 (T_train, N_assets)
         ↓
    [滑动窗口 FFT 特征提取]
         ↓
平均回报序列的FFT特征 (T_train - window_size, 20D)
         ↓
    [GMM 聚类]
         ↓
Regime标签 + 软概率 (3个regime)
         ↓
[计算每个regime的权重]
         ↓
状态权重字典: {regime_id: weight_vector}
         ↓
[训练期回报] → 硬标签权重
[测试期回报] → 软加权
```

---

## 🔧 第一步：滑动窗口 FFT 特征提取

### 核心思想
不是对单个资产的时间序列做FFT，而是对**所有资产平均回报**做FFT。

### 实现细节

```python
# 输入：日回报矩阵
R_train.shape = (T_train, N_assets)  # e.g., (1000, 50)

# 参数
WINDOW_SIZE = 20      # 滑动窗口 20 天
FFT_KEEP = 10         # 保留 10 个频率分量

# 步骤 1: 对每个时间点t，取过去20天的窗口
for i in range(WINDOW_SIZE, num_days_train):
    # 取 [i-20:i] 的20天数据
    chunk = R_train[i - WINDOW_SIZE:i, :]  # shape: (20, 50)
    
    # 步骤 2: 计算该窗口内所有资产的平均回报
    avg_series = chunk.mean(axis=1)  # shape: (20,)
    # avg_series[t] = 第t天所有资产的平均回报
    
    # 步骤 3: 对这个平均序列做 FFT
    fft_vals = np.fft.fft(avg_series)  # 长度为20
    
    # 步骤 4: 提取前10个频率分量的实部和虚部
    feature = np.concatenate([
        np.real(fft_vals[:FFT_KEEP]),    # 实部 (10,)
        np.imag(fft_vals[:FFT_KEEP])     # 虚部 (10,)
    ])  # 特征向量维度 = 20
    
    fft_features.append(feature)
    train_indices.append(i)
```

### 为什么这样做？

| 方面 | 原始方法 | FFT方法 |
|------|---------|--------|
| **输入** | 单个资产的时间序列 | 所有资产的平均序列 |
| **特征维度** | 180D (9列×20) | 20D (10实+10虚) |
| **计算复杂度** | O(N×20) | O(N) |
| **含义** | 单个资产的频域特征 | 市场整体模式 |
| **样本数** | 50个 (每个合约1个) | ~100k个 (每个时间点1个) |

---

## 🎯 第二步：GMM 聚类

```python
# 输入：FFT特征矩阵
fft_features.shape = (n_windows, 20)  # e.g., (980, 20)

# 使用 Gaussian Mixture Model 进行聚类
gmm = GaussianMixture(n_components=3, random_state=42)

# 获得硬聚类标签和软概率
regime_labels = gmm.fit_predict(fft_features)          # shape: (980,)
soft_probs = gmm.predict_proba(fft_features)           # shape: (980, 3)

# 结果
regime_labels[i] ∈ {0, 1, 2}              # 窗口i属于哪个regime
soft_probs[i] = [p0, p1, p2]              # 属于各regime的概率
# p0 + p1 + p2 = 1.0
```

### 为什么用 GMM？

- **K-Means** (硬聚类): 每个样本只属于一个簇
- **GMM** (软聚类): 每个样本有概率属于各个簇
- **优势**: 在市场regime转换期间，能捕捉**过渡态**
  - 例如：从低波动到高波动的过程中，某个窗口可能有 60% 的概率属于低波动regime

---

## ⚖️ 第三步：计算状态特定的资产权重

### 核心理念
每个regime表现不同，所以每个regime应该有**不同的权重**。

```python
state_weights = {}

for cluster in range(3):  # 对每个regime
    # 步骤 1: 找出所有属于该regime的窗口
    cluster_mask = regime_labels == cluster
    cluster_windows = np.where(cluster_mask)[0]  # e.g., [5, 12, 18, 24, ...]
    
    # 步骤 2: 对每个窗口，收集其对应的**回报数据**
    rows_in_cluster = []
    for window_idx in cluster_windows:
        t_end = train_indices[window_idx]  # 窗口结束时间
        t_start = t_end - WINDOW_SIZE      # 窗口开始时间
        
        # 该窗口对应的日回报数据
        window_returns = R_train[t_start:t_end]  # shape: (20, 50)
        rows_in_cluster.append(window_returns)
    
    # 步骤 3: 合并所有属于该regime的回报数据
    stacked = np.vstack(rows_in_cluster)  # shape: (n_days_in_regime, 50)
    
    # 步骤 4: 计算每个资产在该regime下的波动率
    vol = np.std(stacked, axis=0)  # shape: (50,)
    
    # 步骤 5: Inverse Volatility 权重
    #         低波动资产 → 高权重
    #         高波动资产 → 低权重
    weights = 1.0 / vol
    weights = weights / weights.sum()  # 归一化到1
    
    state_weights[cluster] = weights
    
    # 示例
    # cluster 0 (低波动regime): [0.03, 0.02, 0.05, ...]  # 倾向风险资产
    # cluster 1 (高波动regime): [0.01, 0.01, 0.01, ...]  # 更均衡
    # cluster 2 (过渡regime):    [0.02, 0.03, 0.02, ...]
```

### 逻辑图

```
属于 Regime 0 的窗口: t=5,12,18,24,...
    ↓
收集这些时间的回报数据
    ↓
计算波动率：
  - 资产A: 低波动 → 权重高 (0.05)
  - 资产B: 高波动 → 权重低 (0.01)
    ↓
在Regime 0时期，倾向配置到低波动资产
```

---

## 📈 第四步：计算训练期 Portfolio 回报（硬标签）

```python
# 训练期：使用硬regime标签
model_train_return = np.zeros(num_days_train)

for window_idx, t in enumerate(train_indices):
    # 这个窗口属于哪个regime
    cluster = regime_labels[window_idx]
    
    # 获取该regime的权重
    w = state_weights[cluster]
    
    # 计算该时间点的portfolio回报
    # portfolio_return[t] = sum(daily_return[t] * weight)
    model_train_return[t] = R_train[t] @ w
```

### 示例时间轴

```
时间t   | 回报向量R_train[t] | Regime | 权重向量w | Portfolio回报
--------|------------------|--------|---------|---------------
t=20    | [0.01, -0.02, ...] | 0 (低波) | [0.05, 0.01, ...] | 0.002
t=21    | [0.02,  0.01, ...] | 0 (低波) | [0.05, 0.01, ...] | 0.003
t=22    | [-0.01, 0.03, ...] | 1 (高波) | [0.01, 0.03, ...] | 0.001
t=23    | [0.00, -0.01, ...] | 2 (过渡) | [0.02, 0.03, ...] | -0.0005
```

---

## 🎭 第五步：计算测试期 Portfolio 回报（软加权）

```python
# 测试期：使用软概率加权

# 步骤 1: 为测试期提取FFT特征
fft_features_test = []
for i in range(WINDOW_SIZE, num_days_test):
    chunk = R_test[i - WINDOW_SIZE:i, :]
    avg_series = chunk.mean(axis=1)
    fft_vals = np.fft.fft(avg_series)
    feature = np.concatenate([
        np.real(fft_vals[:FFT_KEEP]),
        np.imag(fft_vals[:FFT_KEEP])
    ])
    fft_features_test.append(feature)

# 步骤 2: 获取软概率（使用训练期拟合的GMM）
soft_probs_test = gmm.predict_proba(fft_features_test)
# shape: (n_test_windows, 3)

# 步骤 3: 对每个测试时间点，用软概率加权
model_test_return = np.zeros(num_days_test)

for window_idx, t in enumerate(test_indices):
    # 该窗口对各regime的概率
    probs = soft_probs_test[window_idx]  # [p0, p1, p2]
    
    # 用概率加权各regime的权重向量
    blended_weights = np.zeros(N_assets)
    for cluster in range(3):
        blended_weights += probs[cluster] * state_weights[cluster]
    
    # 计算portfolio回报
    model_test_return[t] = R_test[t] @ blended_weights
```

### 软加权的直观理解

```
测试时间t的FFT特征 → GMM 评估
    ↓
    "这个窗口有 70% 概率是低波动, 30% 概率是高波动"
    ↓
权重 = 0.7 × state_weights[0] + 0.3 × state_weights[1]
    ↓
portfolio_return[t] = R_test[t] @ 权重
```

这样做的好处：
- **平滑过渡**: 不会在regime转换时突然改变权重
- **风险平衡**: 在不确定期间，自动向多个regime分散
- **适应性**: 自动跟踪市场regime的变化

---

## 🔑 关键概念总结

### 硬标签 vs 软概率

```
训练期：
  时间t属于Regime 0 → 使用 w[0]
  时间t属于Regime 1 → 使用 w[1]

测试期：
  时间t有70%概率属于Regime 0, 30%属于Regime 1
  → 使用 0.7×w[0] + 0.3×w[1]
```

### 为什么区分训练/测试？

1. **训练期** (in-sample):
   - 硬标签给出清晰的regime分配
   - 基于历史数据计算的权重
   - 用于回测验证

2. **测试期** (out-of-sample):
   - 软概率处理不确定性
   - 预测性评估
   - 更接近实际应用场景

---

## 📊 参数配置的影响

### window_size = 20

- 更长窗口(50天) → 捕捉长期趋势，但响应慢
- 更短窗口(5天)  → 捕捉短期波动，但噪音多

### n_clusters = 3

| n_clusters | 场景 |
|------------|------|
| 2 | "高波" vs "低波" |
| 3 | "低波 + 高波 + 过渡" (推荐) |
| 4+ | 过度分割，可能过拟合 |

### fft_keep = 10

- 保留10个频率 → 捕捉主要周期性
- 丢弃剩余10个频率 → 去噪

---

## 💡 实际应用流程

```
Day 0-19: 累积数据 (< window_size)
          ↓ (无法提取特征)

Day 20:   窗口[0:20] 
          → FFT特征1
          → GMM评估 → Regime 0
          → 使用权重w[0]
          → portfolio_return[20]

Day 21:   窗口[1:21]
          → FFT特征2
          → GMM评估 → Regime 1
          → 使用权重w[1]
          → portfolio_return[21]

... (持续到数据结束)
```

---

## 🎓 与你的原始方法的对比

### 你的方法：60×9状态矩阵

```
优点：
  ✓ 包含丰富的特征定义(价格、回报、MACD、RSI)
  ✓ 接近学术论文的标准

缺点：
  ✗ 特征维度太高 (180D)
  ✗ 计算复杂度大 O(N×20×9)
  ✗ 样本数太少 (50个合约)
  ✗ 时间对齐复杂
  ✗ 容易超时
```

### period_adaptative 的方法：平均回报FFT

```
优点：
  ✓ 特征维度低 (20D)
  ✓ 计算高效 O(N)
  ✓ 样本数多 (~100k)
  ✓ 时间对齐简单
  ✓ 运行稳定

缺点：
  ✗ 丢失单个资产的细节信息
  ✗ 只捕捉市场整体模式
```

---

## 🚀 下一步建议

1. **理解为什么**
   - FFT捕捉什么？→ 周期性波动模式
   - GMM为什么好？→ 捕捉market states的概率分布
   - 为什么测试用软概率？→ 处理regime转换的不确定性

2. **参数调优**
   - 尝试不同window_size: [10, 20, 40]
   - 尝试不同n_clusters: [2, 3, 4, 5]
   - 评估Sharpe, Sortino等指标

3. **混合方法**（可选）
   - 结合你的状态矩阵特征 + period_adaptative的效率
   - 例如：用5个最重要的特征代替60×9矩阵
