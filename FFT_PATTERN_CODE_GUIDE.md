# FFT Pattern Recognition 核心代码对比

## 第1步：特征提取

### ❌ 你的方法（60×9状态矩阵 → 180D特征）
```python
# 构建状态矩阵
state_matrix = build_state_matrix(prices[i:i+312])  # 60×9
# 计算180D特征
fft_feat = extract_fft_features_from_state(state_matrix)  # 180D

# 问题：
# - 每个合约只有1个状态矩阵
# - 样本数 = 50合约
# - 特征维度 = 180
# - 计算量 ∝ O(50 × 180)
```

### ✅ Period_Adaptive 方法（平均回报 → 20D特征）
```python
# 步骤1：取窗口内所有资产的平均回报
chunk = R_train[i-20:i, :]              # shape: (20, 50)
avg_series = chunk.mean(axis=1)         # shape: (20,)

# 步骤2：对平均序列做FFT
fft_vals = np.fft.fft(avg_series)       # length: 20

# 步骤3：提取实虚部
feature = np.concatenate([
    np.real(fft_vals[:10]),              # 10个实部
    np.imag(fft_vals[:10])               # 10个虚部
])                                       # shape: (20,)

# 优势：
# - 每个时间点都是一个样本
# - 样本数 ≈ 50 × 2000 = 100k
# - 特征维度 = 20（9倍压缩！）
# - 计算量 ∝ O(100k × 20)
```

---

## 第2步：GMM聚类

### 两个方法都一样
```python
from sklearn.mixture import GaussianMixture

# 拟合GMM
gmm = GaussianMixture(n_components=3, random_state=42)

# 训练期：获得硬标签和软概率
regime_labels_train = gmm.fit_predict(fft_features_train)  # (980,)
soft_probs_train = gmm.predict_proba(fft_features_train)   # (980, 3)

# 测试期：使用训练的GMM预测（注意：predict，不fit）
soft_probs_test = gmm.predict_proba(fft_features_test)     # (700, 3)
regime_labels_test = np.argmax(soft_probs_test, axis=1)    # (700,)
```

---

## 第3步：计算状态权重

```python
# 对每个regime计算专有的资产权重
state_weights = {}

for cluster in range(3):
    print(f"\n=== 处理 Regime {cluster} ===")
    
    # 找出属于该regime的所有窗口
    cluster_mask = regime_labels_train == cluster
    cluster_indices = np.where(cluster_mask)[0]
    
    print(f"这个regime有 {len(cluster_indices)} 个窗口")
    
    if len(cluster_indices) == 0:
        print(f"  ⚠️ 没有样本，使用等权重")
        state_weights[cluster] = np.ones(N_assets) / N_assets
        continue
    
    # 收集所有属于该regime的回报数据
    rows = []
    for window_idx in cluster_indices:
        t_end = train_indices[window_idx]
        t_start = t_end - WINDOW_SIZE
        
        # 该窗口对应的日回报
        window_data = R_train[t_start:t_end, :]
        rows.append(window_data)
    
    # 合并
    stacked = np.vstack(rows)  # shape: (n_days_in_cluster, 50)
    print(f"  合并后: {stacked.shape} (天数 × 资产数)")
    
    # 计算每个资产的波动率
    vol = np.std(stacked, axis=0)  # shape: (50,)
    print(f"  波动率范围: [{vol.min():.4f}, {vol.max():.4f}]")
    
    # Inverse volatility 权重
    weights = 1.0 / (vol + 1e-6)  # 避免除以0
    weights = weights / weights.sum()  # 归一化
    
    state_weights[cluster] = weights
    print(f"  权重范围: [{weights.min():.4f}, {weights.max():.4f}]")

# 结果
# state_weights[0] = [0.05, 0.02, 0.03, ...] (长度50)
# state_weights[1] = [0.01, 0.04, 0.02, ...] (长度50)
# state_weights[2] = [0.02, 0.03, 0.04, ...] (长度50)
```

---

## 第4步：计算训练期回报（硬标签）

```python
model_train_return = np.zeros(num_days_train)

for window_idx, t in enumerate(train_indices):
    # 该窗口属于哪个regime
    regime = regime_labels_train[window_idx]
    
    # 获取该regime的权重
    w = state_weights[regime]
    
    # 计算该日的portfolio回报
    daily_ret = R_train[t]  # shape: (50,)
    port_ret = np.dot(daily_ret, w)  # scalar
    
    model_train_return[t] = port_ret

# 示例：
# model_train_return[20] = 0.0025   (2.5bp)
# model_train_return[21] = -0.0010  (-10bp)
# model_train_return[22] = 0.0008   (8bp)
# ...
```

---

## 第5步：计算测试期回报（软加权）

### 核心不同：使用软概率而不是硬标签

```python
model_test_return = np.zeros(num_days_test)

for window_idx, t in enumerate(test_indices):
    # 该窗口对各regime的软概率
    probs = soft_probs_test[window_idx]  # shape: (3,)
    # probs = [0.7, 0.2, 0.1]  例如
    
    # 加权平均各regime的权重向量
    # blended_w = 0.7*w[0] + 0.2*w[1] + 0.1*w[2]
    blended_w = np.zeros(N_assets)
    for regime_id in range(3):
        blended_w += probs[regime_id] * state_weights[regime_id]
    
    # 计算该日的portfolio回报（使用加权权重）
    daily_ret = R_test[t]
    port_ret = np.dot(daily_ret, blended_w)
    
    model_test_return[t] = port_ret
```

### 为什么软加权？

```
假设某天的回报: [+1%, -2%, +0.5%, ...]

硬标签方法（有风险）：
  Regime分类为0 → 使用 w[0]
  → 可能权重配置为"激进" 
  → 如果判错了，损失很大！

软加权方法（更安全）：
  Regime有60%概率属于0, 40%属于1
  → 使用 0.6*w[0] + 0.4*w[1]
  → 自动在两个权重间折中
  → 即使判错，损失也有限
```

---

## 第6步：返回最终结果

```python
return (
    train_weights,              # (num_assets,) - 最后一个训练窗口的权重
    model_train_return,         # (num_days_train,) - 日回报序列
    model_test_return,          # (num_days_test,)  - 日回报序列
    regime_labels_train,        # (num_windows_train,) - 硬标签
    regime_labels_test,         # (num_windows_test,)  - 硬标签
    soft_probs_test             # (num_windows_test, 3) - 软概率
)
```

---

## 参数敏感性分析

### window_size 的影响

```
window_size = 5:
  → 频率分辨率低，捕捉长周期差
  → 特征噪音多
  → 样本数多

window_size = 20:  (推荐)
  → 平衡噪音和信息
  → 捕捉2-20天的周期

window_size = 60:
  → 频率分辨率高，捕捉短周期差
  → 特征更平滑
  → 样本数少，过拟合风险
```

### n_clusters 的影响

```
n_clusters = 2:
  "高波" vs "低波"
  过于简化

n_clusters = 3:  (推荐)
  "低波 vs 过渡 vs 高波"
  充分覆盖market states

n_clusters = 5:
  过度分割
  训练数据可能不足
  样本数少的regime权重不稳定
```

---

## 性能对比表

| 指标 | 你的方法 | Period_Adaptive |
|------|---------|-----------------|
| 特征维度 | 180 | 20 |
| 样本数 | 50 | ~100k |
| 构建时间 | 慢 (状态矩阵复杂) | 快 (简单平均) |
| 内存占用 | ~40MB | ~200MB (更多样本) |
| 时间对齐复杂度 | 高 (60天窗口) | 低 (直接索引) |
| 可解释性 | 高 (纯论文定义) | 中 (市场平均) |
| 运行稳定性 | 低 (容易超时) | 高 (高效) |
| 聚类质量 | 中 (样本少) | 高 (样本多) |

---

## 混合方法（可选）

如果你想保留状态矩阵的细节 + 获得高效性：

```python
# 改进方案：使用更简洁的特征定义

# 代替60×9矩阵，只用5个关键特征：
state_vector = np.array([
    normalized_close_price,      # 1D
    return_1m / vol,             # 1D
    return_1y / vol,             # 1D  
    macd_8_24,                   # 1D
    rsi_30                       # 1D
])  # 总共 5D

# 然后对这个5D向量做FFT
# fft_vals = np.fft.fft(lookback_period_of_5d_vectors)
# features = concat(real, imag)  # 10D特征

# 优势：
# ✓ 保留论文的特征定义
# ✓ 获得计算效率
# ✓ 样本数仍然很多
```

---

## 总结

### 核心思想
```
市场在不同regime下的特征不同
  ↓
用FFT捕捉这些特征的频域表现
  ↓
用GMM聚类识别market states
  ↓
每个state配置不同的权重
  ↓
自动切换投资策略
```

### 三个关键参数
1. **window_size**: 捕捉的时间周期长度
2. **n_clusters**: market states的数量
3. **fft_keep**: 保留的频率分量数

### 两个关键转变
1. **特征定义**: 从单资产 → 市场平均
2. **标签方式**: 从硬聚类 → 软概率

---

## 下一步：实现步骤

1. **加载RAD数据** ✓ (你已做)
2. **计算日回报** → 简单! return = log(price[t]) - log(price[t-1])
3. **滑动窗口FFT** → 20行代码
4. **GMM聚类** → sklearn搞定
5. **权重计算** → inverse volatility
6. **投资组合回报** → 矩阵乘法
7. **结果分析** → Sharpe, Sortino, Calmar
