# 🛡️ 安全并行训练指南

**重要**: 先测试内存，再决定并行数！

---

## 📋 完整流程

### 步骤 1: 单类别内存测试 (5 分钟) ⭐⭐⭐

```bash
python3 test_memory_single.py
```

**输出示例**:
```
📊 系统内存信息:
  总内存：16.0 GB
  可用：8.5 GB (53%)
  已用：7.5 GB (47%)

📈 进程基线内存：512.3 MB

📊 测试 Equity Index
======================================================================
  加载前内存：512.3 MB
  数据加载后：612.5 MB (+100.2 MB)
  环境创建后：625.8 MB (+13.3 MB)
  模型创建后：756.2 MB (+130.4 MB)
  训练后内存：876.5 MB
  清理后内存：645.3 MB

💡 并行训练建议
======================================================================
  系统可用内存：8.5 GB
  单类别峰值：364.2 MB
  单类别平均：342.1 MB

  ✅ 安全并行数：4 进程
  预计总内存：1456.8 MB (1.4 GB)
  内存余量：6731.2 MB (6.6 GB)

  🎉 可以安全并行训练所有 4 个类别！
```

---

### 步骤 2: 根据建议并行训练

#### 情况 A: 内存充足 (>8GB 可用)

```bash
# 4 进程并行
python3 train_parallel_safe.py --parallel 4 --episodes 200
```

**预计**: 80 分钟，峰值内存 ~1.5GB

---

#### 情况 B: 内存中等 (4-8GB 可用)

```bash
# 2 进程并行，分两批
python3 train_parallel_safe.py --parallel 2 --batch-size 2 --episodes 200
```

**预计**: 160 分钟，峰值内存 ~1.0GB

---

#### 情况 C: 内存紧张 (<4GB 可用)

```bash
# 串行训练
python3 train_parallel_safe.py --parallel 1 --episodes 200
```

**预计**: 320 分钟，峰值内存 ~0.5GB

---

## 📊 内存需求估算表

| 系统可用内存 | 推荐并行 | 批次大小 | 预计峰值 | 预计时间 |
|--------------|----------|----------|----------|----------|
| **>12 GB** | 4 | 4 | ~2 GB | 80 分钟 |
| **8-12 GB** | 4 | 4 | ~1.5 GB | 80 分钟 |
| **4-8 GB** | 2 | 2 | ~1 GB | 160 分钟 |
| **2-4 GB** | 1 | 1 | ~0.5 GB | 320 分钟 |
| **<2 GB** | ❌ | - | - | 不建议训练 |

---

## 🔍 单类别内存组成

```
基线 (Python + PyTorch): ~500 MB
数据加载：+100 MB
环境创建：+15 MB
LSTM 模型：+130 MB
Replay Buffer: +50 MB
训练临时：+80 MB
───────────────────────
峰值增量：~360 MB/类别
```

---

## 🛡️ 安全机制

### 1. 分批训练

```python
# 每批后清理内存
for i in range(0, 4, batch_size):
    batch = classes[i:i+batch_size]
    train_batch(batch)
    gc.collect()  # 垃圾回收
    torch.cuda.empty_cache()  # GPU 缓存清理
    time.sleep(10)  # 等待内存稳定
```

### 2. 实时监控

```python
class MemoryMonitor:
    - 基线内存
    - 峰值内存
    - 实时采样
```

### 3. 自动判断

```python
# 不指定 --parallel 时自动判断
if available_gb > 8:
    parallel = 4
elif available_gb > 4:
    parallel = 2
else:
    parallel = 1
```

---

## 🎯 推荐配置

### 最佳配置 (16GB+ 内存)

```bash
# 1. 内存测试
python3 test_memory_single.py

# 2. 4 进程并行
python3 train_parallel_safe.py --parallel 4 --episodes 200
```

**结果**: 80 分钟完成，峰值内存 ~1.5GB

---

### 标准配置 (8GB 内存)

```bash
# 1. 内存测试
python3 test_memory_single.py

# 2. 2 进程并行，分两批
python3 train_parallel_safe.py --parallel 2 --batch-size 2 --episodes 200
```

**结果**: 160 分钟完成，峰值内存 ~1GB

---

### 低配 (4GB 内存)

```bash
# 1. 内存测试
python3 test_memory_single.py

# 2. 串行训练
python3 train_dqn_paper_aligned.py
```

**结果**: 320 分钟完成，峰值内存 ~0.5GB

---

## ⚠️ 警告信号

训练时监控内存，如果出现以下情况立即停止：

```bash
# 使用 htop 或 top 监控
htop
# 或
watch -n 1 "free -h"
```

**警告信号**:
- ⚠️ 内存使用 >90%
- ⚠️ Swap 使用增加
- ⚠️ 系统变慢

**应对**:
```bash
# 立即停止训练
Ctrl+C

# 清理内存
python3 -c "import torch; torch.cuda.empty_cache()"

# 重启训练 (减小并行数)
python3 train_parallel_safe.py --parallel 1
```

---

## 📁 文件清单

| 文件 | 用途 | 时间 |
|------|------|------|
| `test_memory_single.py` | 单类别内存测试 | 5 分钟 |
| `train_parallel_safe.py` | 安全并行训练 | 80-320 分钟 |
| `train_dqn_paper_aligned.py` | 串行训练 (备用) | 320 分钟 |

---

## 💡 优化技巧

### 1. 关闭其他应用

训练前关闭浏览器、IDE 等占用内存的应用

### 2. 减小 Replay Buffer

```python
MEMORY_SIZE = 2000  # 从 5000 减小到 2000
# 节省 ~60% 内存
```

### 3. 减小 Batch Size

```python
BATCH_SIZE = 32  # 从 64 减小到 32
# 节省 ~50% 训练内存
```

### 4. 定期 GC

```python
import gc
gc.collect()
```

---

## 🚀 快速开始

```bash
# 1. 测试内存 (必须！)
python3 test_memory_single.py

# 2. 根据输出建议选择并行数
# 如果显示"安全并行数：4"
python3 train_parallel_safe.py --parallel 4 --episodes 200

# 如果显示"安全并行数：2"
python3 train_parallel_safe.py --parallel 2 --batch-size 2 --episodes 200

# 如果显示"内存紧张"
python3 train_dqn_paper_aligned.py
```

---

**记住**: 先测试，再并行！不要直接 4 进程！🛡️
