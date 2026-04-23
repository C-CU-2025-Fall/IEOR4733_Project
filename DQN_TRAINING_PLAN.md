# DQN Walk-Forward 训练计划

**Last Updated**: 2026-04-22  
**Status**: 准备重新训练（现有 18 个 Forex 模型参数未知）

---

## 📊 资产类别概览

| Asset Class | Contracts | Rounds | Total Models | Priority |
|-------------|-----------|--------|--------------|----------|
| Forex | 9 | 2 | 18 | 1️⃣ (已有数据) |
| Fixed Income | 5 | 2 | 10 | 2️⃣ |
| Equity Index | 11 | 2 | 22 | 3️⃣ |
| Commodity | 25 | 2 | 50 | 4️⃣ |
| **Total** | **50** | **2** | **100** | - |

---

## 🎯 训练参数

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Episodes | 50 | 平衡训练质量与时间 |
| Max Steps/Ep | 500 | 覆盖足够的时间步 |
| Batch Size | 64 | Paper Table 1 |
| Learning Rate | 0.0001 | Paper Table 1 |
| Gamma | 0.3 | Paper Table 1 |
| LSTM Hidden | [64, 32] | Paper-aligned |
| Action Space | {-1, 0, +1} | Discrete |

---

## 📅 训练阶段

### Phase 0: 测试验证 (✅ 已完成)

**目标**: 验证 pipeline 正常工作

```bash
# 单合约 1 episode 测试
python3 dqn/train/train_dqn_walkforward.py --round 1 --asset Forex --episodes 1 --max-steps-per-ep 500
```

**结果**:
- ✅ 9/9 Forex contracts 成功保存
- 单合约时间: ~1.3s (1 episode, 500 steps)
- 步长时间: ~1.6ms/step (GPU)

---

### Phase 1: Forex 正式训练 (重新训练)

**目标**: 重新训练 Forex 18 个模型（50 episodes）

**数据状态**: ✅ 已准备 (18 files in `data/dqn_walkforward/`)

**命令**:
```bash
# Round 1 (9 contracts × 50 episodes)
python3 dqn/train/train_dqn_walkforward.py --round 1 --asset Forex --episodes 50

# Round 2 (9 contracts × 50 episodes)
python3 dqn/train/train_dqn_walkforward.py --round 2 --asset Forex --episodes 50
```

**时间预估**:
| Metric | Value |
|--------|-------|
| Per contract (50 ep) | ~65 秒 |
| Round 1 (9 contracts) | ~10 分钟 |
| Round 2 (9 contracts) | ~10 分钟 |
| **Total Forex** | **~20 分钟** |

**输出**: `models/dqn_walkforward/{AN,BN,CN,DX,FN,JN,MP,NK,SN}_r{1,2}.pt`

---

### Phase 2: Fixed Income 训练

**目标**: 训练 Fixed Income 10 个模型

**步骤 2.1: 准备数据**
```bash
python3 dqn/train/prepare_dqn_data.py --asset "Fixed Income" --round 1
python3 dqn/train/prepare_dqn_data.py --asset "Fixed Income" --round 2
```

**步骤 2.2: 训练**
```bash
python3 dqn/train/train_dqn_walkforward.py --round 1 --asset "Fixed Income" --episodes 50
python3 dqn/train/train_dqn_walkforward.py --round 2 --asset "Fixed Income" --episodes 50
```

**时间预估**:
| Metric | Value |
|--------|-------|
| Per contract (50 ep) | ~65 秒 |
| Round 1 (5 contracts) | ~5.5 分钟 |
| Round 2 (5 contracts) | ~5.5 分钟 |
| **Total Fixed Income** | **~11 分钟** |

---

### Phase 3: Equity Index 训练

**目标**: 训练 Equity Index 22 个模型

**步骤 3.1: 准备数据**
```bash
python3 dqn/train/prepare_dqn_data.py --asset "Equity Index" --round 1
python3 dqn/train/prepare_dqn_data.py --asset "Equity Index" --round 2
```

**步骤 3.2: 训练**
```bash
python3 dqn/train/train_dqn_walkforward.py --round 1 --asset "Equity Index" --episodes 50
python3 dqn/train/train_dqn_walkforward.py --round 2 --asset "Equity Index" --episodes 50
```

**时间预估**:
| Metric | Value |
|--------|-------|
| Per contract (50 ep) | ~65 秒 |
| Round 1 (11 contracts) | ~12 分钟 |
| Round 2 (11 contracts) | ~12 分钟 |
| **Total Equity Index** | **~24 分钟** |

---

### Phase 4: Commodity 训练

**目标**: 训练 Commodity 50 个模型

**步骤 4.1: 准备数据**
```bash
python3 dqn/train/prepare_dqn_data.py --asset Commodity --round 1
python3 dqn/train/prepare_dqn_data.py --asset Commodity --round 2
```

**步骤 4.2: 训练**
```bash
python3 dqn/train/train_dqn_walkforward.py --round 1 --asset Commodity --episodes 50
python3 dqn/train/train_dqn_walkforward.py --round 2 --asset Commodity --episodes 50
```

**时间预估**:
| Metric | Value |
|--------|-------|
| Per contract (50 ep) | ~65 秒 |
| Round 1 (25 contracts) | ~27 分钟 |
| Round 2 (25 contracts) | ~27 分钟 |
| **Total Commodity** | **~54 分钟** |

---

## 📈 总体时间预估

| Phase | Asset | Time | Cumulative |
|-------|-------|------|------------|
| 0 | Test (Forex 1ep) | ~15s | ~15s |
| 1 | Forex (50ep) | ~20 min | ~20 min |
| 2 | Fixed Income | ~11 min | ~31 min |
| 3 | Equity Index | ~24 min | ~55 min |
| 4 | Commodity | ~54 min | **~1.8 hours** |

**全部 100 个模型**: ~1.8 小时 (50 episodes/contract)

---

## 🔍 稳健性检查清单

### 训练前
- [ ] 确认 GPU 可用 (`nvidia-smi`)
- [ ] 确认数据文件存在 (`ls data/dqn_walkforward/`)
- [ ] 确认输出目录可写 (`ls -la models/`)
- [ ] 关闭其他 GPU 占用进程

### 训练中
- [ ] 监控 GPU 利用率 (`watch -n1 nvidia-smi`)
- [ ] 监控内存使用
- [ ] 检查 tqdm 输出是否正常
- [ ] 记录异常中断

### 训练后
- [ ] 验证模型文件存在 (`ls models/dqn_walkforward/*.pt`)
- [ ] 验证模型可加载 (test script)
- [ ] 运行回测验证
- [ ] 备份模型

---

## 🛠️ 快速命令参考

### 检查状态
```bash
# 查看已训练的模型
ls -lh models/dqn_walkforward/*.pt | wc -l

# 查看数据文件
ls data/dqn_walkforward/ | wc -l
```

### 数据准备
```bash
# 单个合约
python3 dqn/train/prepare_dqn_data.py --ticker ES --round 1

# 整个资产类别
python3 dqn/train/prepare_dqn_data.py --asset "Equity Index" --round 1
```

### 训练
```bash
# 快速测试 (1 episode)
python3 dqn/train/train_dqn_walkforward.py --round 1 --asset Forex --episodes 1

# 正式训练 (50 episodes)
python3 dqn/train/train_dqn_walkforward.py --round 1 --asset Forex --episodes 50

# 自定义参数
python3 dqn/train/train_dqn_walkforward.py --round 1 --asset Forex --episodes 100 --max-steps-per-ep 1000
```

### 回测
```bash
# 单个 round
python3 dqn/backtest/backtest_dqn_walkforward.py --round 1 --asset Forex

# 所有 rounds
python3 dqn/backtest/backtest_dqn_walkforward.py --all --asset Forex
```

---

## ⚠️ 注意事项

1. **现有模型**: 当前 `models/dqn_walkforward/` 中的 18 个 Forex 模型训练参数未知，建议重新训练
2. **数据一致性**: 确保使用 `config.py` 中的 `SOURCE_OVERRIDES` 保持一致的数据源
3. **中断恢复**: 训练脚本不支持断点续训，中断后需重新训练该合约
4. **GPU 内存**: LSTM 模型较小，单卡可容纳，无需特殊配置
5. **日志记录**: 建议重定向输出到日志文件以便审查

---

## 📝 训练日志模板

```bash
# 创建日志目录
mkdir -p logs/dqn_train

# 训练 Forex Round 1
python3 dqn/train/train_dqn_walkforward.py --round 1 --asset Forex --episodes 50 \
  2>&1 | tee logs/dqn_train/forex_r1_$(date +%Y%m%d_%H%M%S).log
```

---

**下一步**: 开始 Phase 1 (Forex 正式训练)
