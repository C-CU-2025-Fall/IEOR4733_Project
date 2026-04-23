# DQN Walk-Forward 训练计划 (V2 - 稳健版)

**Last Updated**: 2026-04-22  
**Status**: 分阶段验证 + Early Stopping + 断点续训

---

## ⚠️ V1 计划的问题

| 问题 | V1 缺陷 | V2 改进 |
|------|---------|---------|
| **风险集中** | 一次性训练 50 episodes，结果差则浪费时间 | 分阶段验证，每 10 episodes 检查 |
| **无 Early Stopping** | 固定 50 episodes，无收敛判断 | 学习曲线监控 + 自动停止 |
| **无法续训** | 中断后重头开始 | Checkpoint 支持 resume |
| **无质量门禁** | 训练完才回测，可能全错 | 每阶段回测验证，差则停止 |
| **50 依据不明** | 随意选择 | 根据收敛曲线动态决定 |

---

## 🎯 核心改进

### 1. 分阶段训练 + 质量门禁

```
Forex (9 contracts)
  ↓
Phase 1A: 10 episodes → 回测验证 → 对齐度检查
  ↓ (如果对齐度差，停止并诊断)
Phase 1B: +20 episodes → 回测验证
  ↓ (如果收敛良好，继续)
Phase 1C: +20 episodes → 最终验证
  ↓
Fixed Income (5 contracts)
  ↓
Phase 2A: 10 episodes → 回测验证
  ...
```

### 2. Early Stopping 标准

| 指标 | 阈值 | 动作 |
|------|------|------|
| Reward 连续 5 ep 下降 | -10% | 降低学习率 |
| Reward 连续 10 ep 持平 | ±5% | 停止训练 |
| 回测 E(R) 误差 | >50% | 停止并诊断 |
| 回测 Sharpe 误差 | >30% | 停止并诊断 |

### 3. Checkpoint 续训

每 10 episodes 保存完整状态：
- 模型权重 (`q_net.state_dict()`)
- Optimizer 状态 (`opt.state_dict()`)
- Replay Buffer
- 训练步数
- 学习曲线数据

续训命令：
```bash
python3 dqn/train/train_dqn_walkforward.py --round 1 --asset Forex --resume --episodes 50
```

---

## 📊 训练流程 (稳健版)

### Phase 0: 基线确认 (✅ 1 episode)

**目的**: 确认 pipeline 正常工作

```bash
python3 dqn/train/train_dqn_walkforward.py --round 1 --asset Forex --episodes 1
```

**检查**:
- [ ] 9/9 模型成功保存
- [ ] 无报错
- [ ] 训练时间合理 (~1.5s/contract)

---

### Phase 1A: Forex 探索性训练 (10 episodes)

**目的**: 快速验证对齐度，避免大量浪费

```bash
# Round 1 - 10 episodes
python3 dqn/train/train_dqn_walkforward.py --round 1 --asset Forex --episodes 10

# 回测验证
python3 dqn/backtest/backtest_dqn_walkforward.py --round 1 --asset Forex
```

**质量门禁** (与论文 Table 3 Long Only 对比):

| 指标 | 可接受误差 | 行动 |
|------|------------|------|
| E(R) | <50% | ✅ 继续 / ❌ 停止诊断 |
| std(R) | <30% | ✅ 继续 / ❌ 停止诊断 |
| Sharpe | <40% | ✅ 继续 / ❌ 停止诊断 |
| % +ve | <20% | ✅ 继续 / ❌ 停止诊断 |

**如果 ❌**:
1. 检查特征工程 (feature distribution)
2. 检查 Reward 计算 (Eq.4 对齐)
3. 检查超参数 (lr, gamma, batch_size)
4. 检查数据质量 (source overrides)

**时间**: ~2 分钟 (9 contracts × 10 ep)

---

### Phase 1B: Forex 正式训练 (+20 episodes)

**前提**: Phase 1A 通过质量门禁

```bash
# 续训到 30 episodes
python3 dqn/train/train_dqn_walkforward.py --round 1 --asset Forex --episodes 30 --resume
```

**检查**:
- [ ] 学习曲线收敛 (reward 稳定)
- [ ] 无梯度爆炸/消失
- [ ] GPU 利用率正常

**时间**: ~4 分钟

---

### Phase 1C: Forex 最终训练 (+20 episodes)

**前提**: Phase 1B 学习曲线良好

```bash
# 续训到 50 episodes
python3 dqn/train/train_dqn_walkforward.py --round 1 --asset Forex --episodes 50 --resume

# Round 2 重复 Phase 1A→1B→1C
python3 dqn/train/train_dqn_walkforward.py --round 2 --asset Forex --episodes 10
python3 dqn/backtest/backtest_dqn_walkforward.py --round 2 --asset Forex
# ... 如果通过，继续到 50 episodes
```

**最终验证**:
```bash
python3 dqn/backtest/backtest_dqn_walkforward.py --all --asset Forex
```

**时间**: ~4 分钟 (Round 1) + ~10 分钟 (Round 2) = ~14 分钟

---

### Phase 2-4: 其他资产类别

**仅当 Forex 最终验证通过后才继续**

每个资产类别遵循相同流程：
1. **数据准备** (~1 分钟)
2. **Phase A**: 10 episodes + 回测验证
3. **Phase B**: +20 episodes (如果 A 通过)
4. **Phase C**: +20 episodes (如果 B 收敛)

| Asset | 数据准备 | Phase A | Phase B | Phase C | Total |
|-------|----------|---------|---------|---------|-------|
| Fixed Income | ~1 min | ~1 min | ~2 min | ~2 min | ~6 min |
| Equity Index | ~2 min | ~2 min | ~4 min | ~4 min | ~12 min |
| Commodity | ~5 min | ~5 min | ~10 min | ~10 min | ~30 min |

---

## 🛠️ 技术实现

### Checkpoint 格式

```python
checkpoint = {
    'episode': 30,
    'total_steps': 15000,
    'q_net_state_dict': ...,
    'optimizer_state_dict': ...,
    'replay_buffer': {
        'states': ...,
        'actions': ...,
        'rewards': ...,
        'next_states': ...,
        'dones': ...,
        'position': ...,
        'length': ...
    },
    'learning_curve': {
        'episodes': [...],
        'avg_rewards': [...],
        'losses': [...]
    },
    'metadata': {
        'ticker': 'AN',
        'round': 1,
        'train_start': '2005-01-01',
        'train_end': '2009-12-31',
    }
}
```

### 续训逻辑

```python
def load_checkpoint(path):
    ckpt = torch.load(path)
    agent.q_net.load_state_dict(ckpt['q_net_state_dict'])
    agent.target.load_state_dict(ckpt['q_net_state_dict'])
    agent.opt.load_state_dict(ckpt['optimizer_state_dict'])
    agent.buffer.load(ckpt['replay_buffer'])
    agent.train_steps = ckpt['total_steps']
    return ckpt['episode'], ckpt['learning_curve']

def train_with_resume(ticker, round_num, target_episodes, resume=False):
    if resume:
        ckpt_path = f"{MODEL_DIR}/{ticker}_r{round_num}_ckpt.pt"
        if os.path.exists(ckpt_path):
            start_ep, curve = load_checkpoint(ckpt_path)
            print(f"Resuming from episode {start_ep}")
        else:
            start_ep = 0
    else:
        start_ep = 0
    
    for ep in range(start_ep, target_episodes):
        # ... training loop
        if (ep + 1) % 10 == 0:
            save_checkpoint(ep + 1, curve)
```

### Early Stopping 实现

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.05):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_reward = None
        self.should_stop = False
    
    def __call__(self, avg_reward):
        if self.best_reward is None:
            self.best_reward = avg_reward
        elif avg_reward < self.best_reward * (1 - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            self.best_reward = avg_reward
        else:
            self.counter = 0
```

---

## 📈 监控与日志

### 训练日志结构

```
logs/dqn_train/
├── forex_r1_20260422_174500/
│   ├── training.log          # 完整输出
│   ├── learning_curve.json   # reward/loss 数据
│   ├── checkpoints/          # 每 10 ep 保存
│   │   ├── AN_r1_ep10.ckpt
│   │   ├── AN_r1_ep20.ckpt
│   │   └── ...
│   └── backtest_results/     # 阶段回测
│       ├── ep10_metrics.json
│       ├── ep30_metrics.json
│       └── ep50_metrics.json
```

### 实时监控

```bash
# 训练输出
tail -f logs/dqn_train/forex_r1_*/training.log

# GPU 监控
watch -n 2 nvidia-smi

# 学习曲线 (Python)
python3 scripts/plot_learning_curve.py logs/dqn_train/forex_r1_*/learning_curve.json
```

---

## ⏱️ 总时间预估 (含验证)

| Phase | 训练 | 回测验证 | 决策时间 | Total |
|-------|------|----------|----------|-------|
| Phase 0 (Test) | ~15s | - | - | ~15s |
| Phase 1A (Forex 10ep) | ~2 min | ~1 min | ~2 min | ~5 min |
| Phase 1B (Forex +20ep) | ~4 min | ~1 min | ~1 min | ~6 min |
| Phase 1C (Forex +20ep) | ~4 min | ~2 min | - | ~6 min |
| **Forex Total** | | | | **~17 min** |
| Phase 2 (Fixed Income) | ~5 min | ~1 min | ~1 min | ~7 min |
| Phase 3 (Equity Index) | ~10 min | ~2 min | ~2 min | ~14 min |
| Phase 4 (Commodity) | ~25 min | ~5 min | ~5 min | ~35 min |
| **Grand Total** | | | | **~73 min** |

**最坏情况** (某阶段失败需诊断): +30-60 分钟

---

## 🚨 停止条件

### 立即停止并诊断

1. **Phase 1A 回测失败**: E(R) 误差 >50% 或 Sharpe 误差 >40%
2. **训练发散**: Loss 爆炸 (NaN/Inf)
3. **梯度问题**: 连续 10 ep reward 下降 >20%
4. **数据问题**: 特征包含 NaN/Inf

### 诊断清单

```markdown
## DQN 训练问题诊断

### 1. 特征检查
- [ ] 特征值范围合理 (无极端值)
- [ ] 特征分布稳定 (无 drift)
- [ ] 特征与论文对齐 (8 维)

### 2. Reward 检查
- [ ] Eq.4 实现正确 (gross - tc)
- [ ] Transaction cost 合理
- [ ] Reward 量级正常

### 3. 超参数检查
- [ ] Learning rate (0.0001)
- [ ] Gamma (0.3)
- [ ] Batch size (64)
- [ ] Epsilon decay

### 4. 数据检查
- [ ] Source overrides 正确
- [ ] 训练/测试分割正确
- [ ] 无 look-ahead bias
```

---

## 📝 快速命令参考

### 带检查点的训练
```bash
# 开始新训练 (自动保存 checkpoint)
python3 dqn/train/train_dqn_walkforward.py \
  --round 1 --asset Forex --episodes 10 \
  --checkpoint-interval 10 \
  --log-dir logs/dqn_train/forex_r1

# 续训
python3 dqn/train/train_dqn_walkforward.py \
  --round 1 --asset Forex --episodes 30 \
  --resume \
  --checkpoint-interval 10
```

### 回测验证
```bash
# 单 round
python3 dqn/backtest/backtest_dqn_walkforward.py --round 1 --asset Forex

# 生成对比报告
python3 dqn/backtest/backtest_dqn_walkforward.py \
  --round 1 --asset Forex \
  --output-json logs/dqn_train/forex_r1/backtest_ep10.json
```

### 学习曲线可视化
```bash
python3 scripts/plot_learning_curve.py \
  logs/dqn_train/forex_r1/learning_curve.json \
  --output logs/dqn_train/forex_r1/curve.png
```

---

## ✅ 下一步

1. **更新训练脚本**: 添加 checkpoint 续训 + early stopping
2. **Phase 1A**: 运行 Forex 10 episodes + 回测验证
3. **决策**: 根据对齐度决定是否继续

---

**核心原则**: 快速验证 → 小步迭代 → 质量门禁 → 稳健扩展
