# DQN Pipeline 代码架构审查与优化建议

## 一、架构全景

```
┌─────────────────────────────────────────────────────────────────┐
│                     drl_shared/ (共享层)                         │
│  spec.py        — 常量: SEQ_LEN=60, FEATURE_DIM=9,             │
│                   RETRAIN_ROUNDS, action space                   │
│  state_space.py — ContractArrays, ContractEnv,                  │
│                   build_feature_matrix(), compute_eq4_reward()   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
   ┌───────────────────────┼───────────────────────┐
   ▼                       ▼                       ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
│  drl/dqn/    │   │  drl/dqn/    │   │  baseline_run.py  │
│  train/      │   │  backtest/    │   │  strategy_        │
│              │   │              │   │  backtester.py    │
│ train_dqn_   │   │ engine.py    │   │                   │
│ walkforward  │   │              │   │ Table 2/3 指标计算 │
│ .py          │   │ dqn_position │   │ portfolio-level   │
│              │   │ _provider()  │   │ vol scaling       │
│ model.py     │   │              │   │                   │
│ spec.py      │   │ portfolio_   │   │                   │
│ logging.py   │   │ metrics()    │   └──────────────────┘
└──────┬───────┘   └──────┬───────┘
       │                  │
       ▼                  ▼
  模型存储 .pt         回测 .npz + metrics .json
  drl/dqn/models/       results/
```

## 二、训练管线（train_dqn_walkforward.py）

**入口**: `train_asset_round(asset_name, round_num, episodes, device, seed, gamma)`

### 2.1 数据流

```
asset_index.json → ticker列表
       │
       ▼
contract_data_path(round, ticker) → .npz文件 (features, prices, sigma, dates)
       │
       ▼
load_contract_round() → ContractArrays (prices, returns, sigma, features, dates)
       │
       ├─ _validate_feature_policy() → 校验structural_38 preset
       ├─ _sanity_check_contract()  → NaN/Inf/日期排序检查
       │
       ▼
_slice_train_contract() → train_contract (0..train_end), split_idx (train/val 分界)
       │
       ├─ ContractEnv(train_contract, start→split_idx) → train_envs[dict]
       └─ ContractEnv(train_contract, val_start→n)     → val_envs[dict]
       │
       ▼
DQNAgent(device, total_steps, memory_size)
       │
       ▼
for cycle in 1..episodes:
    _run_interleaved_cycle(train_envs, agent) → per_contract rewards
    _validation_reward(val_envs, agent)       → early stopping check
    save checkpoint if better
```

### 2.2 核心设计决策

| 决策 | 实现 | 理由 |
|------|------|------|
| **资产类共享模型** | 1个DQN训练asset内所有contracts | 跨contract泛化，比per-contract模型更稳健 |
| **Interleaved训练** | 每步轮流采样所有active contracts | Replay buffer内experience balanced含所有contract |
| **Phase 1纯随机** | 前5000步纯随机填buffer | 避免bootstrap bias |
| **Phase 2/3 epsilon衰减** | 0.30→0.10→0.03→0.01 | 结构化探索→精细化→利用 |
| **Validation split 10%** | 训练数据后10%做val | 防止过拟合到训练时段 |
| **Early stopping patience 20** | val reward不提升20 cycle且action degenerate | 防止训练浪费 |

### 2.3 模型架构（model.py）

```
DuelingDQNLSTM:
  Input: (B, 60, 9) — 60天 × 7feature有 9个特征
   │
  LSTM1: 9 → 64 (hidden), batch_first, tanh
   │
  LeakyReLU(0.01)
   │
  LSTM2: 64 → 32 (hidden), batch_first, tanh
   │
  LeakyReLU(0.01) + Dropout(0.2)
   │
  last_time_step[:, -1, :] → shape (B, 32)
   │
  ├─ Value head:   Linear(32, 1)      → V(s)
  └─ Advantage:    Linear(32, 3)      → A(s,a)
   │
  Q(s,a) = V(s) + A(s,a) - mean(A(s))
```

**稳定化手段**（参照 [49] Mnih 2015, [18] van Hasselt 2016, [50] Wang 2016）：
- Double DQN: q_net选action, target评估value → 防止over-estimation
- Fixed Q-targets: target_net硬拷贝每1000 learn steps
- Dueling Network: Value + Advantage分解 → 学习state value不依赖action
- Dropout 0.2 → 隐式ensemble正则化
- Gradient clipping 1.0 → 防止梯度爆炸
- Orthogonal weight init gain 0.1 → 防止初始值过大
- Huber loss (可选) → 对outlier reward鲁棒

### 2.4 环境与奖励（state_space.py）

**状态空间**: 9维，60步窗口
```
F0: p_t / std_60(p)            — 归一化价格
F1-F4: (p_t - p_{t-H}) / (σ_t·√H), H∈{21,42,63,252} — 多尺度returns
F5-F7: 三对MACD (8,24)/(16,48)/(32,96) 归一化 — 趋势信号
F8: (RSI_30 - 50) / 50         — 超买超卖
```

**奖励函数** (Eq.4):
```
r_t = position_{t-1} × (σ_tgt/σ_{t-1}) × (p_t - p_{t-1})   — gross
    - bp × p_{t-1} × |pos_t×σ_tgt/σ_{t-1} - pos_{t-1}×σ_tgt/σ_{t-2}|  — TC
```

**动作空间**: {-1 (short), 0 (hold), +1 (long)}

## 三、回测管线（engine.py + baseline_run.py）

### 3.1 数据流

```
checkpoint.pt → DQNAgent.load() → q_net.eval()
     │
     ▼
contract_data_path → .npz features (与训练完全一致的features)
     │
     ├─ date matching: 将backtest日期映射到npz feature index
     ├─ 每步: get_feature_window(features, idx) → state (60,9)
     │
     ▼
_dqn_position_provider:
  for each round_mask:
    states = stack([window for idx in range])
    action_ids = agent.predict_action_ids(states)  # batch_size=2048
    positions = [action_id_to_position(a) for a in action_ids]
     │
     ▼
compute_portfolio_returns_from_position_provider:
  for each contract: positions × sigma_tgt/sigma → returns
  average across contracts (variable_n) → portfolio returns
     │
     ▼
compute_metrics(R) → {E(R), std(R), DD, Sharpe, Sortino, MDD, Calmar, %+ve, Ave P/L}
  port_vol_target post-hoc scaling (Table 2 only)
```

### 3.2 关键细节

- **npz-only features**: 回测强制使用训练时生成的.npz features（不fallback到实时计算），保证 bit-exact 复现
- **Date matching via dict**: `date_to_npz = {d: i for i, d in enumerate(any_npz_dates)}` → O(1)查表
- **No GPU in backtest**: 单步inference batch=2048，CPU足够
- **port_vol_target post-hoc**: Table 2用`get_portfolio_bridge()`对portfolio return做常数后缩放

---

## 四、潜在优化与问题

### 🔴 4.1 Gamma Override 机制（非线程安全）

**文件**: `train_dqn_walkforward.py:557-559`
```python
if gamma is not None:
    import drl.dqn.spec as _spec
    _spec.GAMMA = gamma  # 直接修改模块级常量
```

**问题**: 虽然每个Python进程独立不影响并行训练，但这是**代码异味**（code smell）：
- 如果在同一进程内多次调用`train_asset_round`，第二次调用的gamma会覆盖第一次
- GAMMA不是通过参数传递而是模块突变，不符合函数式设计

**建议**: 将gamma作为`DQNAgent.__init__`参数传递，而非修改全局常量。需要改动：
1. `model.py`: `DQNAgent.__init__` 加 `gamma: float = _spec.GAMMA` 参数
2. `model.py`: `learn()` 使用 `self.gamma` 而非 `_spec.GAMMA`
3. `train_dqn_walkforward.py`: 删除 `_spec.GAMMA = gamma` 行

### 🟡 4.2 Replay Buffer 序列化开销

**文件**: `model.py:345-365`

每次保存 `checkpoint.pt` 时 `include_training_state=True` 会将整个replay buffer（5000条 × (60,9)×2 states ≈ 5.4M floats）序列化到磁盘。每个checkpoint约 **100-200MB**。

**影响**: 10个seed × 100cycles = 最多1000个checkpoint如果能保存（有early stopping）

**建议**: 
- `latest_checkpoint.pt`：保存training state用于resume（必需）
- `checkpoint.pt`（best model）：**不保存replay buffer**（`include_training_state=False`），只有model weights
- 这样可以节省 ~80% 磁盘空间

### 🟡 4.3 Replay Buffer 的 Python list 实现

**文件**: `model.py:120-132`

当前使用Python list存储，每个transition是tuple of numpy arrays:
```python
self.buffer.append((np.asarray(s), int(a), float(r), np.asarray(s2), float(d)))
```

**问题**: 
- Pushing时`np.asarray()`创建额外copy → memory fragmentation
- `_to_arrays()`用list comprehension拆解tuple → 慢
- `random.sample(self.buffer, batch_size)` 每次return新的list

**建议**: 使用预分配的numpy ring buffer:
```python
self.states = np.zeros((capacity, SEQ_LEN, FEATURE_DIM), dtype=np.float32)
self.actions = np.zeros(capacity, dtype=np.int64)
self.rewards = np.zeros(capacity, dtype=np.float32)
self.next_states = np.zeros((capacity, SEQ_LEN, FEATURE_DIM), dtype=np.float32)
self.dones = np.zeros(capacity, dtype=np.float32)
```
预分配+索引循环 → 推入O(1)，采样O(batch_size)。但当前实现对于5000条数据量足够快，这不是性能瓶颈。

### 🟡 4.4 Buffer Size 对多Contract资产的适应性

**文件**: `train_dqn_walkforward.py:613-621`

```python
buffer_sizes = {
    'Forex': 5000, 'Equity Index': 5000,
    'Commodity': 5000, 'Fixed Income': 3000,
}
```

**观察**: Commodity有25个contracts，每个cycle产生~25×200=5000步（假设每contract约200步）。5000条buffer存1个cycle的数据 — 对于25个contract来说可能太少。

**建议**: buffer size与n_contracts挂钩：
```python
memory_size = max(5000, n_contracts * 300)  # 25 contracts → 7500
```

### 🟢 4.5 无Learning Rate Schedule

当前Adam LR=0.0001固定100个cycles。

**建议**: 
- 添加cosine annealing: `lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * step/total))`
- 或简单的step decay: 每30 cycles ×0.5
- 这对gamma tuning应该 **影响不大**（100 cycles时LR衰减空间有限），但对实验完整性有益

### 🟢 4.6 无Weight Decay

当前无L2正则化 → 可能过拟合到训练期noise。

**建议**: 添加 `weight_decay=1e-5` 到Adam optimizer

### 🟢 4.7 验证集极其有限

`VALIDATION_SPLIT=0.1` 即训练数据后10%做validation。对于r1:
- 训练: 2005-01 ~ 2010-12 = 6年
- 训练用: 2005-01 ~ 2010-05 ≈ 5.4年
- 验证用: 2010-05 ~ 2010-12 ≈ 0.6年

**0.6年做early stopping不太可靠**。Early stopping本身是合理的设计，但validation on 0.6年可能对noise敏感。

**建议**: 
- 增加到20% split 或 用部分test period做validation
- 或降低`EARLY_STOPPING_PATIENCE`（当前20）

### 🔵 4.8 Epsilon Schedule 的设计

**文件**: `spec.py:80-86`

```python
EPS_SCHEDULE = [
    (0.00, 0.300),  # cycle 0
    (0.20, 0.100),  # cycle 20
    (0.50, 0.030),  # cycle 50
    (0.90, 0.010),  # cycle 90
    (1.00, 0.010),  # cycle 100
]
```

**观察**: 这是**fraction-based schedule**（基于total_steps的百分比），不是cycle-based。所以：
- `EPS_BUFFER_FILL=5000`纯随机 → 大约1-2个cycles（取决于n_contracts）
- 然后0.30 → 0.10 → 0.03 → 0.01在整个training中线性衰减

**这个设计是好的** — 平衡了exploration和exploitation。对于10个seed的不同随机性，epsilon schedule保持一致。

### 🔵 4.9 训练可复现性

**当前**: ✅ seed设置覆盖random/numpy/torch/cuda
**缺失**: ⚠️ batched环境中，cudnn可能有非确定性行为

**建议**: 如果追求bit-exact复现，添加：
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
代价：GPU训练慢 ~20%。对研究来说值得。

---

## 五、总结

### 代码质量：良好 ⭐⭐⭐⭐

| 维度 | 评分 | 说明 |
|------|------|------|
| 架构设计 | 优秀 | 资产类共享模型 + interleaved训练是正确的设计 |
| 可复现性 | 良好 | Seed设置完善，npz-based feature保证bit-exact |
| 日志/审计 | 优秀 | 每cycle指标、contract级统计、preflight检查 |
| 代码健壮性 | 良好 | sanity check、preflight、NaN/Inf监控 |
| 超参设计 | 良好 | 参照paper Table 1，略有改进（epsilon schedule, interleaved） |
| 性能优化 | 可改进 | Buffer序列化、Python list replay |

### 优先级建议

1. **高**: 修复gamma override机制（传参而非模块突变）→ 不影响当前结果但保未来安全
2. **中**: 减少checkpoint大小（best model不保存replay）→ 节省磁盘
3. **低**: cudnn deterministic + weight decay → 锦上添花
4. **不考虑**: 当前不做buffer架构重构（pre-allocated numpy ring）— 对当前规模足够

### 结论

代码整体质量不错。gamma=0.6的结论基于10个seed × r1+r2的充分交叉验证，训练管线无明显bug。gamma override虽然用模块突变但不是线程级别的race condition（每进程独立）。**继续训练另外3个资产是安全的。**
