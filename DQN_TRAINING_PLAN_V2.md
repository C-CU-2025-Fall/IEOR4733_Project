# DQN Walk-Forward Training Plan (V2 - Robust)

**Last Updated**: 2026-04-22  
**Status**: Phased validation + Early Stopping + Checkpoint resume

---

## ⚠️ V1 Plan Issues

| Issue | V1 Deficiency | V2 Improvement |
|------|---------|---------|
| **Concentrated risk** | Train 50 episodes at once, waste time if results are bad | Phased validation, check every 10 episodes |
| **No Early Stopping** | Fixed 50 episodes, no convergence check | Learning curve monitoring + auto-stop |
| **Cannot resume** | Restart from scratch after interruption | Checkpoint supports resume |
| **No quality gates** | Backtest only after training, might all be wrong | Backtest validation each phase, stop if poor |
| **50 episodes arbitrary** | Randomly chosen | Dynamically determined from convergence curve |

---

## 🎯 Core Improvements

### 1. Phased Training + Quality Gates

```
Forex (9 contracts)
  ↓
Phase 1A: 10 episodes → backtest validation → alignment check
  ↓ (stop and diagnose if alignment is poor)
Phase 1B: +20 episodes → backtest validation
  ↓ (continue if convergence is good)
Phase 1C: +20 episodes → final validation
  ↓
Fixed Income (5 contracts)
  ↓
Phase 2A: 10 episodes → backtest validation
  ...
```

### 2. Early Stopping Criteria

| Metric | Threshold | Action |
|------|------|------|
| Reward drops for 5 consecutive ep | -10% | Reduce learning rate |
| Reward flat for 10 consecutive ep | ±5% | Stop training |
| Backtest E(R) error | >50% | Stop and diagnose |
| Backtest Sharpe error | >30% | Stop and diagnose |

### 3. Checkpoint Resume

Save full state every 10 episodes:
- Model weights (`q_net.state_dict()`)
- Optimizer state (`opt.state_dict()`)
- Replay Buffer
- Training steps
- Learning curve data

Resume command:
```bash
python3 dqn/train/train_dqn_walkforward.py --round 1 --asset Forex --resume --episodes 50
```

---

## 📊 Training Flow (Robust)

### Phase 0: Baseline Confirmation (✅ 1 episode)

**Purpose**: Confirm pipeline works correctly

```bash
python3 dqn/train/train_dqn_walkforward.py --round 1 --asset Forex --episodes 1
```

**Checks**:
- [ ] 9/9 models saved successfully
- [ ] No errors
- [ ] Training time reasonable (~1.5s/contract)

---

### Phase 1A: Forex Exploratory Training (10 episodes)

**Purpose**: Quickly validate alignment, avoid large waste

```bash
# Round 1 - 10 episodes
python3 dqn/train/train_dqn_walkforward.py --round 1 --asset Forex --episodes 10

# Backtest validation
python3 dqn/backtest/backtest_dqn_walkforward.py --round 1 --asset Forex
```

**Quality Gate** (vs paper Table 3 Long Only):

| Metric | Acceptable Error | Action |
|------|------------|------|
| E(R) | <50% | ✅ Continue / ❌ Stop & diagnose |
| std(R) | <30% | ✅ Continue / ❌ Stop & diagnose |
| Sharpe | <40% | ✅ Continue / ❌ Stop & diagnose |
| % +ve | <20% | ✅ Continue / ❌ Stop & diagnose |

**If ❌**:
1. Check feature engineering (feature distribution)
2. Check Reward calculation (Eq.4 alignment)
3. Check hyperparameters (lr, gamma, batch_size)
4. Check data quality (source overrides)

**Time**: ~2 min (9 contracts × 10 ep)

---

### Phase 1B: Forex Full Training (+20 episodes)

**Prerequisite**: Phase 1A passes quality gate

```bash
# Resume to 30 episodes
python3 dqn/train/train_dqn_walkforward.py --round 1 --asset Forex --episodes 30 --resume
```

**Checks**:
- [ ] Learning curve converges (reward stable)
- [ ] No gradient explosion/vanishing
- [ ] GPU utilization normal

**Time**: ~4 min

---

### Phase 1C: Forex Final Training (+20 episodes)

**Prerequisite**: Phase 1B learning curve is good

```bash
# Resume to 50 episodes
python3 dqn/train/train_dqn_walkforward.py --round 1 --asset Forex --episodes 50 --resume

# Round 2 repeats Phase 1A→1B→1C
python3 dqn/train/train_dqn_walkforward.py --round 2 --asset Forex --episodes 10
python3 dqn/backtest/backtest_dqn_walkforward.py --round 2 --asset Forex
# ... if passed, continue to 50 episodes
```

**Final validation**:
```bash
python3 dqn/backtest/backtest_dqn_walkforward.py --all --asset Forex
```

**Time**: ~4 min (Round 1) + ~10 min (Round 2) = ~14 min

---

### Phase 2-4: Other Asset Classes

**Only proceed after Forex final validation passes**

Each asset class follows the same flow:
1. **Data preparation** (~1 min)
2. **Phase A**: 10 episodes + backtest validation
3. **Phase B**: +20 episodes (if A passes)
4. **Phase C**: +20 episodes (if B converges)

| Asset | Data Prep | Phase A | Phase B | Phase C | Total |
|-------|----------|---------|---------|---------|-------|
| Fixed Income | ~1 min | ~1 min | ~2 min | ~2 min | ~6 min |
| Equity Index | ~2 min | ~2 min | ~4 min | ~4 min | ~12 min |
| Commodity | ~5 min | ~5 min | ~10 min | ~10 min | ~30 min |

---

## 🛠️ Technical Implementation

### Checkpoint Format

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

### Resume Logic

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

### Early Stopping Implementation

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

## 📈 Monitoring & Logging

### Training Log Structure

```
logs/dqn_train/
├── forex_r1_20260422_174500/
│   ├── training.log          # full output
│   ├── learning_curve.json   # reward/loss data
│   ├── checkpoints/          # saved every 10 ep
│   │   ├── AN_r1_ep10.ckpt
│   │   ├── AN_r1_ep20.ckpt
│   │   └── ...
│   └── backtest_results/     # phased backtest
│       ├── ep10_metrics.json
│       ├── ep30_metrics.json
│       └── ep50_metrics.json
```

### Real-time Monitoring

```bash
# Training output
tail -f logs/dqn_train/forex_r1_*/training.log

# GPU monitoring
watch -n 2 nvidia-smi

# Learning curve (Python)
python3 scripts/plot_learning_curve.py logs/dqn_train/forex_r1_*/learning_curve.json
```

---

## ⏱️ Total Time Estimate (with validation)

| Phase | Training | Backtest Validation | Decision Time | Total |
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

**Worst case** (a phase fails requiring diagnosis): +30-60 min

---

## 🚨 Stop Conditions

### Stop immediately and diagnose

1. **Phase 1A backtest fails**: E(R) error >50% or Sharpe error >40%
2. **Training diverges**: Loss explodes (NaN/Inf)
3. **Gradient issues**: Reward drops >20% for 10 consecutive ep
4. **Data issues**: Features contain NaN/Inf

### Diagnosis Checklist

```markdown
## DQN Training Problem Diagnosis

### 1. Feature Check
- [ ] Feature value ranges reasonable (no extreme values)
- [ ] Feature distribution stable (no drift)
- [ ] Features aligned with paper (8 dimensions)

### 2. Reward Check
- [ ] Eq.4 implemented correctly (gross - tc)
- [ ] Transaction cost reasonable
- [ ] Reward magnitude normal

### 3. Hyperparameter Check
- [ ] Learning rate (0.0001)
- [ ] Gamma (0.3)
- [ ] Batch size (64)
- [ ] Epsilon decay

### 4. Data Check
- [ ] Source overrides correct
- [ ] Train/test split correct
- [ ] No look-ahead bias
```

---

## 📝 Quick Command Reference

### Training with checkpoints
```bash
# Start new training (auto-save checkpoint)
python3 dqn/train/train_dqn_walkforward.py \
  --round 1 --asset Forex --episodes 10 \
  --checkpoint-interval 10 \
  --log-dir logs/dqn_train/forex_r1

# Resume training
python3 dqn/train/train_dqn_walkforward.py \
  --round 1 --asset Forex --episodes 30 \
  --resume \
  --checkpoint-interval 10
```

### Backtest validation
```bash
# Single round
python3 dqn/backtest/backtest_dqn_walkforward.py --round 1 --asset Forex

# Generate comparison report
python3 dqn/backtest/backtest_dqn_walkforward.py \
  --round 1 --asset Forex \
  --output-json logs/dqn_train/forex_r1/backtest_ep10.json
```

### Learning curve visualization
```bash
python3 scripts/plot_learning_curve.py \
  logs/dqn_train/forex_r1/learning_curve.json \
  --output logs/dqn_train/forex_r1/curve.png
```

---

## ✅ Next Steps

1. **Update training script**: Add checkpoint resume + early stopping
2. **Phase 1A**: Run Forex 10 episodes + backtest validation
3. **Decision**: Whether to proceed based on alignment

---

**Core principle**: Quick validation → Small-step iteration → Quality gates → Robust expansion
