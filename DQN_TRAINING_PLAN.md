# DQN Walk-Forward Training Plan

**Last Updated**: 2026-04-22  
**Status**: Preparing retraining (existing 18 Forex model parameters unknown)

---

## Asset Class Overview

| Asset Class | Contracts | Rounds | Total Models | Priority |
|-------------|-----------|--------|--------------|----------|
| Forex | 9 | 2 | 18 | 1️⃣ (data ready) |
| Fixed Income | 5 | 2 | 10 | 2️⃣ |
| Equity Index | 11 | 2 | 22 | 3️⃣ |
| Commodity | 25 | 2 | 50 | 4️⃣ |
| **Total** | **50** | **2** | **100** | - |

---

## Training Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Episodes | 50 | Balance training quality and time |
| Max Steps/Ep | 500 | Cover sufficient time steps |
| Batch Size | 64 | Paper Table 1 |
| Learning Rate | 0.0001 | Paper Table 1 |
| Gamma | 0.3 | Paper Table 1 |
| LSTM Hidden | [64, 32] | Paper-aligned |
| Action Space | {-1, 0, +1} | Discrete |

---

## Training Phases

### Phase 0: Test Validation (✅ Completed)

**Goal**: Verify pipeline works correctly

```bash
# Single-contract 1-episode test
python3 dqn/train/train_dqn_walkforward.py --round 1 --asset Forex --episodes 1 --max-steps-per-ep 500
```

**Results**:
- ✅ 9/9 Forex contracts saved successfully
- Per-contract time: ~1.3s (1 episode, 500 steps)
- Step time: ~1.6ms/step (GPU)

---

### Phase 1: Forex Full Training (Retraining)

**Goal**: Retrain 18 Forex models (50 episodes)

**Data Status**: ✅ Ready (18 files in `data/dqn_walkforward/`)

**Command**:
```bash
# Round 1 (9 contracts × 50 episodes)
python3 dqn/train/train_dqn_walkforward.py --round 1 --asset Forex --episodes 50

# Round 2 (9 contracts × 50 episodes)
python3 dqn/train/train_dqn_walkforward.py --round 2 --asset Forex --episodes 50
```

**Time Estimate**:
| Metric | Value |
|--------|-------|
| Per contract (50 ep) | ~65s |
| Round 1 (9 contracts) | ~10 min |
| Round 2 (9 contracts) | ~10 min |
| **Total Forex** | **~20 min** |

**Output**: `models/dqn_walkforward/{AN,BN,CN,DX,FN,JN,MP,NK,SN}_r{1,2}.pt`

---

### Phase 2: Fixed Income Training

**Goal**: Train 10 Fixed Income models

**Step 2.1: Prepare data**
```bash
python3 dqn/train/prepare_dqn_data.py --asset "Fixed Income" --round 1
python3 dqn/train/prepare_dqn_data.py --asset "Fixed Income" --round 2
```

**Step 2.2: Train**
```bash
python3 dqn/train/train_dqn_walkforward.py --round 1 --asset "Fixed Income" --episodes 50
python3 dqn/train/train_dqn_walkforward.py --round 2 --asset "Fixed Income" --episodes 50
```

**Time Estimate**:
| Metric | Value |
|--------|-------|
| Per contract (50 ep) | ~65s |
| Round 1 (5 contracts) | ~5.5 min |
| Round 2 (5 contracts) | ~5.5 min |
| **Total Fixed Income** | **~11 min** |

---

### Phase 3: Equity Index Training

**Goal**: Train 22 Equity Index models

**Step 3.1: Prepare data**
```bash
python3 dqn/train/prepare_dqn_data.py --asset "Equity Index" --round 1
python3 dqn/train/prepare_dqn_data.py --asset "Equity Index" --round 2
```

**Step 3.2: Train**
```bash
python3 dqn/train/train_dqn_walkforward.py --round 1 --asset "Equity Index" --episodes 50
python3 dqn/train/train_dqn_walkforward.py --round 2 --asset "Equity Index" --episodes 50
```

**Time Estimate**:
| Metric | Value |
|--------|-------|
| Per contract (50 ep) | ~65s |
| Round 1 (11 contracts) | ~12 min |
| Round 2 (11 contracts) | ~12 min |
| **Total Equity Index** | **~24 min** |

---

### Phase 4: Commodity Training

**Goal**: Train 50 Commodity models

**Step 4.1: Prepare data**
```bash
python3 dqn/train/prepare_dqn_data.py --asset Commodity --round 1
python3 dqn/train/prepare_dqn_data.py --asset Commodity --round 2
```

**Step 4.2: Train**
```bash
python3 dqn/train/train_dqn_walkforward.py --round 1 --asset Commodity --episodes 50
python3 dqn/train/train_dqn_walkforward.py --round 2 --asset Commodity --episodes 50
```

**Time Estimate**:
| Metric | Value |
|--------|-------|
| Per contract (50 ep) | ~65s |
| Round 1 (25 contracts) | ~27 min |
| Round 2 (25 contracts) | ~27 min |
| **Total Commodity** | **~54 min** |

---

## 📈 Overall Time Estimate

| Phase | Asset | Time | Cumulative |
|-------|-------|------|------------|
| 0 | Test (Forex 1ep) | ~15s | ~15s |
| 1 | Forex (50ep) | ~20 min | ~20 min |
| 2 | Fixed Income | ~11 min | ~31 min |
| 3 | Equity Index | ~24 min | ~55 min |
| 4 | Commodity | ~54 min | **~1.8 hours** |

**All 100 models**: ~1.8 hours (50 episodes/contract)

---

## 🔍 Robustness Checklist

### Pre-training
- [ ] Confirm GPU available (`nvidia-smi`)
- [ ] Confirm data files exist (`ls data/dqn_walkforward/`)
- [ ] Confirm output directory writable (`ls -la models/`)
- [ ] Close other GPU-consuming processes

### During training
- [ ] Monitor GPU utilization (`watch -n1 nvidia-smi`)
- [ ] Monitor memory usage
- [ ] Check tqdm output is normal
- [ ] Record any abnormal interruptions

### Post-training
- [ ] Verify model files exist (`ls models/dqn_walkforward/*.pt`)
- [ ] Verify models are loadable (test script)
- [ ] Run backtest validation
- [ ] Backup models

---

## Quick Command Reference

### Check status
```bash
# View trained models
ls -lh models/dqn_walkforward/*.pt | wc -l

# View data files
ls data/dqn_walkforward/ | wc -l
```

### Data preparation
```bash
# Single contract
python3 dqn/train/prepare_dqn_data.py --ticker ES --round 1

# Entire asset class
python3 dqn/train/prepare_dqn_data.py --asset "Equity Index" --round 1
```

### Training
```bash
# Quick test (1 episode)
python3 dqn/train/train_dqn_walkforward.py --round 1 --asset Forex --episodes 1

# Full training (50 episodes)
python3 dqn/train/train_dqn_walkforward.py --round 1 --asset Forex --episodes 50

# Custom parameters
python3 dqn/train/train_dqn_walkforward.py --round 1 --asset Forex --episodes 100 --max-steps-per-ep 1000
```

### Backtest
```bash
# Single round
python3 dqn/backtest/backtest_dqn_walkforward.py --round 1 --asset Forex

# All rounds
python3 dqn/backtest/backtest_dqn_walkforward.py --all --asset Forex
```

---

## ⚠️ Notes

1. **Existing models**: The 18 Forex models currently in `models/dqn_walkforward/` have unknown training parameters; retraining recommended
2. **Data consistency**: Ensure `SOURCE_OVERRIDES` in `config.py` are used for consistent data sources
3. **Interruption recovery**: Training script does not support checkpoint resume; interrupted contracts must be retrained from scratch
4. **GPU memory**: LSTM models are small, single GPU sufficient, no special configuration needed
5. **Logging**: Redirect output to log files for review

---

## 📝 Training Log Template

```bash
# Create log directory
mkdir -p logs/dqn_train

# Train Forex Round 1
python3 dqn/train/train_dqn_walkforward.py --round 1 --asset Forex --episodes 50 \
  2>&1 | tee logs/dqn_train/forex_r1_$(date +%Y%m%d_%H%M%S).log
```

---

**Next step**: Start Phase 1 (Forex full training)
