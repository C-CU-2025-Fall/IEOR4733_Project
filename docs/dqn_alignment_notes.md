# DQN Alignment Notes

## Two Key Concepts

### 1️⃣ Implementation Alignment
**Goal**: 100% reproduce the paper's methodology, even if the paper itself has flaws

**Current Status**:
- ✅ Hyperparameters (Table 1): LR=0.0001, γ=0.3, batch=64, LSTM[64,32]
- ✅ Training period: 2005-2010 (6 years, paper uses 10+ years but data limited)
- ✅ Test period: 2011-2019 (9 years, consistent with paper)
- ✅ Dual Q-networks (Fixed Q-targets)
- ✅ Double DQN
- ✅ Gradient Clipping (0.5)
- ✅ LSTM + Leaky-ReLU
- ✅ State space 8D (multi-scale MACD fixed)
- ⚠️ Action space: discrete {-1,0,+1} → need to confirm if paper uses discrete or continuous
- ⚠️ Reward function: need to confirm if there's risk-adjustment

**Items to confirm**:
1. Does the paper's DQN use discrete or continuous actions?
2. Does the paper's reward function have a risk-adjustment term?
3. Does the paper use walk-forward validation?

### 2️⃣ Paper Limitations
**Goal**: Identify potential flaws in the paper's methodology as future improvement directions

**Potential issues** (do not affect alignment):
- Only 10 years of training data, may overfit
- 50 contracts trained independently, ignoring inter-contract relationships
- Discrete action space may limit strategy expressiveness
- No risk-adjusted reward, may lead to excessive risk-taking
- No walk-forward validation, time-varying performance unknown

---

## Alignment Plan (focus on 1️⃣)

### ✅ Step 1: Paper Details Confirmation (COMPLETED)
- ✅ **Action Space**: discrete {-1, 0, +1} (output_dim=3, argmax selection)
- ✅ **Reward function**: `reward = (action × vol_scale × return) - cost` (no risk-adjustment)
- ✅ **Training period**: paper uses 2011-2015 (5 years), we use 2005-2010 (6 years, data limited)
- ✅ **Test period**: 2011-2019 (9 years, consistent with paper Table 3)
- ✅ **Network**: LSTM[64,32] + Leaky-ReLU + output_dim=3
- ✅ **Double DQN**: main network selects action, target network computes Q-value
- ✅ **Fixed Q-targets**: Target Network updated every 1000 steps

### ⚠️ Step 2: Current Implementation Differences
- ✅ **Action space**: discrete {-1,0,+1} → **aligned**
- ✅ **Reward**: simple reward → **aligned**
- ✅ **State Space**: 8D (with multi-scale MACD) → **aligned**
- ⚠️ **Training data**: 2005-2010 (6 years) vs paper 2011-2015 (5 years) → **different data window but similar length**
- ⚠️ **Inference output**: currently uses softmax to convert to continuous position → **should change to discrete action mapping**

### Step 3: Items to Fix
- [ ] **Inference action mapping**: discrete action → position (handled in baseline_run)
- [ ] **Validate alignment**: single-contract training, compare with original code results
- [ ] **Record differences**: if differences from paper Table 3 DQN results, record and analyze

### Step 4: Training Plan
- [ ] Retrain 50 contracts (100 episodes, 2005-2010 data)
- [ ] Test 2011-2019 alignment
- [ ] Compare against Long Only baseline

---

**Last Updated**: 2026-04-22
