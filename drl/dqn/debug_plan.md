# DQN Non-Equity Performance Debug Plan

## Problem Statement
DQN shows inconsistent performance across non-equity contracts:
- **Good**: ZH (+1.58), ZN (+2.48), ZU (+1.01) — positive reward
- **Bad**: US (-2.34), KC (stuck) — negative reward
- **Question**: Is this a DQN architecture issue or a reward signal issue?

## Key Finding: TC/Gross Ratio

| Contract | TC/Gross | DQN Result |
|----------|----------|------------|
| ZH       | 1.54x    | ✅ +1.58   |
| ZN       | 1.58x    | ✅ +2.48   |
| GI       | 2.10x    | ✅ ~0      |
| ER       | 3.94x    | ❓         |
| ES       | 4.60x    | ❓         |
| JN       | 9.37x    | ❌         |
| US       | 17.15x   | ❌ -2.34   |
| KC       | 34.27x   | ❌         |

**Strong correlation**: TC/Gross > 5x → DQN fails

## Root Cause Analysis

### 1. Reward Signal Quality
The reward is: `r = gross - tc`
- `gross = action * (σ_tgt/σ) * return`
- `tc = bp * price * |position_change|`

For high-price, low-sigma contracts (US, KC):
- TC dominates gross by 17-34x
- Signal-to-noise ratio is terrible
- DQN cannot distinguish profitable actions from unprofitable ones

### 2. Gamma = 0.3
- Effective horizon: ~6 steps
- For contracts with high TC, the agent needs to hold positions longer to recoup costs
- Gamma=0.3 makes the agent myopic — it can't plan for long-term payoff

### 3. Discrete Actions
- Only 3 actions: {-1, 0, +1}
- For high-TC contracts, frequent position changes are penalized heavily
- Agent needs fine-grained position sizing, not binary decisions

## Debug Plan

### Phase 1: Verify Signal Quality (Immediate)
- [ ] Compute TC/Gross for all 50 contracts
- [ ] Correlate TC/Gross with DQN final reward
- [ ] Identify threshold where DQN fails

### Phase 2: Reward Reformulation (Medium)
- [ ] Test normalized reward: `r_norm = (gross - tc) / (bp * price * sigma_tgt)`
- [ ] Test return-based reward: `r = action * return - tc / price`
- [ ] Compare DQN performance with different reward formulations

### Phase 3: Architecture Changes (Long)
- [ ] Test continuous action space (position sizing)
- [ ] Test higher gamma (0.9) for long-horizon planning
- [ ] Test reward shaping: add position change penalty

## Next Steps for Proof
To prove DQN can work on non-equity:
1. **Pick a good contract** (ZH or ZN) — already working
2. **Pick a bad contract** (US or KC) — need reward reformulation
3. **Test normalized reward** on bad contract
4. **Compare** with original reward

## Files to Modify
- `drl_shared/state_space.py` — `compute_eq4_reward()` for reward reformulation
- `drl/dqn/spec.py` — GAMMA, DISCRETE_ACTION_VALUES
- `drl/dqn/model.py` — action space if needed
