# Feature 0 Z-Score Fix

## Date: 2026-04-26

## Problem
Feature 0 (Normalized close price series) was implemented as `price / rolling_std(price, 60)`, producing values with mean=110, std=42, range=[35, 242]. This was 30-100x larger than the other 6 features (all in [-4, +4] range), causing LSTM to be dominated by feature 0.

## Root Cause
Paper (ZZR2019 JFDS 2020) says "Normalized close price series" but doesn't give an explicit formula for feature 0. The normalization formula `(price - rolling_mean) / rolling_std` (z-score) is the standard interpretation matching other features' scale.

## Fix
Changed `drl_shared/state_space.py`:
```python
# Old:
feats[:, 0] = prices / (rolling_std + 1e-10)

# New (z-score):
rolling_mean = pd.Series(prices).rolling(60, min_periods=5).mean()
feats[:, 0] = (prices - rolling_mean) / (rolling_std + 1e-10)
```

## Verification
All 4 asset classes × 2 rounds verified:
- Feature 0: mean≈0, std≈1.3, range≈[-4, +4]
- All 7 features now in consistent scale

## Impact
- DQN trained with old feature 0 learned trivial "always flat" policy (100% action=1)
- With z-score, feature signal is balanced, model should learn directional positions
- All feature npz files reprepared for all asset classes

## Related
- Paper reference: Lim, Zohren, Roberts (2019) for normalization approach
- Paper only gives explicit formula for returns normalization: `r_{t-H,t} / (σ_t * √H)`
- Feature 0 formula is not specified in paper
