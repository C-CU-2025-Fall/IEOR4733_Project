# IEOR4733 DQN Project Changelog

## 2026-05-04 Pipeline Alignment & Reward Fix

### What was done
1. **V1 rollback complete**: 9D features + V1 additive reward, stash@{0} backs up V2/V3
2. **Reward timing bug fixed**: ContractEnv.step() now uses self.last_position (paper's A_{t-1})
3. Cleaned up duplicate checkpoints (074328, 12D model)
4. Created `docs/DQN_PIPELINE.md` architecture document
5. Rewrote `scripts/alpha_decay_analysis.py`, using baseline_run reward

### Data pipeline conclusions
- baseline_run and training use the **same** load_clc_full, prices are identical
- Different lengths due to different start_date (2009 vs 2004), not data source differences
- **Previously misreported** "return correlation 0.03" — purely an index alignment error, not a real difference
- On-the-fly features and npz features are mathematically identical in test period

### Current model status
- R1 (074157): Sharpe +0.10 (baseline_run reward), driven by 2014 single year
- R2 (074703): Sharpe -1.26, overall losses
- **Both models were trained under incorrect reward (1-day lookahead)**
- Need to retrain with corrected reward

### Lessons learned
1. **Read documentation before diagnosing** — don't start debugging based on assumptions
2. **Verify index alignment** — confirm both arrays correspond to the same dates before comparing
3. **Don't stack unverified conclusions** — one wrong conclusion pollutes all subsequent reasoning
4. **False alarms are more dangerous than bugs** — waste time, break trust

### TODO
- [ ] Retrain Forex R1/R2 with corrected reward
- [ ] Verify training/backtest reward fully aligned after fix
- [ ] Analyze why model has Long=0% (Hold reward=0 bias)
- [ ] Regenerate features for other asset classes (still 12D or old date range)
