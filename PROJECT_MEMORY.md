# PROJECT_MEMORY.md — AI Context Pickup File
# > Last updated: 2026-04-14
# > Read this FIRST when starting a new session on this project

---

## 1. Project Goal

Reproduce **"Deep Reinforcement Learning for Trading"** (Zhang, Zohren, Roberts, 2019)
- Paper: https://arxiv.org/pdf/1911.10107
- 50 futures contracts, 4 asset classes, test period 2011-2019
- Baselines: Long, Sign(R), MACD → compare with DQN

## 2. Core Equations

```
Eq 4: R_t = A_{t-1} × (σ_tgt / σ_{t-1}) × r_t  −  bp × p_{t-1} × |Δscaled_pos|
         where r_t = p_t − p_{t-1} (additive), σ_t = EWMA(60) std of r_t

Eq 13: R_port = (1/N) × Σ R_i  (equal-weight portfolio)

Metrics: E(R), std(R), DD, Sharpe, Sortino, MDD, Calmar, % +ve, Ave P/L
```

Parameters: bp=0.002, EWMA=60, σ_tgt≈0.063, T=252, Sign lookback=252

## 3. Data Architecture

```
CLC data files (data/CLC/):
  NON = raw prices (no roll info)
  RAD = NON × cumulative_ratio  ← paper uses this for backtesting
  REV = NON + cumulative_adj    ← used for roll detection & validation
  ASC = vendor roll records     ← gold standard (27/50 contracts have ASC)

  RAD_v2 = regenerated RAD for 4 damaged contracts (ZH, ZU, US, ZN)
           Algorithm: detect rolls from REV adj_change≠0, compute ratio=NON[t]/NON[t+1],
           forward-accumulate, RAD = NON × cum_ratio
```

## 4. Data Validation Status (2026-04-14)

**50/50 contracts validated:**
- 27 VERIFIED (ASC cross-checked, price error <1%)
- 23 CROSS_VALIDATED (REV cross-checked, adj noise = exact 0)
- 26/26 commodity + US fixed income pass 3-check validation (validate_commodity_rad.py)

**4 RAD_v2 contracts (ZH, ZU, US, ZN):**
- Math proof: non-roll returns match NON exactly (corr=1.000, ratio=1.000)
- Roll-day continuity: MaxJump=0.000% (by construction)
- All CROSS_VALIDATED against REV (adj noise = exact 0)

| Contract | Original Issue | Rolls Detected | NR_Corr | NR_Ratio | MaxJump | Validation |
|----------|---------------|----------------|---------|----------|---------|------------|
| ZH | Vendor RAD all-zero | 106 | 1.0000 | 1.0000 | 0.000% | ✅ CROSS_VALIDATED |
| ZU | Vendor RAD all-zero | 106 | 1.0000 | 1.0000 | 0.000% | ✅ CROSS_VALIDATED |
| US | Vendor RAD 99% NaN | 36 | 1.0000 | 1.0000 | 0.000% | ✅ CROSS_VALIDATED |
| ZN | Vendor RAD quarterly only | 104 | 1.0000 | 1.0000 | 0.000% | ✅ CROSS_VALIDATED |

## 5. Key Files

| File | Purpose |
|------|---------|
| `baseline_run.py` | Main entry: Table 2 & 3 reproduction |
| `config.py` | Parameters, paper target values, 50 contract definitions |
| `data_loader.py` | CLC data loading (auto-selects RAD_v2 for damaged contracts) |
| `strategies.py` | Long / Sign(R) / MACD signal generators |
| `metrics.py` | 9 portfolio metrics (additive framework) |
| `vol_scaling.py` | Volatility scaling utilities |
| `indicators.py` | Technical indicators |
| `train_dqn_paper_aligned.py` | DQN training (future) |

**Core test files (keep):**
| File | Purpose |
|------|---------|
| `tests/roll_validation_final.py` | 50-contract RAD cross-validation |
| `tests/validate_commodity_rad.py` | 26-contract 3-check validation |
| `tests/test_rad_algorithm.py` | RAD_v2 generation + math proof |
| `tests/decomposition_audit.py` | Per-contract E(R) = signal − TC decomposition |
| `tests/investigate_tc.py` | TC & σ_t diagnostic |
| `tests/generate_rad_v2_validated.py` | RAD_v2 CSV generator |

## 6. Current Baseline Results (σ_tgt=0.063, Long only)

| Asset Class | E(R) Ours/Paper | std(R) Ours/Paper | MDD Ours/Paper | Key Issue |
|-------------|-----------------|-------------------|----------------|-----------|
| Commodity | -0.224/-0.298 (25%) | 0.406/0.412 (1.5%) | **2.255/0.248** | MDD broken |
| Equity Index | +0.559/+0.504 (11%) | 0.917/0.928 (1.2%) | OK | Good |
| Fixed Income | +0.519/+0.605 (14%) | 0.927/0.939 (1.3%) | **0.427/0.108** | MDD broken |
| Forex | -0.213/-0.198 (8%) | 0.458/0.472 (3.0%) | OK | Good |

## 7. Known Issues & Root Causes

### Issue 1: MDD exploding (Commodity 2.255 vs 0.248, FI 0.427 vs 0.108)
- **Root cause**: `metrics.py` line 70-73 uses additive wealth = W₀ + cumsum(R)
- When cumsum(R) < -W₀, wealth goes negative → (peak-wealth)/peak explodes
- Commodity: 1281/2266 days have negative wealth
- **Likely paper uses**: multiplicative wealth = cumprod(1+R_port/W₀), or MDD = max(cumsum drawdown) / std(R)_annual
- **Fix needed**: investigate paper's MDD definition, likely multiplicative

### Issue 2: E(R) bias (Commodity 25%, FI 14%)
- std(R) matches within 1.5% → Eq 4 implementation is correct
- E(R) = E(signal) − E(TC): signal return differs from paper
- **Possible causes**:
  - σ_tgt value (paper may use different value per asset class)
  - Paper may use p0-normalized prices (we skip p0 norm)
  - CLC data version differences
- ZN prices are tiny (0.002-0.022) due to ratio adjustment → very small position scaling

### Issue 3: σ_tgt selection
- Current: σ_tgt=0.063 for all contracts
- For high-price commodities (CC~2542, KW~1720, ZI~2823): σ_t is huge → sp≈0.01 → almost no position
- Paper may use different σ_tgt per asset class or p0 normalization
- decomposition_audit.py shows σ/P ratio varies 0.12-0.30 across commodities

## 8. Next Steps

1. **Fix MDD calculation** — likely switch to multiplicative wealth or different normalization
2. **Investigate σ_tgt** — try per-asset-class σ_tgt or p0-normalization to fix E(R) bias
3. **Run Sign(R) and MACD strategies** — currently commented out in baseline_run.py line 170
4. **DQN training** — train_dqn_paper_aligned.py exists but needs work
5. **Final presentation** — deck-v2.pptx

## 9. Quick Commands

```bash
# Validate all 50 contracts
python tests/roll_validation_final.py

# Validate 26 commodity + US contracts (3-check)
python tests/validate_commodity_rad.py

# Run baseline (Table 3)
python baseline_run.py

# Run decomposition audit
python tests/decomposition_audit.py

# Run single asset class
python baseline_run.py --asset Commodity --all-metrics