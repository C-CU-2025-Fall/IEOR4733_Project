# IEOR4733_Project — Deep Reinforcement Learning for Trading

Reproduction of **"Deep Reinforcement Learning for Trading"** by Zhang, Zohren, and Roberts (Oxford, 2019)

📄 Paper: https://arxiv.org/pdf/1911.10107

> **🤖 AI Context**: Read [`PROJECT_MEMORY.md`](./PROJECT_MEMORY.md) first for full project state, known issues, and next steps.

---

## Quick Start

```bash
pip install numpy pandas

# Run Table 3 (per-contract vol scaling only)
python baseline_run.py

# Run Table 2 (+ portfolio-level vol scaling)
python baseline_run.py --table 2

# Single asset class
python baseline_run.py --asset Commodity --all-metrics

# Validate all 50 contracts
python tests/roll_validation_final.py

# Run per-contract E(R) decomposition
python tests/decomposition_audit.py
```

---

## Methodology

### Paper Equations

**Eq 4 — Trade return per contract:**
```
R_t = A_{t-1} × (σ_tgt / σ_{t-1}) × r_t − bp × p_{t-1} × |Δscaled_pos|
```
- `r_t = p_t − p_{t-1}` (additive profits on RAD prices)
- `σ_{t-1}` = EWMA(60) std of r_t
- `A_{t-1}` = position signal (Long=+1, Sign(R), MACD)
- `bp = 0.0020` (20 bps transaction cost)

**Eq 13 — Portfolio:**
```
R_port = (1/N) × Σ R_i    (equal-weight)
```

### Table 2 vs Table 3
- **Table 3**: Per-contract vol scaling (Eq 4) only
- **Table 2**: Table 3 + portfolio-level vol scaling → target std ≈ 0.97

---

## Project Structure

```
├── baseline_run.py              # Main entry (Table 2 & 3)
├── config.py                    # Parameters + paper target values + 50 contracts
├── data_loader.py               # CLC data loading (auto RAD_v2 for damaged contracts)
├── strategies.py                # Long / Sign(R) / MACD signals
├── metrics.py                   # 9 portfolio metrics
├── vol_scaling.py               # Volatility scaling utilities
├── indicators.py                # Technical indicators
├── train_dqn_paper_aligned.py   # DQN training (future work)
├── PROJECT_MEMORY.md            # AI context pickup file
│
├── data/CLC/                    # 50 futures contracts (*_RAD.CSV, *_NON.CSV, *_REV.CSV)
├── config/TEMP/                 # ASC files for cross-validation
│
└── tests/                       # Validation & diagnostic scripts
    ├── roll_validation_final.py       # 50-contract RAD cross-validation
    ├── validate_commodity_rad.py      # 26-contract 3-check validation
    ├── test_rad_algorithm.py          # RAD_v2 generation + math proof
    ├── generate_rad_v2_validated.py   # RAD_v2 CSV generator
    ├── decomposition_audit.py         # Per-contract E(R) decomposition
    └── investigate_tc.py              # TC & σ_t diagnostic
```

---

## Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Transaction cost (bp) | 0.0020 | Paper Table 1 |
| EWMA span | 60 | Paper Section 3.2 |
| σ_tgt (per contract) | 0.063 | σ_annual=10% / √252 |
| Trading days/year | 252 | Standard |
| Sign(R) lookback | 252 | Paper Eq 10 |
| MACD pairs | (8,24),(16,48),(32,96) | Ref [4] |
| Test period | 2011-2019 | Paper Section 4.1 |

### 50 Contracts

| Asset Class | # | Tickers |
|-------------|---|---------|
| Commodity | 25 | CC,DA,GI,JO,KC,KW,LB,NR,SB,ZA,ZC,ZF,ZG,ZH,ZI,ZK,ZL,ZO,ZP,ZR,ZT,ZU,ZW,ZZ,ZN |
| Equity Index | 11 | CA,EN,ER,ES,LX,MD,SC,SP,XU,XX,YM |
| Fixed Income | 5 | DT,FB,TY,UB,US |
| Forex | 9 | AN,BN,CN,DX,FN,JN,MP,NK,SN |

> ZN = 24HR NATL GAS (Natural Gas), not 10-Year T-Note.

---

## Data Validation

### RAD Cross-Validation (2026-04-14)

**50/50 contracts validated** via deterministic roll detection:
- 27 VERIFIED (ASC cross-checked)
- 23 CROSS_VALIDATED (REV adj noise = exact 0)

### 4 RAD_v2 Contracts (damaged vendor RAD repaired)

| Contract | Issue | Fix |
|----------|-------|-----|
| ZH | Vendor RAD all-zero | RAD_v2 from REV+NON |
| ZU | Vendor RAD all-zero | RAD_v2 from REV+NON |
| US | Vendor RAD 99% NaN | RAD_v2 from REV+NON |
| ZN | Vendor RAD quarterly only | RAD_v2 from REV+NON |

**RAD_v2 algorithm**: detect rolls (REV adj_change≠0) → compute ratio=NON[t]/NON[t+1] → forward-accumulate → RAD=NON×cum_ratio. Math proof: non-roll returns match NON exactly (corr=1.000).

### CLC Data Relationships

```
NON = raw prices              RAD = NON × cum_ratio  (forward, paper uses this)
ASC = vendor roll records     REV = NON + cum_adj    (backward, for validation)
```

---

## Current Results (σ_tgt=0.063, Long, Table 3)

| Asset Class | E(R) | Paper | std(R) | Paper | Status |
|-------------|------|-------|--------|-------|--------|
| Commodity | -0.224 | -0.298 | 0.406 | 0.412 | std✅ E(R) 25% off |
| Equity Index | +0.559 | +0.504 | 0.917 | 0.928 | ✅ All <15% |
| Fixed Income | +0.519 | +0.605 | 0.927 | 0.939 | std✅ E(R) 14% off |
| Forex | -0.213 | -0.198 | 0.458 | 0.472 | ✅ All <15% |

**Key insight**: std(R) matches within 3% across all asset classes → **Eq 4 implementation is correct**.

### Known Issues

1. **MDD calculation broken** (Commodity 2.255 vs 0.248): additive wealth goes negative → MDD explodes. Need multiplicative wealth.
2. **E(R) bias** (Commodity 25%, FI 14%): likely σ_tgt selection or p0-normalization issue.
3. **Sign(R) and MACD strategies**: not yet run (commented out in baseline_run.py).

See [`PROJECT_MEMORY.md`](./PROJECT_MEMORY.md) §7 for detailed root cause analysis.

---

## TODO

- [x] 50/50 contract RAD cross-validation
- [x] 4 damaged contracts repaired (RAD_v2)
- [x] Baseline Long strategy framework
- [ ] Fix MDD calculation (multiplicative wealth)
- [ ] Fix E(R) bias (σ_tgt or p0 normalization)
- [ ] Run Sign(R) and MACD baselines
- [ ] DQN training and comparison
- [ ] Final presentation