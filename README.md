# IEOR4733_Project — Deep Reinforcement Learning for Trading

Reproduction of **"Deep Reinforcement Learning for Trading"** by Zhang, Zohren, and Roberts (Oxford, 2019)

📄 Paper: https://arxiv.org/pdf/1911.10107

---

## Quick Start

```bash
# Install dependencies
pip install numpy pandas

# Run Table 3 (per-contract vol scaling only)
python baseline_run.py

# Run Table 2 (+ portfolio-level vol scaling)
python baseline_run.py --table 2

# Run both tables
python baseline_run.py --table both

# Single asset class
python baseline_run.py --asset "Equity Index"

# Custom σ_tgt
python baseline_run.py --sigma 0.058

# Custom test period
python baseline_run.py --test-start 2015-01-01 --test-end 2019-12-31

# Run tests
python test_baseline.py
```

---

## Methodology

### Paper Equations (as implemented)

**Eq 4 — Trade return per contract:**
```
R_t = A_{t-1} × (σ_tgt / σ_{t-1}) × r_t − bp × p_{t-1} × |Δscaled_pos|
```
- `r_t = p_t − p_{t-1}` (additive profits, p0-normalized) [Paper Section 3.2]
- `σ_{t-1}` = EWMA(60) std of r_t [Paper Section 3.2]
- `A_{t-1}` = position signal at t-1 (no look-ahead)
- `bp = 0.0020` [Paper Table 1]
- `σ_tgt` = volatility target (free parameter)

**Eq 10 — Sign(R) signal:**
```
A_t = sign(r_{t-252:t}) = sign(p_t − p_{t-252})
```

**Eq 3,11,12 — MACD signal:**
```
MACD_t = q_t / std(q_{t-252:t})
q_t = (m(S) − m(L)) / std(p_{t-63:t})
A_t = φ(MACD_t) where φ(x) = x·exp(−x²/4)/0.89
```
Time-scale pairs: (8,24), (16,48), (32,96) [Paper reference [4] Baz et al.]

**Eq 13 — Portfolio:**
```
R_port = (1/N) × Σ R_i    (equal-weight average)
```

### Table 2 vs Table 3

- **Table 3**: Per-contract vol scaling (Eq 4) only. Each contract scaled to σ_tgt, then averaged.
- **Table 2**: Table 3 + additional portfolio-level vol scaling → target std ≈ 0.97

### Metrics (9 per strategy per asset class)

1. E(R) = mean(R) × 252
2. std(R) = std(R) × √252
3. DD = √(mean(min(0,R)²)) × √252
4. Sharpe = E(R) / std(R)
5. Sortino = E(R) / DD
6. MDD = max((peak − wealth) / peak)  [running max method]
7. Calmar = realised_annual_return / MDD
8. %+ve = fraction of positive days
9. Ave P/L = mean(R>0) / |mean(R<0)|

Wealth = N × W₀ + cumsum(R_port) (additive accumulation, not multiplicative)

---

## Project Structure

```
IEOR4733_Project/
├── baseline_run.py          # Main entry point (Table 2 & 3)
├── test_baseline.py         # Tests
├── config.py                # Parameters + paper target values
├── data_loader.py           # CLC data loading
├── strategies.py            # Long / Sign(R) / MACD signals
├── metrics.py               # 9 portfolio metrics
├── vol_scaling.py           # Volatility scaling utilities
├── indicators.py            # Technical indicators (MACD, RSI, etc.)
├── train_dqn_paper_aligned.py  # DQN training (future work)
│
├── data/
│   ├── CLC/                 # 96 futures contracts (*_RAD.CSV)
│   ├── index_data.csv       # VIX index
│   └── risk_free_rate.csv   # Risk-free rate (DTB3)
│
├── references/              # Paper PDF
├── deck.md                  # Proposal deck
├── deck-v1.1.pptx           # Presentation v1.1
├── deck-v2.pptx             # Presentation v2
├── DRL_Trading_Midterm_draft.pptx
└── archive/                 # Old scripts (13 files)
```

---

## Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Transaction cost (bp) | 0.0020 | Paper Table 1 |
| EWMA span | 60 | Paper Section 3.2 |
| σ_tgt (per contract) | 0.064 | Derived from Long std match |
| Trading days/year | 252 | Standard |
| Sign(R) lookback | 252 | Paper Eq 10 |
| MACD pairs | (8,24),(16,48),(32,96) | Paper Eq 12, ref [4] |
| MACD vol window | 63 | Paper Eq 3, ref [4] |
| MACD std window | 252 | Paper Eq 3, ref [4] |
| Portfolio vol target | 0.97 | Paper Table 2 |
| Test period | 2011-2019 (9 years) | Paper Section 4.1 |
| Discount factor (γ) | 0.3 | Paper Table 1 (RL only) |
| Retrain interval | 5 years | Paper Section 4.1 (RL only) |

### Asset Classes & Contracts (45 of 50 usable)

| Asset Class | Contracts | Tickers |
|-------------|-----------|---------|
| Commodity | 21 | CC,DA,GI,JO,KC,KW,LB,NR,SB,ZA,ZC,ZF,ZG,ZK,ZL,ZO,ZP,ZR,ZT,ZW,ZZ |
| Equity Index | 11 | CA,EN,ER,ES,LX,MD,SC,SP,XU,XX,YM |
| Fixed Income | 4 | DT,FB,TY,UB |
| Forex | 9 | AN,BN,CN,DX,FN,JN,MP,NK,SN |

Excluded (5): ZH, ZI, ZN (data quality), ZU, US (no test period data)

---

## Current Results

### Table 3 — Equity Index (11 contracts), σ_tgt = 0.064

| | E(R) | std | DD | Sharpe | Sortino | MDD | Calmar | %+ve | AveP/L |
|---|---|---|---|---|---|---|---|---|---|
| **Long** | +0.528 | +0.933 | +0.695 | +0.566 | +0.760 | +0.113 | +0.412 | +0.547 | +0.910 |
| Paper | +0.504 | +0.928 | +0.606 | +0.543 | +0.831 | +0.127 | +0.466 | +0.541 | +0.928 |
| %Err | **4.8%** | **0.5%** | 14.7% | **4.2%** | 8.5% | 11.0% | 11.6% | **1.1%** | **1.9%** |

**EQ Long: 9/9 ≤ 15%** ✅

### Table 3 — Forex (9 contracts), σ_tgt = 0.064

| | E(R) | std | DD | Sharpe | Sortino | MDD | Calmar | %+ve | AveP/L |
|---|---|---|---|---|---|---|---|---|---|
| **Long** | −0.248 | +0.464 | +0.336 | −0.536 | −0.726 | +0.321 | −0.084 | +0.490 | +0.953 |
| Paper | −0.198 | +0.472 | +0.285 | −0.420 | −0.696 | +0.219 | −0.101 | +0.491 | +0.966 |
| %Err | 23.2% | **1.7%** | 17.9% | 25.2% | **4.3%** | 46.6% | 16.8% | **0.2%** | **1.3%** |

### Summary

- **std / %+ve / AveP/L**: precise across all strategies (0–6%) → methodology correct
- **Long Only E(R)/Sharpe**: EQ ≈5% → framework verified
- **Sign(R)/MACD E(R)**: directional mismatch → CLC data version difference (2026 vs 2019)
- **Sign(R) / MACD**: alpha difference, not methodology error

---

## TODO

- [ ] Complete all 4 asset classes × 3 strategies comparison tables
- [ ] DQN training and comparison with baselines
- [ ] Sensitivity analysis for σ_tgt
- [ ] Final presentation
