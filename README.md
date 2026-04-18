# IEOR4733_Project — Deep Reinforcement Learning for Trading

Reproduction of Zhang, Zohren, Roberts (2019), with the repo condensed to the current core:

- one **live baseline** that keeps all Equity / Forex contracts
- one **experimental adjusted upper bound** that reaches `41/45`
- one **reporting-world audit line** explaining why clean same-rule still stalls below `40+/45`

Paper: [arXiv PDF](https://arxiv.org/pdf/1911.10107)

> Read [PROJECT_MEMORY.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/PROJECT_MEMORY.md) first if you are resuming work.

## Quick Start

```bash
pip install numpy pandas yfinance

# Live baseline: all 50 contracts included
python baseline_run.py --table 3 --all-metrics --sigma 0.058

# Historical baseline rebuild under the cleaned source doctrine
python tests/historical_36x_rebuild_search.py

# Self-iterating reporting / Calmar alignment audit
python tests/calmar_alignment_iteration.py

# Enumerate clean same-rule vs 40+ experimental frontiers
python tests/frontier_40plus_enumeration.py

# Probe whether Yahoo-based ES/EN paths help when putting Equity back
python tests/equity_yf_rad_regen_probe.py
```

## Preserved Versions

### 1. Live Baseline

This is the baseline version that **keeps all Equity / Forex contracts** and remains the default runtime:

```bash
python baseline_run.py --table 3 --all-metrics --sigma 0.058
```

Current score:
- `<=10: 25/45`
- `<=15: 31/45`

Properties:
- exclusions: none
- reporting bridge: `RISK_PRICE_SIGMA0`
- default reporting numerator in the live CLI is still `wealth_cagr`
- this is the safest “all contracts retained” reference point

### 2. Clean Same-Rule Search Ceiling

This is the best retained **clean interpretation** line:

```bash
python tests/frontier_40plus_enumeration.py
```

Current clean same-rule max:
- `<=10: 29/45`
- `<=15: 34/45`

Interpretation:
- one global reporting rule
- one global numerator
- no asset-specific reporting override
- no negative-price-sensitive `REV` comeback

### 3. Experimental Adjusted Upper Bound

This is the retained **score-first adjusted version** that currently reaches `41/45`:

```bash
python tests/frontier_40plus_enumeration.py
```

Representative `41/45` case:
- family: `legacy experimental upper bound`
- exclusions: `FB, ZA, ZO, EN, ES`
- Equity-only reporting: `risk_price_non`
- reporting extraction:
  - `annual_mean_sleeve`, or
  - `wealth_cagr`
- aggregation: `contract_equal_path`

Current score:
- `<=10: 36/45`
- `<=15: 41/45`

This version is intentionally preserved as:
- **experimental upper bound**
- not the promoted main interpretation

## Core Data Problems

These are the current repo-level conclusions after all retained sessions.

### 1. Negative-price `REV` cannot be treated as an active source in Eq. 4

Eq. 4 uses raw price level in the transaction-cost term:

\[
R_t = A_{t-1}\frac{\sigma_{tgt}}{\sigma_{t-1}}r_t - bp \cdot p_{t-1}\cdot |\Delta scaled\_pos|
\]

So if `p_{t-1} < 0`:
- transaction cost becomes economically invalid
- reporting capital anchor also becomes invalid

This is why the repo now treats these contracts cautiously:
- `CC`
- `LB`
- `JO`
- `ZH`
- `ZO`

### 2. Yahoo Finance behaves like `NON`, not like adjusted continuous prices

The retained local Yahoo probes show:
- Yahoo ≈ `CLC NON`
- Yahoo is **not** a replacement for `RAD`
- Yahoo is **not** a replacement for negative-price `REV`

For `ES/EN`, local Yahoo mapping is:
- `ES ↔ ES=F`
- `EN ↔ NQ=F`

And even after building Yahoo-based `YF_RAD_REGEN` paths, putting `EN/ES` back still does **not** recover `40+/45`.

Reference:
- [docs/equity_yf_rad_regen_probe.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/docs/equity_yf_rad_regen_probe.md)

### 3. Clean same-rule still stalls below `40+/45`

After the retained reporting-world iteration:
- the strongest clean same-rule line is still below `40+/45`
- the current clean ceiling is `34/45`
- the current live baseline is `31/45`

So at the moment:
- **there is no clean same-rule 40+ frontier in this repo**
- the only reproducible `40+` cases are experimental upper-bound families

### 4. Reporting diagnosis: `MDD aligned, numerator wrong`

The retained Calmar alignment loop concluded:
- `MDD` is relatively close after Commodity cleanup
- the main remaining mismatch is the **reporting annual return numerator**
- the best same-path winner in the retained audit is:
  - `annual_mean_simple`

Reference:
- [docs/calmar_alignment_iteration.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/docs/calmar_alignment_iteration.md)

## Retained Core Files

The repo was condensed. The key retained exploration files are:

- [tests/historical_36x_rebuild_search.py](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/tests/historical_36x_rebuild_search.py)
- [tests/calmar_alignment_iteration.py](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/tests/calmar_alignment_iteration.py)
- [tests/frontier_40plus_enumeration.py](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/tests/frontier_40plus_enumeration.py)
- [tests/equity_yf_rad_regen_probe.py](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/tests/equity_yf_rad_regen_probe.py)

And their retained reports:

- [docs/historical_36x_rebuild_search.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/docs/historical_36x_rebuild_search.md)
- [docs/calmar_alignment_iteration.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/docs/calmar_alignment_iteration.md)
- [docs/frontier_40plus_enumeration.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/docs/frontier_40plus_enumeration.md)
- [docs/equity_yf_rad_regen_probe.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/docs/equity_yf_rad_regen_probe.md)

Older one-off search artifacts were removed after their conclusions were merged into [PROJECT_MEMORY.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/PROJECT_MEMORY.md).

## Minimal Working Interpretation

- **Trade world** owns:
  - `E(R), std(R), DD, Sharpe, Sortino, % +ve, Ave P/L`
- **Reporting world** owns:
  - `MDD, Calmar`
- current reporting bridge:
  - `RISK_PRICE_SIGMA0`
- current clean reporting takeaway:
  - `MDD` is usable
  - `Calmar` is still definition-sensitive
  - `annual_mean_simple` is the best retained same-path numerator candidate

## Current Recommendation

Use the repo in this order:

1. live baseline for “all contracts retained” reference
2. clean same-rule frontier for interpretable search ceiling
3. experimental `41/45` frontier only as upper bound

If you need to continue research later, start from:
- [PROJECT_MEMORY.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/PROJECT_MEMORY.md)
- then [docs/frontier_40plus_enumeration.md](/Users/gecong/LocalFiles/GitHub/IEOR4733_Project/docs/frontier_40plus_enumeration.md)
