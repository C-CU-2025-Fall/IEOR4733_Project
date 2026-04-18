# PROJECT_MEMORY.md
# Last updated: 2026-04-18

Read this first when resuming work on this repo.

---

## 0. Current Truth

The repo is now intentionally condensed around two preserved version lines and one explanation line:

1. **Live baseline**
- keeps all 50 contracts, including all Equity / Forex contracts
- command:
  - `python baseline_run.py --table 3 --all-metrics --sigma 0.058`
- current score:
  - `<=10 25/45`
  - `<=15 31/45`

2. **Clean same-rule ceiling**
- best clean interpretation under current source doctrine
- command:
  - `python tests/frontier_40plus_enumeration.py`
- current score:
  - `<=10 29/45`
  - `<=15 34/45`

3. **Experimental adjusted upper bound**
- the retained `41/45` family
- command:
  - `python tests/frontier_40plus_enumeration.py`
- representative case:
  - exclusions: `FB, ZA, ZO, EN, ES`
  - Equity-only reporting: `risk_price_non`
  - numerator: `annual_mean_sleeve`
  - all mode: `contract_equal_path`
- current score:
  - `<=10 36/45`
  - `<=15 41/45`

Interpretation:
- clean same-rule is still below `40+/45`
- current `41/45` is **experimental upper bound**, not promoted main line

---

## 1. Core Files Retained

Retained scripts:
- `tests/historical_36x_rebuild_search.py`
- `tests/calmar_alignment_iteration.py`
- `tests/frontier_40plus_enumeration.py`
- `tests/equity_yf_rad_regen_probe.py`

Retained reports:
- `docs/historical_36x_rebuild_search.md`
- `docs/calmar_alignment_iteration.md`
- `docs/frontier_40plus_enumeration.md`
- `docs/equity_yf_rad_regen_probe.md`

Everything else from the last exploration branch that was only intermediate has been deleted after the conclusions were merged here.

---

## 2. Data / Formula Conclusions

### 2.1 Negative-price `REV` is not economically safe as an active source

Under the paper-aligned Eq. 4 implementation:

\[
R_t = A_{t-1}\frac{\sigma_{tgt}}{\sigma_{t-1}}r_t - bp \cdot p_{t-1}\cdot |\Delta scaled\_pos|
\]

If `p_{t-1}` is negative:
- transaction cost becomes economically invalid
- reporting capital anchor also becomes invalid

Therefore the repo no longer treats negative-price-sensitive `REV` series as clean active search candidates.

Problem contracts:
- `CC`
- `LB`
- `JO`
- `ZH`
- `ZO`

Allowed-source policy currently retained:
- `CC`: `RAD_REGEN`, `NON_FWD_ANCHORED`
- `LB`: `RAD`, `NON_FWD_ANCHORED`, `RAD_REGEN`
- `JO`: `RAD_REGEN`
- `ZH`: `RAD_REGEN`
- `ZO`: `RAD`, `RAD_REGEN`, `NON_FWD_ANCHORED`

`REV` on these is reference-only.

### 2.2 Yahoo Finance behaves like `NON`

Retained conclusion:
- Yahoo ≈ `CLC NON`
- Yahoo is not a substitute for `RAD`
- Yahoo is not a substitute for negative-price `REV`

For the recent Equity probe:
- `ES ↔ ES=F`
- `EN ↔ NQ=F`

And even after building Yahoo-based `YF_RAD_REGEN` proxies, putting `EN/ES` back still does not recover `40+/45`.

See:
- `docs/equity_yf_rad_regen_probe.md`

### 2.3 Current reporting diagnosis

The latest retained reporting-world diagnosis is:
- **`MDD aligned, numerator wrong`**

This came after the Commodity cleanup loop in:
- `tests/calmar_alignment_iteration.py`

That loop established:
- dominant Commodity distortion contracts:
  - `SB`
  - `KC`
  - `ZL`
  - `NR`
  - `ZC`
- best cleanup move in the frozen framework:
  - drop `SB`
  - drop `KC`
  - drop `ZL`
  - `NR: RAD -> NON`
  - `ZC: RAD -> NON`

After that cleaning step:
- there is a global same-path numerator winner
- best retained candidate:
  - `annual_mean_simple`
- best same-path extraction remained:
  - `contract_equal_path + annual_mean_simple`

Important nuance:
- this improves reporting annual return alignment
- it does **not** magically produce a clean `40+/45`

---

## 3. Session Record (Condensed)

### Session A — Historical `36/40` rebuild under cleaned doctrine

Script:
- `tests/historical_36x_rebuild_search.py`

What it established:
- the old `36/40` skeleton still reruns under the old source structure
- once rebuilt under the cleaner source doctrine, it drops
- no source-only or source+exclusion candidate could satisfy the strict per-asset `% +ve <= 2%` and `Ave P/L <= 2%` rule

This was the formal trigger for separating the reporting / Calmar explanation line from the trade/source line.

### Session B — Reporting-world self-iteration

Script:
- `tests/calmar_alignment_iteration.py`

What it established:
- Commodity distortions were the main remaining contract-driven problem
- after cleaning them, the diagnosis changes from
  - `contract-driven distortion`
  to
  - `MDD aligned, numerator wrong`

This is the current best explanation line.

### Session C — Unified frontier enumeration

Script:
- `tests/frontier_40plus_enumeration.py`

What it established:
- clean same-rule max: `34/45`
- under current cleaner doctrine, coherent override / structural-heavy top out at `38/45`
- only legacy experimental upper-bound family reaches `40+`
- representative current upper bound reaches `41/45`

This is the retained scorecard separating:
- clean main interpretation
- experimental upper bound

### Session D — Equity Yahoo probe

Script:
- `tests/equity_yf_rad_regen_probe.py`

What it established:
- Yahoo for `ES` and `EN` is almost exactly `NON`
- replacing `EN/ES` with Yahoo-based `YF_NON` or `YF_RAD_REGEN` does **not** recover the lost score when they are put back

Key result:
- legacy upper bound with `EN/ES` excluded:
  - `41/45`
- put `EN/ES` back with current CLC:
  - `38/45`
- put `EN/ES` back with `YF_NON`:
  - `38/45`
- put `EN/ES` back with `YF_RAD_REGEN`:
  - `37/45`

Conclusion:
- Yahoo does not solve the Equity re-entry problem

---

## 4. What To Trust

### Trust as main reference
- live baseline (`31/45`)
- clean same-rule max (`34/45`)
- reporting diagnosis:
  - `MDD aligned, numerator wrong`

### Trust as experimental upper bound only
- legacy `41/45` family

It is useful for review and comparison, but do **not** silently treat it as the promoted model because it depends on:
- `EN/ES` exclusion
- Equity-only reporting `risk_price_non`

---

## 5. Current Recommendation

If you resume later:

1. run the live baseline
2. run `frontier_40plus_enumeration.py`
3. read `docs/frontier_40plus_enumeration.md`
4. only after that, use `calmar_alignment_iteration.py` if you want to continue the reporting-world explanation line

If the goal is:
- **clean interpretation**: stay on the same-rule line
- **upper-bound score**: use the experimental `41/45` family explicitly labeled as such

---

## 6. One-Sentence Summary

The repo currently supports:
- one safe all-contract baseline,
- one clean-but-sub-40 same-rule interpretation,
- and one explicit experimental `41/45` upper bound;
the main unresolved issue is no longer `MDD`, but the reporting annual-return / `Calmar` definition and Equity-sensitive upper-bound structure.
