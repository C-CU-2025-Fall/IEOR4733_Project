# PROJECT_MEMORY.md
# Last updated: 2026-04-22

Read this first when resuming work on this repo.

---

## 0. Current Truth

The repo presentation is now intentionally split into two layers:

- `README.md`
  - minimal operational documentation only
- `PROJECT_MEMORY.md`
  - all abandoned search history
  - retained upper-bound context
  - archived exploration lines

The active repo story is now:

1. **Live baseline**
- command:
  - `python baseline_run.py --table 3 --all-metrics --sigma 0.058`
- current score:
  - `<=10 25/45`
  - `<=15 31/45`

2. **Trade-world structural reference**
- command:
  - `python tests/run_structural_38.py --table 3`
- purpose:
  - current trade-world-only reference
  - excludes `MDD` / `Calmar`

3. **Experimental adjusted upper bound**
- retained only as runner + preset:
  - `python tests/run_legacy_41.py`
  - `frontier_presets.py`
- interpretation:
  - explicit experimental upper bound only
  - not promoted mainline

4. **Attribution base**
- intentionally preserved in compressed form:
  - runner:
    - `python tests/er_attribution_analysis.py`
  - machine-readable base:
    - `docs/contract_version_matrix_master.csv`

Historical search / optimization lines remain below as archive context.

---

## DQN Walk-Forward (2026-04-22)

### Current Direction

The branch has now pivoted from a per-contract walk-forward prototype to a
paper-faithful **shared-model DQN infrastructure**.

### Locked DQN spec

- shared-model training mode is now the default interpretation
- discrete action space: `{-1, 0, +1}`
- shared state schema:
  - 8 features
  - 60-step windows
- Eq.4-style additive reward with transaction cost
- architecture:
  - LSTM `[64, 32]`
  - Leaky-ReLU
  - fixed Q-targets
  - Double DQN
  - dueling DQN

### Active DQN modules

- `dqn/spec.py`
  - canonical shared DQN spec
- `dqn/pipeline.py`
  - shared state + reward construction
- `dqn/model.py`
  - shared dueling DQN model + agent
- `dqn/train/prepare_dqn_walkforward.py`
  - shared round data prep
- `dqn/train/train_dqn_walkforward.py`
  - shared round training
- `dqn/backtest/backtest_dqn_walkforward.py`
  - shared round inference / backtest

### Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Shared spec | ✅ Complete | one source of truth |
| State pipeline | ✅ Complete | one shared feature builder |
| Dueling DQN | ✅ Complete | now in retained model |
| Backtest inference | ✅ Complete | no longer falls back to `Long` |
| GPU training run | ⏳ Not started | intentionally deferred |

### Important interpretation

- the state schema is shared across all contracts and future models
- only the input time series differ by contract
- new checkpoints are expected under:
  - `dqn/models/shared_rounds/...`
- prepared data are expected under:
  - `dqn/data/shared_rounds/...`
- old per-contract `dqn/models/walkforward/` artifacts are legacy prototype outputs

### Documentation

- `dqn/README.md`
- `dqn/docs/dqn_alignment_notes.md`
### Archived comparison lines

1. **Archived same-rule candidate**
- retained only as a historical comparison point
- it is **not valid under the current no-direct-NON doctrine**
- command:
  - `python archive/tests/frontier_40plus_enumeration.py`
- current score:
  - `<=10 29/45`
  - `<=15 34/45`
- invalidating reason:
  - it contains `NR:NON`
  - it contains `ZC:NON`

3. **Experimental adjusted upper bound**
- the retained `41/45` family
- command:
  - `python archive/tests/frontier_40plus_enumeration.py`
- representative case:
  - exclusions: `FB, ZA, ZO, EN, ES`
  - Equity-only reporting: `risk_price_non`
  - numerator: `annual_mean_sleeve`
  - all mode: `contract_equal_path`
- current score:
  - `<=10 36/45`
  - `<=15 41/45`

Interpretation:
- the archived same-rule candidate is below `40+/45`
- current `41/45` is **experimental upper bound**, not promoted main line

Archived reproduction scripts are pinned to their own historical reporting settings
and should not silently inherit new `baseline_run.py` defaults. In particular:
- `tests/run_structural_38.py` is intentionally fixed to:
  - detailed tables use trade-world metrics only:
    - `E(R), std(R), DD, Sharpe, Sortino, % +ve, Ave P/L`
  - no `MDD`
  - no `Calmar`
  - `report_source="trade"`
- this keeps it as a structural-38 trade-world reproducer rather than a live-baseline reporting runner

4. **Version-matrix audit line**
- command:
  - removed from working tree during cleanup
- retained machine-readable base:
  - `docs/contract_version_matrix_master.csv`
- scope:
  - all 50 contracts
  - versions per contract
  - contexts:
    - `baseline_50`
    - `structural_38_family`
    - `legacy_41_family`
  - sigma:
    - `0.058`
    - `0.059`
    - `0.060`
  - both Table 2 and Table 3
- purpose:
  - per-contract version / exclusion rule base
  - future searches should read this matrix first, not search blindly

5. **Baseline-first optimization / upper-bound line**
- command:
  - removed from working tree during cleanup
- note:
  - conclusions retained below in memory only
- current retained conclusion:
  - strict `baseline_50`, frozen formula, `0-drop`:
    - Table 3 `29/45`
    - Table 2 `12/45`
    - version mix:
      - `47 × RAD_RAW`
      - `3 × RAD_V2`
  - global consistent upper bound:
    - Table 3 `27/45`
    - Table 2 `16/45`
    - version mix:
      - `22 × RAD_REGEN`
      - `12 × RAD_RAW`
      - `9 × REV`
      - `7 × NON_FWD_ANCHORED`
  - formula-relaxed upper bound:
    - still does **not** touch `40+/45`
- interpretation:
  - under current frozen doctrine, the available version pool is capped well below `40+/45`
  - formula relaxation alone does not fix that
  - but the matrix already reveals one clean divide-and-conquer fact:
    - `Forex` is keep-all feasible under the `% +ve / Ave P/L` gate
    - `Equity Index` and `Fixed Income` are not

---

## 1. Current Keep / Archive / Candidate Cleanup

### Keep as current core

Runtime:
- `baseline_run.py`
- `metrics.py`
- `vol_scaling.py`
- `config.py`
- `data_loader.py`

Current reference runners:
- `tests/run_structural_38.py`
- `tests/run_legacy_41.py`

Retained preset/config:
- `frontier_presets.py`

Current docs:
- `docs/data_issues.md`
- `docs/paper_table_suspicious_cells.md`
- `docs/structural38_trade_tables_paper_style_a4.png`
- `docs/reproducibility_external_search.md`

Attribution base kept on purpose:
- runner:
  - `tests/er_attribution_analysis.py`
- machine-readable base:
  - `docs/contract_version_matrix_master.csv`

### Keep as archived-but-retained

- `archive/tests/frontier_40plus_enumeration.py`
- `archive/docs/frontier_40plus_enumeration.md`
- `archive/tests/run_legacy_40.py`
- `archive/tests/table2_reporting_bridge_mode_probe.py`
- `archive/docs/table2_reporting_bridge_mode_probe.md`
- `archive/tests/equity_yf_rad_regen_probe.py`
- `archive/docs/equity_yf_rad_regen_probe.md`
- old attribution markdown outputs:
  - `archive/docs/current_baseline_attribution.md`
  - `archive/docs/er_attribution_report.md`
  - `archive/docs/equity_contract_contribution_report.md`
  - `archive/docs/all_row_contribution_report.md`

### Candidate cleanup list

Removed during cleanup; conclusions retained here only:
- `tests/historical_36x_rebuild_search.py`
- `docs/historical_36x_rebuild_search.md`
- `tests/calmar_alignment_iteration.py`
- `docs/calmar_alignment_iteration.md`
- `tests/current_baseline_search_v2.py`
- `docs/current_baseline_search_v2.md`
- `docs/current_baseline_search_v2_results.csv`
- `tests/contract_version_matrix_audit.py`
- `tests/contract_version_optimization.py`
- `docs/contract_version_matrix_audit.md`
- `docs/contract_version_optimization.md`
- `docs/contract_version_optimization_results.csv`
- `docs/contract_version_upper_bounds.csv`

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
- `archive/docs/equity_yf_rad_regen_probe.md`

### 2.3 Current reporting diagnosis

The latest retained reporting-world diagnosis is:
- **`MDD aligned, numerator wrong`**

This came after the archived Commodity cleanup loop.
The implementation/report files were removed during cleanup, but the retained
conclusion remains recorded here.

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

Important:
- those last two switches were useful in that reporting-world iteration
- but they violate the later no-direct-NON rule you imposed
- so that line is now treated as an archived diagnostic result, not an active promoted frontier

After that cleaning step:
- there is a global same-path numerator winner
- best retained candidate:
  - `annual_mean_simple`
- best same-path extraction remained:
  - `contract_equal_path + annual_mean_simple`

Important nuance:
- this improves reporting annual return alignment
- it does **not** magically produce a clean `40+/45`

Additional retained clarification:
- the project previously had a **split-world Table 2 bug**
  - trade metrics used the bridged trade-lane additive return series
  - but `MDD/Calmar` stayed on the old `RISK_PRICE_SIGMA0` reporting path
- that bug is now treated as **legacy only**
  - `baseline_run.py` now defaults Table 2 reporting to `same_as_port_contract`
  - this preserves contract / position information by pushing the portfolio bridge
    multipliers back into contract Eq.4 rewards before rebuilding the reporting path
- `split_world` is still available only as a diagnostic / legacy comparison mode
- the project still distinguishes:
  - `legacy split_world`
  - `same_as_port_contract` default runtime
  - deeper reporting / Calmar audit lines

### 2.4 Divide-and-conquer status by asset class

Using `baseline_50` only, and requiring for each contract that there exists a promotable version with:
- Table 3 `% +ve gap <= 2%`
- Table 3 `Ave P/L gap <= 2%`
- Table 2 `% +ve gap <= 2%`
- Table 2 `Ave P/L gap <= 2%`

Current status:

- **Forex**
  - all 9 contracts satisfy this keep-all test
  - therefore Forex is the cleanest asset class to isolate first
  - current issue in FX is not exclusion pressure; it is mainly:
    - Table 2 `E(R)`
    - reporting `MDD`
    - reporting `Calmar`

### 2.5 Table 2 bridge consistency

The paper and references support an **additional portfolio-level volatility scaling**
layer for Table 2, but they do **not** identify the exact bridge or justify the
local fitted target `0.97`.

Current retained implementation status:

- `vol_scaling.py` is now the single home for portfolio bridges
  - `constant_posthoc`
  - `ewma60_lagged`
  - `rolling252_lagged`
- bridge application and bridge multipliers were deduplicated so the trade lane
  and reporting lane use the **same bridge family implementation**
- Table 2 reporting default in `baseline_run.py` is now:
  - `report_bridge_mode = same_as_port_contract`
- this is a real bug fix over the previous split-world default
- there was a second bug in the first `same_as_port_contract` attempt:
  - it pushed bridge multipliers directly into additive contract rewards
  - this could drive reporting NAV negative and produce absurd `MDD > 1`
  - Commodity under Table 2 `constant_posthoc` reached `MDD ≈ 1.50`, which is not financially meaningful
- the current `same_as_port_contract` implementation was corrected to:
  - preserve contract information
  - apply bridge multipliers to sleeve simple returns
  - rebuild sleeve wealth multiplicatively
  - this removes the negative-NAV blow-up

Retained evidence from the explicit probe:
- `Forex`
  - `constant_posthoc + split_world`: `<=15 6/9`
  - `constant_posthoc + same_as_port_contract`: `<=15 6/9`
  - `rolling252_lagged + split_world`: `<=15 8/9`
  - `rolling252_lagged + same_as_port_contract`: `<=15 8/9`
- `Commodity`
  - `constant_posthoc + same_as_port_contract`: `MDD ≈ 0.849` (fixed from the broken `~1.50+`)
  - `rolling252_lagged + same_as_port_contract`: `MDD ≈ 0.847`
- across the four asset rows, the corrected `same_as_port_contract` is a *consistency fix*, not yet a higher-scoring global bridge
- all explicit bridge tables are preserved in:
  - `archive/docs/table2_reporting_bridge_mode_probe.md`

- **Equity Index**
  - no contract currently clears the keep-all distribution gate in both tables
  - the binding failure is not `% +ve`; it is mainly:
    - Table 2 `Ave P/L` at roughly `~3%`
  - deeper misses then remain in:
    - `E(R)`
    - `MDD`
    - `Calmar`

- **Fixed Income**
  - no contract currently clears the keep-all distribution gate
  - this class fails earlier and more clearly than Equity:
    - `% +ve`
    - `Ave P/L`
  - and then also misses:
    - `E(R)`
    - `MDD`
    - `Calmar`

- **Commodity**
  - still the messiest class overall
  - mixed source problems, path distortion, and structural exclusion pressure

### 2.5 Table 2 bridge reference trace and quick bridge probe

Main-paper evidence now retained:
- `references/DRL_main.pdf`, p.7:
  - “an additional layer of portfolio-level volatility scaling”
  - “This brings the volatility of different methods to a same target”
- `references/DRL_main.pdf`, p.16 / Appendix B:
  - “Table 3 presents the performance metrics for portfolios without additional layer of volatility scaling.”

Reference-chain evidence retained:
- `references/DRL_27.pdf`, p.8:
  - “portfolios with an additional layer of volatility scaling”
  - “brings overall strategy returns to match the 15% volatility target”
- `references/DRL_4.pdf`, p.12:
  - “computed the ex-ante volatility of each of the portfolios”
  - “scaled their allocations for an annualized volatility of 15% per portfolio”

Interpretation:
- the literature supports **an extra portfolio-level scaling layer**
- but it does **not** identify the exact bridge formula used in the DRL paper
- in particular, the retained repo has no direct citation supporting the specific numeric target `0.97`
- therefore `0.97` is currently treated as a fitting parameter / local bridge assumption, not a paper-grounded constant

Quick bridge probe retained at `sigma=0.058`, comparing Table 2 candidates against paper Table 2 using the current live trade lane:
- bridges tested:
  - `constant_posthoc_0.97`
  - `ewma60_lagged_0.97`
  - `rolling252_lagged_0.97`

Key result by asset class:
- `Forex`
  - `rolling252_lagged_0.97` is the best trade-side match
  - four-asset trade-style errors:
    - `E(R)` `6.4%`
    - `std(R)` `0.7%`
    - `DD` `8.9%`
    - `Sharpe` `5.7%`
    - `Sortino` `2.4%`
    - `% +ve` `0.2%`
    - `Ave P/L` `0.2%`
  - but `MDD` and especially `Calmar` still remain far off
- `Commodity`
  - `rolling252_lagged_0.97` is slightly better than `constant_posthoc_0.97` on the trade-style metrics
- `Equity Index`
  - `constant_posthoc_0.97` and `rolling252_lagged_0.97` are close; no decisive rolling advantage
- `Fixed Income`
  - `constant_posthoc_0.97` and `rolling252_lagged_0.97` are also close
- `All`
  - no tested bridge explains `All` well; `Calmar` remains unusable for bridge identification

Four-asset (`Commodity`, `Equity Index`, `Fixed Income`, `Forex`) alignment counts against paper Table 2 with `sigma=0.058`:
- `constant_posthoc_0.97`
  - `<=10`: `19/36`
  - `<=15`: `20/36`
  - mean percent error over 36 cells: `110.9%`
- `rolling252_lagged_0.97`
  - `<=10`: `19/36`
  - `<=15`: `22/36`
  - mean percent error over 36 cells: `110.3%`

Interpretation:
- `rolling252_lagged_0.97` improves the four-asset `<=15` count by `+2` versus `constant_posthoc_0.97`
- but it is **not** a universal winner across all asset classes
- the most defensible retained statement is:
  - the paper is consistent with **some** portfolio-level volatility scaling beyond Table 3
  - `Forex` most strongly suggests a time-varying bridge such as rolling scaling
  - the current evidence still does **not** uniquely identify one global Table 2 bridge

---

## 3. Session Record (Condensed)

### Session A — Historical `36/40` rebuild under cleaned doctrine

Script status:
- removed from working tree during cleanup

What it established:
- the old `36/40` skeleton still reruns under the old source structure
- once rebuilt under the cleaner source doctrine, it drops
- no source-only or source+exclusion candidate could satisfy the strict per-asset `% +ve <= 2%` and `Ave P/L <= 2%` rule

This was the formal trigger for separating the reporting / Calmar explanation line from the trade/source line.

### Session B — Reporting-world self-iteration

Script status:
- removed from working tree during cleanup

What it established:
- Commodity distortions were the main remaining contract-driven problem
- after cleaning them, the diagnosis changes from
  - `contract-driven distortion`
  to
  - `MDD aligned, numerator wrong`

This is the current best explanation line.

### Session C — Unified frontier enumeration

Script:
- `archive/tests/frontier_40plus_enumeration.py`

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
- `archive/tests/equity_yf_rad_regen_probe.py`

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
- trade-world structural reference (`tests/run_structural_38.py`)
- data issues note (`docs/data_issues.md`)
- reporting diagnosis (retained in memory):
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
2. run `tests/run_structural_38.py`
3. read `docs/data_issues.md`
4. use `tests/run_legacy_41.py` only if you explicitly want the retained experimental upper bound

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
