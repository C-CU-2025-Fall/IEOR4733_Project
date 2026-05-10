# MACD / Sign(R) Diagnostic

- Date: 2026-04-22
- Stack: current unified baseline backtest only
- Table: Table 3 only | sigma_tgt=0.058
- Sources: current `SOURCE_OVERRIDES` unless otherwise specified
- Exclusions: none

## Reference Trace

- Paper baseline section and Eq. (10)/(11)/(12): `references/DRL_journal.txt` around lines 308-326
- Current baseline strategy code: `strategies.py`
- Current shared execution path: `baseline_run.py`

## 1. Anchor Summary

| Asset | Strategy | # | E ours | E paper | ΔE | flip E | Sh ours | Sh paper | ΔSh | flip Sh | So ours | So paper | ΔSo | flip So | std err | DD err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Commodity | Long | 25 | -0.232 | -0.298 | +0.066 | N | -0.621 | -0.723 | +0.102 | N | -0.907 | -1.152 | +0.245 | N | 9.2% | 0.7% |
| Commodity | Sign(R) | 25 | -0.092 | +0.101 | -0.193 | Y | -0.327 | +0.325 | -0.652 | Y | -0.482 | +0.548 | -1.030 | Y | 9.5% | 3.6% |
| Commodity | MACD | 25 | -0.199 | -0.039 | -0.160 | N | -0.912 | -0.174 | -0.738 | N | -1.371 | -0.290 | -1.081 | N | 3.7% | 7.0% |
| Equity Index | Long | 11 | +0.526 | +0.504 | +0.022 | N | +0.627 | +0.543 | +0.084 | N | +0.798 | +0.831 | -0.033 | N | 9.6% | 8.9% |
| Equity Index | Sign(R) | 11 | +0.043 | +0.168 | -0.125 | N | +0.060 | +0.211 | -0.151 | N | +0.076 | +0.319 | -0.243 | N | 10.8% | 7.5% |
| Equity Index | MACD | 11 | -0.240 | -0.068 | -0.172 | N | -0.430 | -0.117 | -0.313 | N | -0.551 | -0.178 | -0.373 | N | 4.8% | 13.1% |
| Fixed Income | Long | 5 | +0.471 | +0.605 | -0.134 | N | +0.552 | +0.645 | -0.093 | N | +0.847 | +1.081 | -0.234 | N | 9.1% | 0.9% |
| Fixed Income | Sign(R) | 5 | -0.079 | +0.189 | -0.268 | Y | -0.107 | +0.237 | -0.344 | Y | -0.157 | +0.381 | -0.538 | Y | 7.1% | 2.0% |
| Fixed Income | MACD | 5 | -0.354 | +0.136 | -0.490 | Y | -0.597 | +0.224 | -0.821 | Y | -0.898 | +0.371 | -1.269 | Y | 2.7% | 7.3% |
| Forex | Long | 9 | -0.173 | -0.198 | +0.025 | N | -0.409 | -0.420 | +0.011 | N | -0.635 | -0.696 | +0.061 | N | 10.4% | 4.3% |
| Forex | Sign(R) | 9 | -0.401 | -0.113 | -0.288 | N | -0.802 | -0.207 | -0.595 | N | -1.141 | -0.332 | -0.809 | N | 9.2% | 3.1% |
| Forex | MACD | 9 | -0.387 | +0.016 | -0.403 | Y | -0.955 | +0.037 | -0.992 | Y | -1.347 | +0.061 | -1.408 | Y | 4.4% | 11.0% |

## 2. Eq.4 Downstream Equivalence Check

| Asset | Strategy | max abs diff (built-in vs explicit) | Eq.4 downstream identical |
|---|---|---|---|
| Commodity | Long | 0.000e+00 | Y |
| Commodity | Sign(R) | 0.000e+00 | Y |
| Commodity | MACD | 0.000e+00 | Y |
| Equity Index | Long | 0.000e+00 | Y |
| Equity Index | Sign(R) | 0.000e+00 | Y |
| Equity Index | MACD | 0.000e+00 | Y |
| Fixed Income | Long | 0.000e+00 | Y |
| Fixed Income | Sign(R) | 0.000e+00 | Y |
| Fixed Income | MACD | 0.000e+00 | Y |
| Forex | Long | 0.000e+00 | Y |
| Forex | Sign(R) | 0.000e+00 | Y |
| Forex | MACD | 0.000e+00 | Y |

Interpretation:
- If these rows are numerically zero, the divergence is entering before the shared Eq.4 execution and metric stack.

## 3. Sign(R) Interpretation Audit

### Position Equivalence: additive vs simple-return signal

| Asset | avg position disagreement |
|---|---|
| Forex | 0.0% |
| Equity Index | 0.0% |

### Variant Comparison

| Asset | Variant | E(R) | Sharpe | Sortino | focus err | flip E | flip Sh | flip So |
|---|---|---|---|---|---|---|---|---|
| Forex | current_additive | -0.401 | -0.802 | -1.141 | 159.7% | N | N | N |
| Forex | simple_current | -0.401 | -0.802 | -1.141 | 159.7% | N | N | N |
| Forex | current_additive_extra_lag | -0.378 | -0.752 | -1.073 | 146.7% | N | N | N |
| Forex | current_additive_lookahead_diag | +0.286 | +0.572 | +0.857 | 219.8% | Y | Y | Y |
| Equity Index | current_additive | +0.043 | +0.060 | +0.076 | 48.2% | N | N | N |
| Equity Index | simple_current | +0.043 | +0.060 | +0.076 | 48.2% | N | N | N |
| Equity Index | current_additive_extra_lag | +0.072 | +0.100 | +0.125 | 37.9% | N | N | N |
| Equity Index | current_additive_lookahead_diag | +0.773 | +1.089 | +1.390 | 225.7% | N | N | N |

Variants:
- `current_additive`: current production implementation
- `simple_current`: sign of 12M simple return
- `current_additive_extra_lag`: one extra lag on top of the shared Eq.4 lag
- `current_additive_lookahead_diag`: diagnostic-only left shift to test timing sensitivity

## 4. MACD Formula Audit

| Asset | Variant | E(R) | Sharpe | Sortino | focus err | flip E | flip Sh | flip So |
|---|---|---|---|---|---|---|---|---|
| Forex | current_pairwise_phi | -0.387 | -0.955 | -1.347 | 1505.3% | Y | Y | Y |
| Forex | avg_q_then_std_phi | -0.379 | -0.922 | -1.313 | 1465.1% | Y | Y | Y |
| Forex | current_no_phi | -0.611 | -0.982 | -1.295 | 1804.6% | Y | Y | Y |
| Forex | current_adjust_true_phi | -0.387 | -0.955 | -1.347 | 1505.4% | Y | Y | Y |
| Forex | current_extra_lag | -0.397 | -0.982 | -1.385 | 1544.6% | Y | Y | Y |
| Forex | current_lookahead_diag | +0.145 | +0.362 | +0.531 | 493.7% | N | N | N |
| Equity Index | current_pairwise_phi | -0.240 | -0.430 | -0.551 | 149.4% | N | N | N |
| Equity Index | avg_q_then_std_phi | -0.264 | -0.478 | -0.614 | 171.7% | N | N | N |
| Equity Index | current_no_phi | -0.313 | -0.288 | -0.321 | 165.3% | N | N | N |
| Equity Index | current_adjust_true_phi | -0.240 | -0.430 | -0.551 | 149.4% | N | N | N |
| Equity Index | current_extra_lag | -0.254 | -0.456 | -0.582 | 161.7% | N | N | N |
| Equity Index | current_lookahead_diag | +0.234 | +0.424 | +0.565 | 267.3% | Y | Y | Y |

Variants:
- `current_pairwise_phi`: current production implementation
- `avg_q_then_std_phi`: average q across pairs before the 252-day standardization
- `current_no_phi`: skip the final phi transformation
- `current_adjust_true_phi`: change EMA convention to `adjust=True`
- `current_extra_lag`: add one extra lag
- `current_lookahead_diag`: diagnostic-only lead shift

## 5. Source Sensitivity Pilot

| Asset | Strategy | # current | # RAD | E current | E RAD | ΔE RAD-current | Sh current | Sh RAD | ΔSh RAD-current | E err current | E err RAD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Forex | Sign(R) | 9 | 9 | -0.401 | -0.378 | +0.023 | -0.802 | -0.755 | +0.047 | 255.2% | 234.7% |
| Forex | MACD | 9 | 9 | -0.387 | -0.378 | +0.009 | -0.955 | -0.931 | +0.024 | 2520.2% | 2462.9% |
| Equity Index | Sign(R) | 11 | 11 | +0.043 | -0.002 | -0.045 | +0.060 | -0.003 | -0.063 | 74.6% | 101.2% |
| Equity Index | MACD | 11 | 11 | -0.240 | -0.220 | +0.020 | -0.430 | -0.393 | +0.037 | 252.6% | 223.0% |

## 6. Ranked Diagnosis

1. **Eq.4 / backtester is unlikely to be the primary culprit.**
   Built-in `Long / Sign(R) / MACD` and explicit position-provider runs are numerically identical across assets, so the large gaps are entering before the shared Eq.4 execution layer.
2. **`Sign(R)` is not primarily an additive-vs-simple-return misunderstanding.**
   On the current positive-price data, `sign(sum additive returns over 252 days)` and `sign(simple 12M return)` generate the same positions in the pilot assets, so that interpretation does not explain the gap.
3. **`Sign(R)` still looks timing-sensitive.**
   The meaningful remaining `Sign(R)` risks are signal timestamp / lag handling and data-source sensitivity, not the additive-vs-simple-return definition itself.
4. **`MACD` remains the strongest formula-mismatch candidate.**
   Across the pilot assets, the best realistic MACD alternatives are `Forex:avg_q_then_std_phi`, `Equity Index:current_pairwise_phi`, which means MACD is materially sensitive to implementation details such as standardization order, EMA convention, and lag handling.
5. **Source policy looks secondary unless a pilot asset shows a very large swing.**
   The source-sensitivity pilot compares current live overrides with pure default-RAD loading; this should be treated as a second-order explanation after the signal-definition audit.
6. **Current working classification: mixed, but signal-layer first.**
   Provisional ranking: signal-definition/timing mismatch first, source/data second, generic shared-metrics bug low probability.

## 7. Provisional Judgment

Current best classification:
- **mostly signal formula / timing mismatch, with source sensitivity as a secondary contributor**
- `Long` is much closer than `Sign(R)` / `MACD`, which weakens the generic-metrics-bug hypothesis
- `Sign(R)` additive-vs-simple-return interpretation is probably not the real issue on current positive-price data
- `MACD` remains the most likely place where paper interpretation and implementation have drifted

