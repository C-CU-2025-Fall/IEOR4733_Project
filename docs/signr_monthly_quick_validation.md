# Sign(R) Monthly Quick Validation

- Date: 2026-04-22
- Goal: test whether current daily `252d sign` is materially different from a monthly `12m signal + next-month hold` interpretation
- Evaluation stack: current unified baseline backtest only
- sigma_tgt = 0.058

Reference used:
- `DRL_37.pdf` (Moskowitz, Ooi, Pedersen 2012) describes a canonical `12-month lookback + 1-month holding` TSMOM strategy with inverse-volatility sizing.
- This is not the same object as a daily re-evaluated `sign(p_t - p_{t-252})` signal unless the two constructions happen to align in practice.

## Variant Comparison

| Asset | Variant | # | E(R) | Sharpe | Sortino | std err | DD err | focus err |
|---|---|---|---|---|---|---|---|---|
| Forex | current_daily_252d | 9 | -0.401 | -0.802 | -1.141 | 9.2% | 3.1% | 159.7% |
| Forex | monthly_12m_next_month_hold | 9 | -0.269 | -0.528 | -0.751 | 7.6% | 5.0% | 86.4% |
| Forex | monthly_12m_same_month_diag | 9 | +0.457 | +0.901 | +1.320 | 8.0% | 1.4% | 309.3% |
| Equity Index | current_daily_252d | 11 | +0.043 | +0.060 | +0.076 | 10.8% | 7.5% | 48.2% |
| Equity Index | monthly_12m_next_month_hold | 11 | +0.190 | +0.261 | +0.318 | 9.0% | 13.6% | 12.0% |
| Equity Index | monthly_12m_same_month_diag | 11 | +0.780 | +1.118 | +1.408 | 12.7% | 5.3% | 230.6% |
| Fixed Income | current_daily_252d | 5 | -0.079 | -0.107 | -0.157 | 7.1% | 2.0% | 87.5% |
| Fixed Income | monthly_12m_next_month_hold | 5 | +0.057 | +0.078 | +0.112 | 8.2% | 2.3% | 43.6% |
| Fixed Income | monthly_12m_same_month_diag | 5 | +0.501 | +0.687 | +1.005 | 8.2% | 0.5% | 105.5% |
| Commodity | current_daily_252d | 25 | -0.092 | -0.327 | -0.482 | 9.5% | 3.6% | 118.7% |
| Commodity | monthly_12m_next_month_hold | 25 | -0.093 | -0.321 | -0.461 | 7.3% | 8.6% | 118.1% |
| Commodity | monthly_12m_same_month_diag | 25 | +0.713 | +2.490 | +3.798 | 8.2% | 1.5% | 375.1% |

## Position Disagreement

| Asset | current vs monthly-next | current vs same-month-diag |
|---|---|---|
| Forex | 9.5% | 7.4% |
| Equity Index | 8.3% | 6.0% |
| Fixed Income | 7.7% | 6.0% |
| Commodity | 9.2% | 8.5% |

## Quick Reading

- `monthly_12m_next_month_hold` is the most faithful monthly TSMOM analogue in this repo.
- `monthly_12m_same_month_diag` is intentionally look-ahead contaminated and is only a timing-sensitivity probe.
- If `current_daily_252d` and `monthly_12m_next_month_hold` differ meaningfully, then the daily Sign(R) implementation is not just a harmless restatement of the original monthly TSMOM logic.

