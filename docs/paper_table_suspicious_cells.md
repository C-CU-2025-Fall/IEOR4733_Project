# Paper Table Suspicious Cells

This note flags cells and Table 2 vs Table 3 transitions that look suspicious
under a basic interpretation of the paper's setup:

- Long-only portfolio
- per-contract volatility scaling already applied in Table 3
- Table 2 described as adding an extra portfolio-level volatility scaling layer

Under that interpretation, some relations should be at least directionally
stable:

- if `std(R)` rises, `DD` usually should not remain exactly unchanged
- if return rises and `MDD` falls sharply, `Calmar` should usually improve clearly
- a simple or monotone portfolio scaling should not make an aggregate row flip
  sign in a dramatic way without also leaving clear traces elsewhere

The table below is not claiming proof of an error. It is a structured list of
cells / transitions that deserve suspicion.

| Location | Metric(s) | Table 3 | Table 2 | Comments |
| --- | --- | ---: | ---: | --- |
| Equity Index / Long | `DD` | `0.606` | `0.606` | Most suspicious single cell. In long-only, if volatility rises (`std(R) 0.928 -> 0.970`), downside deviation staying exactly unchanged is very unnatural under a normal portfolio-level volatility scaling. |
| Equity Index / Long | `std(R)`, `DD`, `Sortino` | `0.928`, `0.606`, `0.831` | `0.970`, `0.606`, `1.102` | Table 2 is internally self-consistent, but the bridge from Table 3 looks abnormal: `std(R)` rises, `DD` does not move, and `Sortino` jumps sharply. |
| Equity Index / Long | `E(R)` vs `std(R)` | `0.504`, `0.928` | `0.668`, `0.970` | Return rises by about `32.5%`, while volatility rises only about `4.5%`. That is unusually imbalanced for a “portfolio vol scaling” story. |
| Fixed Income / Long | `MDD` | `0.108` | `0.061` | Very suspicious. `std(R)` and `DD` both rise slightly (`0.939 -> 0.975`, `0.561 -> 0.576`), yet `MDD` almost halves. |
| Fixed Income / Long | `E(R)`, `MDD`, `Calmar` | `0.605`, `0.108`, `0.455` | `0.680`, `0.061`, `0.444` | If return rises and `MDD` falls this much, `Calmar` would usually improve clearly. Instead it is nearly flat / slightly worse. |
| All / Long | `E(R)` | `-0.013` | `+0.055` | Very suspicious aggregate behavior. Overall return flips sign from negative to positive. A simple post-Table-3 scaling cannot do that. |
| All / Long | `Sharpe`, `Sortino`, `Calmar` | `-0.036`, `-0.057`, `-0.009` | `+0.058`, `+0.092`, `+0.013` | The whole family of risk-adjusted metrics flips sign at once. This strongly suggests Table 2 is not just a clean volatility-scaled version of Table 3. |
| Commodity / Long | `MDD` | `0.248` | `0.350` | Less suspicious than Equity / FI. This one at least moves in the expected direction as volatility increases. The issue here looks more like reporting-path definition than an obvious internal inconsistency. |
| Commodity / Long | `E(R)`, `std(R)`, `DD` ratios | `-0.298`, `0.412`, `0.258` | `-0.710`, `0.979`, `0.604` | These ratios are relatively coherent. Commodity looks more like a genuine scaled transform than Equity or Fixed Income do. |
| Forex / Long | `E(R)`, `std(R)`, `DD` ratios | `-0.198`, `0.472`, `0.285` | `-0.344`, `0.973`, `0.583` | Also relatively coherent. Forex behaves more like “something was scaled” than Equity does. Not perfect, but less suspicious. |

## Highest-priority anomalies

1. `Equity Index / Long / DD: 0.606 -> 0.606`
2. `Fixed Income / Long / MDD: 0.108 -> 0.061`
3. `All / Long / return and risk-adjusted metrics flipping sign`

## Lower-priority anomalies

- `Commodity` has reporting-path issues, but the Table 2 / Table 3 trade-side
  ratios look comparatively coherent.
- `Forex` also looks more like a real scaling transform than the Equity row.
