# Current Baseline Attribution Report

- Focus: current live Table 3 Long baseline attribution under the **current** reference sigma.
- Metric definition: `additive_subset`
- Sigma: `0.058`
- Active excluded set: `(none)`
- Source policy: current `config.SOURCE_OVERRIDES` only; no historical memory_5 preset.
- This report supersedes the older attribution docs that were generated under `0.0618 / 0.0627 / 0.0630` historical sweeps.

## Math Identity

For variable-N aggregation,

```
R_port,t = (1 / N_t) * Σ_i R_i,t
E(R_port) = 252 * mean_t[(1 / N_t) * Σ_i R_i,t]
```

So each contract has realized annualized contribution

```
contrib_i = 252 * mean_t[I_i,t * R_i,t / N_t]
```

and because `R_i,t = signal_i,t - tc_i,t`, the same identity holds for the signal and transaction-cost pieces.

## Current 45-Comparison Context

- Current live split-world baseline score: `n10=25/45`, `n15=31/45`
- Trade-lane additive context used by the attribution math: `n10=23/45`, `n15=29/45`
- The first table below is the current live baseline context; the second is the additive trade-lane context used by the attribution identities.

| Asset | # | n10 | n15 | E(R) ours | E(R) paper | |E(R) gap| | Sharpe ours | Sharpe paper | |Sharpe gap| |
|---|---|---|---|---|---|---|---|---|---|
| Commodity | 25 | 5 | 6 | -0.232 | -0.298 | 0.066 | -0.621 | -0.723 | 0.102 |
| Equity Index | 11 | 6 | 6 | +0.526 | +0.504 | 0.022 | +0.627 | +0.543 | 0.084 |
| Fixed Income | 5 | 4 | 6 | +0.471 | +0.605 | 0.134 | +0.552 | +0.645 | 0.093 |
| Forex | 9 | 6 | 9 | -0.173 | -0.198 | 0.025 | -0.409 | -0.420 | 0.011 |
| All | 50 | 4 | 4 | +0.037 | -0.013 | 0.050 | +0.111 | -0.036 | 0.147 |

### Additive Trade-Lane Context Used For Attribution

| Asset | # | n10 | n15 | E(R) ours | E(R) paper | |E(R) gap| | Sharpe ours | Sharpe paper | |Sharpe gap| |
|---|---|---|---|---|---|---|---|---|---|
| Commodity | 25 | 4 | 5 | -0.232 | -0.298 | 0.066 | -0.621 | -0.723 | 0.102 |
| Equity Index | 11 | 6 | 7 | +0.526 | +0.504 | 0.022 | +0.627 | +0.543 | 0.084 |
| Fixed Income | 5 | 4 | 5 | +0.471 | +0.605 | 0.134 | +0.552 | +0.645 | 0.093 |
| Forex | 9 | 5 | 8 | -0.173 | -0.198 | 0.025 | -0.409 | -0.420 | 0.011 |
| All | 50 | 4 | 4 | +0.037 | -0.013 | 0.050 | +0.111 | -0.036 | 0.147 |

## Commodity

### Active Sources

| Ticker | Source |
|---|---|
| CC | RAD_REGEN |
| DA | RAD_REGEN |
| GI | RAD_REGEN |
| JO | RAD_REGEN |
| KC | RAD |
| KW | REV |
| LB | RAD_REGEN |
| NR | RAD |
| SB | RAD |
| ZA | RAD_REGEN |
| ZC | RAD |
| ZF | REV |
| ZG | RAD_REGEN |
| ZH | RAD_REGEN |
| ZI | REV |
| ZK | REV |
| ZL | RAD |
| ZO | RAD_REGEN |
| ZP | RAD |
| ZR | REV |
| ZT | RAD_REGEN |
| ZU | REV |
| ZW | REV |
| ZZ | RAD |
| ZN | REV |

### Realized E(R) Contributions

| Ticker | Source | Trade contrib | Signal contrib | TC contrib | Obs |
|---|---|---|---|---|---|
| KW | REV | -0.022 | -0.020 | +0.002 | 2267 |
| ZL | RAD | -0.021 | -0.020 | +0.002 | 2267 |
| SB | RAD | -0.021 | -0.020 | +0.001 | 2266 |
| GI | RAD_REGEN | -0.021 | -0.019 | +0.002 | 2267 |
| ZP | RAD | -0.019 | -0.018 | +0.002 | 2267 |
| KC | RAD | -0.019 | -0.018 | +0.001 | 2266 |
| ZW | REV | -0.017 | -0.015 | +0.002 | 2267 |
| ZN | REV | -0.017 | -0.014 | +0.003 | 2267 |
| ZR | REV | -0.016 | -0.014 | +0.002 | 2267 |
| NR | RAD | -0.016 | -0.014 | +0.002 | 2265 |
| ZC | RAD | -0.016 | -0.014 | +0.001 | 2267 |
| ZU | REV | -0.015 | -0.014 | +0.001 | 2267 |
| ZK | REV | -0.014 | -0.012 | +0.002 | 2267 |
| ZI | REV | -0.008 | -0.006 | +0.002 | 2267 |
| ZZ | RAD | -0.005 | -0.004 | +0.001 | 2267 |
| ZH | RAD_REGEN | -0.005 | -0.003 | +0.001 | 2267 |
| ZF | REV | -0.003 | +0.001 | +0.004 | 2267 |
| CC | RAD_REGEN | -0.002 | -0.000 | +0.001 | 2266 |
| ZT | RAD_REGEN | -0.002 | +0.001 | +0.002 | 2267 |
| ZG | RAD_REGEN | -0.001 | +0.001 | +0.002 | 2267 |
| JO | RAD_REGEN | +0.001 | +0.002 | +0.001 | 2267 |
| ZO | RAD_REGEN | +0.002 | +0.003 | +0.001 | 2267 |
| LB | RAD_REGEN | +0.003 | +0.005 | +0.001 | 2264 |
| DA | RAD_REGEN | +0.007 | +0.008 | +0.002 | 2263 |
| ZA | RAD_REGEN | +0.014 | +0.016 | +0.001 | 2267 |

### Leave-One-Out Diagnostics

| Ticker | Source | Δ asset |E(R) gap| | Δ asset |Sharpe gap| | Δ All |E(R) gap| | Δ All |Sharpe gap| | Class |
|---|---|---|---|---|---|---|
| ZA | RAD_REGEN | -0.025 | -0.070 | -0.008 | -0.022 | pressure_source |
| DA | RAD_REGEN | -0.016 | -0.024 | -0.004 | -0.011 | pressure_source |
| LB | RAD_REGEN | -0.014 | -0.019 | -0.002 | -0.007 | pressure_source |
| ZO | RAD_REGEN | -0.012 | -0.030 | -0.002 | -0.004 | pressure_source |
| JO | RAD_REGEN | -0.011 | -0.010 | -0.001 | -0.002 | pressure_source |
| ZG | RAD_REGEN | -0.009 | -0.022 | +0.000 | +0.000 | pressure_source |
| ZT | RAD_REGEN | -0.008 | -0.012 | +0.000 | +0.001 | pressure_source |
| CC | RAD_REGEN | -0.008 | -0.010 | +0.000 | +0.001 | pressure_source |
| ZF | REV | -0.007 | -0.004 | +0.001 | +0.002 | pressure_source |
| ZH | RAD_REGEN | -0.005 | -0.021 | +0.002 | +0.007 | pressure_source |
| ZZ | RAD | -0.005 | +0.001 | +0.002 | +0.006 | mixed |
| ZI | REV | -0.001 | -0.009 | +0.003 | +0.012 | pressure_source |
| ZK | REV | +0.004 | +0.009 | +0.006 | +0.021 | supporting_fit |
| ZU | REV | +0.005 | +0.006 | +0.007 | +0.022 | supporting_fit |
| ZC | RAD | +0.006 | +0.015 | +0.007 | +0.023 | supporting_fit |
| NR | RAD | +0.006 | +0.023 | +0.007 | +0.022 | supporting_fit |
| ZR | REV | +0.007 | +0.025 | +0.008 | +0.023 | supporting_fit |
| ZW | REV | +0.008 | +0.020 | +0.008 | +0.025 | supporting_fit |
| ZN | REV | +0.008 | +0.036 | +0.008 | +0.023 | supporting_fit |
| ZP | RAD | +0.010 | +0.021 | +0.009 | +0.029 | supporting_fit |
| KC | RAD | +0.010 | +0.030 | +0.009 | +0.027 | supporting_fit |
| GI | RAD_REGEN | +0.012 | +0.016 | +0.010 | +0.033 | supporting_fit |
| ZL | RAD | +0.012 | +0.029 | +0.010 | +0.032 | supporting_fit |
| SB | RAD | +0.012 | +0.041 | +0.010 | +0.031 | supporting_fit |
| KW | REV | +0.013 | +0.034 | +0.011 | +0.033 | supporting_fit |

- Strongest current pressure sources: `ZA, DA, LB, ZO, JO`
- Strongest current supporting-fit contracts: `KC, GI, ZL, SB, KW`

## Equity Index

### Active Sources

| Ticker | Source |
|---|---|
| CA | RAD |
| EN | RAD_REGEN |
| ER | RAD |
| ES | RAD_REGEN |
| LX | RAD |
| MD | RAD |
| SC | RAD_REGEN |
| SP | RAD_REGEN |
| XU | RAD |
| XX | RAD |
| YM | RAD |

### Realized E(R) Contributions

| Ticker | Source | Trade contrib | Signal contrib | TC contrib | Obs |
|---|---|---|---|---|---|
| LX | RAD | +0.022 | +0.028 | +0.006 | 2279 |
| XU | RAD | +0.027 | +0.032 | +0.005 | 2284 |
| ER | RAD | +0.030 | +0.034 | +0.005 | 2302 |
| MD | RAD | +0.036 | +0.041 | +0.005 | 2259 |
| XX | RAD | +0.036 | +0.042 | +0.006 | 2284 |
| CA | RAD | +0.037 | +0.042 | +0.005 | 2298 |
| SC | RAD_REGEN | +0.062 | +0.069 | +0.007 | 2269 |
| ES | RAD_REGEN | +0.062 | +0.069 | +0.007 | 2269 |
| SP | RAD_REGEN | +0.062 | +0.069 | +0.007 | 2269 |
| YM | RAD | +0.075 | +0.082 | +0.007 | 2269 |
| EN | RAD_REGEN | +0.076 | +0.082 | +0.006 | 2269 |

### Leave-One-Out Diagnostics

| Ticker | Source | Δ asset |E(R) gap| | Δ asset |Sharpe gap| | Δ All |E(R) gap| | Δ All |Sharpe gap| | Class |
|---|---|---|---|---|---|---|
| ES | RAD_REGEN | -0.017 | -0.015 | -0.014 | -0.039 | pressure_source |
| SC | RAD_REGEN | -0.017 | -0.015 | -0.014 | -0.040 | pressure_source |
| SP | RAD_REGEN | -0.017 | -0.015 | -0.014 | -0.040 | pressure_source |
| YM | RAD | -0.013 | -0.033 | -0.017 | -0.048 | pressure_source |
| EN | RAD_REGEN | -0.011 | -0.040 | -0.017 | -0.049 | pressure_source |
| MD | RAD | +0.013 | +0.017 | -0.008 | -0.022 | supporting_fit |
| ER | RAD | +0.015 | +0.016 | -0.005 | -0.012 | supporting_fit |
| XX | RAD | +0.017 | +0.013 | -0.007 | -0.018 | supporting_fit |
| CA | RAD | +0.018 | +0.013 | -0.005 | -0.014 | supporting_fit |
| LX | RAD | +0.026 | +0.017 | -0.005 | -0.013 | supporting_fit |
| XU | RAD | +0.029 | +0.035 | -0.003 | -0.006 | supporting_fit |

- Strongest current pressure sources: `ES, SC, SP, YM, EN`
- Strongest current supporting-fit contracts: `ER, XX, CA, LX, XU`

## Fixed Income

### Active Sources

| Ticker | Source |
|---|---|
| DT | RAD |
| FB | RAD |
| TY | RAD |
| UB | RAD |
| US | RAD |

### Realized E(R) Contributions

| Ticker | Source | Trade contrib | Signal contrib | TC contrib | Obs |
|---|---|---|---|---|---|
| FB | RAD | +0.018 | +0.077 | +0.059 | 2268 |
| TY | RAD | +0.062 | +0.097 | +0.035 | 2268 |
| US | RAD | +0.083 | +0.101 | +0.018 | 2268 |
| UB | RAD | +0.117 | +0.193 | +0.076 | 2284 |
| DT | RAD | +0.191 | +0.224 | +0.034 | 2284 |

### Leave-One-Out Diagnostics

| Ticker | Source | Δ asset |E(R) gap| | Δ asset |Sharpe gap| | Δ All |E(R) gap| | Δ All |Sharpe gap| | Class |
|---|---|---|---|---|---|---|
| FB | RAD | -0.078 | -0.087 | -0.002 | -0.009 | pressure_source |
| TY | RAD | -0.023 | -0.031 | -0.007 | -0.023 | pressure_source |
| US | RAD | +0.001 | +0.004 | -0.009 | -0.029 | supporting_fit |
| UB | RAD | +0.005 | +0.018 | -0.013 | -0.041 | supporting_fit |
| DT | RAD | +0.094 | +0.114 | -0.019 | -0.059 | supporting_fit |

- Strongest current pressure sources: `FB, TY, US, UB, DT`
- Strongest current supporting-fit contracts: `FB, TY, US, UB, DT`

## Forex

### Active Sources

| Ticker | Source |
|---|---|
| AN | RAD |
| BN | RAD |
| CN | RAD |
| DX | RAD |
| FN | RAD |
| JN | REV |
| MP | RAD_REGEN |
| NK | RAD_REGEN |
| SN | REV |

### Realized E(R) Contributions

| Ticker | Source | Trade contrib | Signal contrib | TC contrib | Obs |
|---|---|---|---|---|---|
| JN | REV | -0.068 | -0.055 | +0.013 | 2269 |
| FN | RAD | -0.055 | -0.044 | +0.011 | 2268 |
| CN | RAD | -0.050 | -0.038 | +0.012 | 2269 |
| BN | RAD | -0.037 | -0.026 | +0.011 | 2269 |
| AN | RAD | -0.036 | -0.027 | +0.009 | 2269 |
| MP | RAD_REGEN | -0.030 | -0.022 | +0.008 | 2262 |
| SN | REV | -0.011 | +0.003 | +0.014 | 2269 |
| DX | RAD | +0.030 | +0.045 | +0.015 | 2313 |
| NK | RAD_REGEN | +0.083 | +0.089 | +0.006 | 2326 |

### Leave-One-Out Diagnostics

| Ticker | Source | Δ asset |E(R) gap| | Δ asset |Sharpe gap| | Δ All |E(R) gap| | Δ All |Sharpe gap| | Class |
|---|---|---|---|---|---|---|
| SN | REV | -0.012 | +0.024 | +0.002 | +0.005 | mixed |
| MP | RAD_REGEN | +0.007 | +0.004 | +0.005 | +0.018 | supporting_fit |
| DX | RAD | +0.008 | +0.007 | -0.003 | -0.013 | supporting_fit |
| AN | RAD | +0.017 | +0.010 | +0.006 | +0.021 | supporting_fit |
| BN | RAD | +0.018 | +0.027 | +0.006 | +0.020 | supporting_fit |
| CN | RAD | +0.033 | +0.058 | +0.009 | +0.030 | supporting_fit |
| FN | RAD | +0.038 | +0.076 | +0.010 | +0.030 | supporting_fit |
| JN | REV | +0.052 | +0.141 | +0.012 | +0.034 | supporting_fit |
| NK | RAD_REGEN | +0.066 | +0.201 | -0.008 | -0.023 | supporting_fit |

- Strongest current pressure sources: `SN, MP, DX, AN, BN`
- Strongest current supporting-fit contracts: `BN, CN, FN, JN, NK`

## Interpretation

- Use this report for any current contract-level attribution discussion.
- Older reports that mention `memory_5` or `sigma=0.0618/0.0627/0.0630` are historical archive only.
- If we want a new exclusion or source claim, it should be justified against the deltas in this report, not the archived sigma-grid runs.
