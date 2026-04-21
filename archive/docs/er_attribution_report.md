# E(R) Attribution Report

- Focus: full current-baseline attribution across all 50 live contracts and all four asset classes.
- Scope: Table 3 Long, current source doctrine, no historical memory_5 preset.
- Sigma band: `0.058, 0.059, 0.060`

## Sigma Sweep Summary

| Sigma | Live n10/45 | Live n15/45 | Additive n10/45 | Additive n15/45 | All E(R) | All Sharpe |
|---|---|---|---|---|---|---|
| 0.058 | 25/45 | 31/45 | 23/45 | 29/45 | +0.037 | +0.111 |
| 0.059 | 25/45 | 31/45 | 23/45 | 28/45 | +0.037 | +0.111 |
| 0.060 | 26/45 | 31/45 | 24/45 | 28/45 | +0.038 | +0.111 |

## Sigma `0.058`

- Live split-world baseline: `n10=25/45`, `n15=31/45`
- Additive trade-lane attribution context: `n10=23/45`, `n15=29/45`

### Live Split-World Context

| Asset | # | n10 | n15 | E(R) ours | E(R) paper | |E(R) gap| | Sharpe ours | Sharpe paper | |Sharpe gap| |
|---|---|---|---|---|---|---|---|---|---|
| Commodity | 25 | 5 | 6 | -0.232 | -0.298 | 0.066 | -0.621 | -0.723 | 0.102 |
| Equity Index | 11 | 6 | 6 | +0.526 | +0.504 | 0.022 | +0.627 | +0.543 | 0.084 |
| Fixed Income | 5 | 4 | 6 | +0.471 | +0.605 | 0.134 | +0.552 | +0.645 | 0.093 |
| Forex | 9 | 6 | 9 | -0.173 | -0.198 | 0.025 | -0.409 | -0.420 | 0.011 |
| All | 50 | 4 | 4 | +0.037 | -0.013 | 0.050 | +0.111 | -0.036 | 0.147 |

### Additive Trade-Lane Context

| Asset | # | n10 | n15 | E(R) ours | E(R) paper | |E(R) gap| | Sharpe ours | Sharpe paper | |Sharpe gap| |
|---|---|---|---|---|---|---|---|---|---|
| Commodity | 25 | 4 | 5 | -0.232 | -0.298 | 0.066 | -0.621 | -0.723 | 0.102 |
| Equity Index | 11 | 6 | 7 | +0.526 | +0.504 | 0.022 | +0.627 | +0.543 | 0.084 |
| Fixed Income | 5 | 4 | 5 | +0.471 | +0.605 | 0.134 | +0.552 | +0.645 | 0.093 |
| Forex | 9 | 5 | 8 | -0.173 | -0.198 | 0.025 | -0.409 | -0.420 | 0.011 |
| All | 50 | 4 | 4 | +0.037 | -0.013 | 0.050 | +0.111 | -0.036 | 0.147 |

### Add-Back Candidates From Current Excluded Set

| Ticker | Asset | Source | Asset E(R) after add-back | Δ asset |E(R) gap| | Δ asset |Sharpe gap| | Δ All |E(R) gap| | Δ All |Sharpe gap| |
|---|---|---|---|---|---|---|---|

### Commodity

#### Active Sources

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

#### Realized E(R) Contributions

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

#### Leave-One-Out Diagnostics

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

### Equity Index

#### Active Sources

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

#### Realized E(R) Contributions

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

#### Leave-One-Out Diagnostics

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

### Fixed Income

#### Active Sources

| Ticker | Source |
|---|---|
| DT | RAD |
| FB | RAD |
| TY | RAD |
| UB | RAD |
| US | RAD |

#### Realized E(R) Contributions

| Ticker | Source | Trade contrib | Signal contrib | TC contrib | Obs |
|---|---|---|---|---|---|
| FB | RAD | +0.018 | +0.077 | +0.059 | 2268 |
| TY | RAD | +0.062 | +0.097 | +0.035 | 2268 |
| US | RAD | +0.083 | +0.101 | +0.018 | 2268 |
| UB | RAD | +0.117 | +0.193 | +0.076 | 2284 |
| DT | RAD | +0.191 | +0.224 | +0.034 | 2284 |

#### Leave-One-Out Diagnostics

| Ticker | Source | Δ asset |E(R) gap| | Δ asset |Sharpe gap| | Δ All |E(R) gap| | Δ All |Sharpe gap| | Class |
|---|---|---|---|---|---|---|
| FB | RAD | -0.078 | -0.087 | -0.002 | -0.009 | pressure_source |
| TY | RAD | -0.023 | -0.031 | -0.007 | -0.023 | pressure_source |
| US | RAD | +0.001 | +0.004 | -0.009 | -0.029 | supporting_fit |
| UB | RAD | +0.005 | +0.018 | -0.013 | -0.041 | supporting_fit |
| DT | RAD | +0.094 | +0.114 | -0.019 | -0.059 | supporting_fit |

### Forex

#### Active Sources

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

#### Realized E(R) Contributions

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

#### Leave-One-Out Diagnostics

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

## Sigma `0.059`

- Live split-world baseline: `n10=25/45`, `n15=31/45`
- Additive trade-lane attribution context: `n10=23/45`, `n15=28/45`

### Live Split-World Context

| Asset | # | n10 | n15 | E(R) ours | E(R) paper | |E(R) gap| | Sharpe ours | Sharpe paper | |Sharpe gap| |
|---|---|---|---|---|---|---|---|---|---|
| Commodity | 25 | 5 | 6 | -0.236 | -0.298 | 0.062 | -0.621 | -0.723 | 0.102 |
| Equity Index | 11 | 5 | 6 | +0.535 | +0.504 | 0.031 | +0.627 | +0.543 | 0.084 |
| Fixed Income | 5 | 4 | 6 | +0.479 | +0.605 | 0.126 | +0.552 | +0.645 | 0.093 |
| Forex | 9 | 7 | 9 | -0.176 | -0.198 | 0.022 | -0.409 | -0.420 | 0.011 |
| All | 50 | 4 | 4 | +0.037 | -0.013 | 0.050 | +0.111 | -0.036 | 0.147 |

### Additive Trade-Lane Context

| Asset | # | n10 | n15 | E(R) ours | E(R) paper | |E(R) gap| | Sharpe ours | Sharpe paper | |Sharpe gap| |
|---|---|---|---|---|---|---|---|---|---|
| Commodity | 25 | 4 | 5 | -0.236 | -0.298 | 0.062 | -0.621 | -0.723 | 0.102 |
| Equity Index | 11 | 5 | 7 | +0.535 | +0.504 | 0.031 | +0.627 | +0.543 | 0.084 |
| Fixed Income | 5 | 4 | 5 | +0.479 | +0.605 | 0.126 | +0.552 | +0.645 | 0.093 |
| Forex | 9 | 6 | 7 | -0.176 | -0.198 | 0.022 | -0.409 | -0.420 | 0.011 |
| All | 50 | 4 | 4 | +0.037 | -0.013 | 0.050 | +0.111 | -0.036 | 0.147 |

### Add-Back Candidates From Current Excluded Set

| Ticker | Asset | Source | Asset E(R) after add-back | Δ asset |E(R) gap| | Δ asset |Sharpe gap| | Δ All |E(R) gap| | Δ All |Sharpe gap| |
|---|---|---|---|---|---|---|---|

### Commodity

#### Active Sources

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

#### Realized E(R) Contributions

| Ticker | Source | Trade contrib | Signal contrib | TC contrib | Obs |
|---|---|---|---|---|---|
| KW | REV | -0.023 | -0.021 | +0.002 | 2267 |
| ZL | RAD | -0.022 | -0.020 | +0.002 | 2267 |
| SB | RAD | -0.022 | -0.021 | +0.001 | 2266 |
| GI | RAD_REGEN | -0.022 | -0.020 | +0.002 | 2267 |
| ZP | RAD | -0.020 | -0.018 | +0.002 | 2267 |
| KC | RAD | -0.019 | -0.018 | +0.001 | 2266 |
| ZW | REV | -0.018 | -0.015 | +0.002 | 2267 |
| ZN | REV | -0.017 | -0.015 | +0.003 | 2267 |
| ZR | REV | -0.017 | -0.014 | +0.002 | 2267 |
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
| ZG | RAD_REGEN | -0.001 | +0.001 | +0.003 | 2267 |
| JO | RAD_REGEN | +0.001 | +0.002 | +0.001 | 2267 |
| ZO | RAD_REGEN | +0.002 | +0.004 | +0.001 | 2267 |
| LB | RAD_REGEN | +0.004 | +0.005 | +0.001 | 2264 |
| DA | RAD_REGEN | +0.007 | +0.009 | +0.002 | 2263 |
| ZA | RAD_REGEN | +0.015 | +0.016 | +0.001 | 2267 |

#### Leave-One-Out Diagnostics

| Ticker | Source | Δ asset |E(R) gap| | Δ asset |Sharpe gap| | Δ All |E(R) gap| | Δ All |Sharpe gap| | Class |
|---|---|---|---|---|---|---|
| ZA | RAD_REGEN | -0.025 | -0.070 | -0.007 | -0.022 | pressure_source |
| DA | RAD_REGEN | -0.017 | -0.024 | -0.003 | -0.011 | pressure_source |
| LB | RAD_REGEN | -0.014 | -0.019 | -0.002 | -0.007 | pressure_source |
| ZO | RAD_REGEN | -0.012 | -0.030 | -0.001 | -0.004 | pressure_source |
| JO | RAD_REGEN | -0.011 | -0.010 | +0.000 | -0.002 | pressure_source |
| ZG | RAD_REGEN | -0.009 | -0.022 | +0.001 | +0.000 | pressure_source |
| ZT | RAD_REGEN | -0.008 | -0.012 | +0.001 | +0.001 | pressure_source |
| CC | RAD_REGEN | -0.008 | -0.010 | +0.001 | +0.001 | pressure_source |
| ZF | REV | -0.007 | -0.004 | +0.001 | +0.002 | pressure_source |
| ZH | RAD_REGEN | -0.005 | -0.021 | +0.002 | +0.007 | pressure_source |
| ZZ | RAD | -0.005 | +0.001 | +0.003 | +0.006 | mixed |
| ZI | REV | -0.002 | -0.009 | +0.004 | +0.012 | pressure_source |
| ZK | REV | +0.004 | +0.009 | +0.007 | +0.021 | supporting_fit |
| ZU | REV | +0.005 | +0.006 | +0.007 | +0.022 | supporting_fit |
| ZC | RAD | +0.006 | +0.015 | +0.008 | +0.023 | supporting_fit |
| NR | RAD | +0.007 | +0.023 | +0.008 | +0.022 | supporting_fit |
| ZR | REV | +0.007 | +0.025 | +0.008 | +0.023 | supporting_fit |
| ZW | REV | +0.008 | +0.020 | +0.009 | +0.025 | supporting_fit |
| ZN | REV | +0.008 | +0.036 | +0.009 | +0.023 | supporting_fit |
| ZP | RAD | +0.010 | +0.021 | +0.010 | +0.029 | supporting_fit |
| KC | RAD | +0.010 | +0.030 | +0.009 | +0.027 | supporting_fit |
| GI | RAD_REGEN | +0.013 | +0.016 | +0.011 | +0.033 | supporting_fit |
| ZL | RAD | +0.013 | +0.029 | +0.011 | +0.032 | supporting_fit |
| SB | RAD | +0.013 | +0.041 | +0.011 | +0.031 | supporting_fit |
| KW | REV | +0.014 | +0.034 | +0.012 | +0.033 | supporting_fit |

### Equity Index

#### Active Sources

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

#### Realized E(R) Contributions

| Ticker | Source | Trade contrib | Signal contrib | TC contrib | Obs |
|---|---|---|---|---|---|
| LX | RAD | +0.022 | +0.028 | +0.006 | 2279 |
| XU | RAD | +0.028 | +0.033 | +0.005 | 2284 |
| ER | RAD | +0.030 | +0.035 | +0.005 | 2302 |
| MD | RAD | +0.036 | +0.042 | +0.005 | 2259 |
| XX | RAD | +0.037 | +0.043 | +0.006 | 2284 |
| CA | RAD | +0.038 | +0.043 | +0.005 | 2298 |
| SC | RAD_REGEN | +0.064 | +0.071 | +0.007 | 2269 |
| ES | RAD_REGEN | +0.064 | +0.071 | +0.007 | 2269 |
| SP | RAD_REGEN | +0.064 | +0.071 | +0.007 | 2269 |
| YM | RAD | +0.076 | +0.083 | +0.007 | 2269 |
| EN | RAD_REGEN | +0.078 | +0.083 | +0.006 | 2269 |

#### Leave-One-Out Diagnostics

| Ticker | Source | Δ asset |E(R) gap| | Δ asset |Sharpe gap| | Δ All |E(R) gap| | Δ All |Sharpe gap| | Class |
|---|---|---|---|---|---|---|
| YM | RAD | -0.031 | -0.033 | -0.016 | -0.048 | pressure_source |
| EN | RAD_REGEN | -0.029 | -0.040 | -0.017 | -0.049 | pressure_source |
| ES | RAD_REGEN | -0.018 | -0.015 | -0.013 | -0.039 | pressure_source |
| SC | RAD_REGEN | -0.018 | -0.015 | -0.013 | -0.040 | pressure_source |
| SP | RAD_REGEN | -0.018 | -0.015 | -0.013 | -0.040 | pressure_source |
| MD | RAD | +0.013 | +0.017 | -0.008 | -0.022 | supporting_fit |
| ER | RAD | +0.016 | +0.016 | -0.004 | -0.012 | supporting_fit |
| CA | RAD | +0.018 | +0.013 | -0.005 | -0.014 | supporting_fit |
| XX | RAD | +0.018 | +0.013 | -0.006 | -0.018 | supporting_fit |
| LX | RAD | +0.027 | +0.017 | -0.004 | -0.013 | supporting_fit |
| XU | RAD | +0.030 | +0.035 | -0.002 | -0.006 | supporting_fit |

### Fixed Income

#### Active Sources

| Ticker | Source |
|---|---|
| DT | RAD |
| FB | RAD |
| TY | RAD |
| UB | RAD |
| US | RAD |

#### Realized E(R) Contributions

| Ticker | Source | Trade contrib | Signal contrib | TC contrib | Obs |
|---|---|---|---|---|---|
| FB | RAD | +0.018 | +0.078 | +0.060 | 2268 |
| TY | RAD | +0.063 | +0.099 | +0.035 | 2268 |
| US | RAD | +0.085 | +0.103 | +0.018 | 2268 |
| UB | RAD | +0.119 | +0.196 | +0.077 | 2284 |
| DT | RAD | +0.194 | +0.228 | +0.034 | 2284 |

#### Leave-One-Out Diagnostics

| Ticker | Source | Δ asset |E(R) gap| | Δ asset |Sharpe gap| | Δ All |E(R) gap| | Δ All |Sharpe gap| | Class |
|---|---|---|---|---|---|---|
| FB | RAD | -0.079 | -0.087 | -0.002 | -0.009 | pressure_source |
| TY | RAD | -0.024 | -0.031 | -0.007 | -0.023 | pressure_source |
| US | RAD | +0.001 | +0.004 | -0.009 | -0.029 | supporting_fit |
| UB | RAD | +0.005 | +0.018 | -0.013 | -0.041 | supporting_fit |
| DT | RAD | +0.095 | +0.114 | -0.019 | -0.059 | supporting_fit |

### Forex

#### Active Sources

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

#### Realized E(R) Contributions

| Ticker | Source | Trade contrib | Signal contrib | TC contrib | Obs |
|---|---|---|---|---|---|
| JN | REV | -0.069 | -0.056 | +0.013 | 2269 |
| FN | RAD | -0.056 | -0.045 | +0.011 | 2268 |
| CN | RAD | -0.051 | -0.038 | +0.013 | 2269 |
| BN | RAD | -0.038 | -0.026 | +0.011 | 2269 |
| AN | RAD | -0.037 | -0.027 | +0.009 | 2269 |
| MP | RAD_REGEN | -0.030 | -0.022 | +0.008 | 2262 |
| SN | REV | -0.011 | +0.004 | +0.014 | 2269 |
| DX | RAD | +0.031 | +0.046 | +0.015 | 2313 |
| NK | RAD_REGEN | +0.085 | +0.091 | +0.006 | 2326 |

#### Leave-One-Out Diagnostics

| Ticker | Source | Δ asset |E(R) gap| | Δ asset |Sharpe gap| | Δ All |E(R) gap| | Δ All |Sharpe gap| | Class |
|---|---|---|---|---|---|---|
| SN | REV | -0.012 | +0.024 | +0.002 | +0.005 | mixed |
| MP | RAD_REGEN | +0.007 | +0.004 | +0.006 | +0.018 | supporting_fit |
| DX | RAD | +0.015 | +0.007 | -0.003 | -0.013 | supporting_fit |
| AN | RAD | +0.017 | +0.010 | +0.007 | +0.021 | supporting_fit |
| BN | RAD | +0.018 | +0.027 | +0.007 | +0.020 | supporting_fit |
| CN | RAD | +0.033 | +0.058 | +0.010 | +0.030 | supporting_fit |
| FN | RAD | +0.038 | +0.076 | +0.010 | +0.030 | supporting_fit |
| JN | REV | +0.053 | +0.141 | +0.013 | +0.034 | supporting_fit |
| NK | RAD_REGEN | +0.074 | +0.201 | -0.007 | -0.023 | supporting_fit |

## Sigma `0.060`

- Live split-world baseline: `n10=26/45`, `n15=31/45`
- Additive trade-lane attribution context: `n10=24/45`, `n15=28/45`

### Live Split-World Context

| Asset | # | n10 | n15 | E(R) ours | E(R) paper | |E(R) gap| | Sharpe ours | Sharpe paper | |Sharpe gap| |
|---|---|---|---|---|---|---|---|---|---|
| Commodity | 25 | 5 | 6 | -0.240 | -0.298 | 0.058 | -0.621 | -0.723 | 0.102 |
| Equity Index | 11 | 5 | 6 | +0.544 | +0.504 | 0.040 | +0.627 | +0.543 | 0.084 |
| Fixed Income | 5 | 4 | 6 | +0.488 | +0.605 | 0.117 | +0.552 | +0.645 | 0.093 |
| Forex | 9 | 8 | 9 | -0.179 | -0.198 | 0.019 | -0.409 | -0.420 | 0.011 |
| All | 50 | 4 | 4 | +0.038 | -0.013 | 0.051 | +0.111 | -0.036 | 0.147 |

### Additive Trade-Lane Context

| Asset | # | n10 | n15 | E(R) ours | E(R) paper | |E(R) gap| | Sharpe ours | Sharpe paper | |Sharpe gap| |
|---|---|---|---|---|---|---|---|---|---|
| Commodity | 25 | 4 | 5 | -0.240 | -0.298 | 0.058 | -0.621 | -0.723 | 0.102 |
| Equity Index | 11 | 5 | 7 | +0.544 | +0.504 | 0.040 | +0.627 | +0.543 | 0.084 |
| Fixed Income | 5 | 4 | 5 | +0.488 | +0.605 | 0.117 | +0.552 | +0.645 | 0.093 |
| Forex | 9 | 7 | 7 | -0.179 | -0.198 | 0.019 | -0.409 | -0.420 | 0.011 |
| All | 50 | 4 | 4 | +0.038 | -0.013 | 0.051 | +0.111 | -0.036 | 0.147 |

### Add-Back Candidates From Current Excluded Set

| Ticker | Asset | Source | Asset E(R) after add-back | Δ asset |E(R) gap| | Δ asset |Sharpe gap| | Δ All |E(R) gap| | Δ All |Sharpe gap| |
|---|---|---|---|---|---|---|---|

### Commodity

#### Active Sources

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

#### Realized E(R) Contributions

| Ticker | Source | Trade contrib | Signal contrib | TC contrib | Obs |
|---|---|---|---|---|---|
| KW | REV | -0.023 | -0.021 | +0.002 | 2267 |
| ZL | RAD | -0.022 | -0.021 | +0.002 | 2267 |
| SB | RAD | -0.022 | -0.021 | +0.001 | 2266 |
| GI | RAD_REGEN | -0.022 | -0.020 | +0.002 | 2267 |
| ZP | RAD | -0.020 | -0.018 | +0.002 | 2267 |
| KC | RAD | -0.019 | -0.018 | +0.001 | 2266 |
| ZW | REV | -0.018 | -0.016 | +0.002 | 2267 |
| ZN | REV | -0.018 | -0.015 | +0.003 | 2267 |
| ZR | REV | -0.017 | -0.015 | +0.002 | 2267 |
| NR | RAD | -0.016 | -0.015 | +0.002 | 2265 |
| ZC | RAD | -0.016 | -0.015 | +0.002 | 2267 |
| ZU | REV | -0.015 | -0.014 | +0.001 | 2267 |
| ZK | REV | -0.014 | -0.012 | +0.002 | 2267 |
| ZI | REV | -0.008 | -0.006 | +0.002 | 2267 |
| ZZ | RAD | -0.005 | -0.004 | +0.001 | 2267 |
| ZH | RAD_REGEN | -0.005 | -0.003 | +0.002 | 2267 |
| ZF | REV | -0.003 | +0.001 | +0.004 | 2267 |
| CC | RAD_REGEN | -0.002 | -0.000 | +0.001 | 2266 |
| ZT | RAD_REGEN | -0.002 | +0.001 | +0.002 | 2267 |
| ZG | RAD_REGEN | -0.001 | +0.001 | +0.003 | 2267 |
| JO | RAD_REGEN | +0.001 | +0.002 | +0.001 | 2267 |
| ZO | RAD_REGEN | +0.002 | +0.004 | +0.001 | 2267 |
| LB | RAD_REGEN | +0.004 | +0.005 | +0.001 | 2264 |
| DA | RAD_REGEN | +0.007 | +0.009 | +0.002 | 2263 |
| ZA | RAD_REGEN | +0.015 | +0.016 | +0.001 | 2267 |

#### Leave-One-Out Diagnostics

| Ticker | Source | Δ asset |E(R) gap| | Δ asset |Sharpe gap| | Δ All |E(R) gap| | Δ All |Sharpe gap| | Class |
|---|---|---|---|---|---|---|
| ZA | RAD_REGEN | -0.026 | -0.070 | -0.008 | -0.022 | pressure_source |
| DA | RAD_REGEN | -0.017 | -0.024 | -0.003 | -0.011 | pressure_source |
| LB | RAD_REGEN | -0.014 | -0.019 | -0.002 | -0.007 | pressure_source |
| ZO | RAD_REGEN | -0.013 | -0.030 | -0.001 | -0.004 | pressure_source |
| JO | RAD_REGEN | -0.011 | -0.010 | +0.000 | -0.002 | pressure_source |
| ZG | RAD_REGEN | -0.009 | -0.022 | +0.000 | +0.000 | pressure_source |
| ZT | RAD_REGEN | -0.009 | -0.012 | +0.001 | +0.001 | pressure_source |
| CC | RAD_REGEN | -0.009 | -0.010 | +0.000 | +0.001 | pressure_source |
| ZF | REV | -0.007 | -0.004 | +0.001 | +0.002 | pressure_source |
| ZH | RAD_REGEN | -0.005 | -0.021 | +0.002 | +0.007 | pressure_source |
| ZZ | RAD | -0.005 | +0.001 | +0.002 | +0.006 | mixed |
| ZI | REV | -0.002 | -0.009 | +0.004 | +0.012 | pressure_source |
| ZK | REV | +0.005 | +0.009 | +0.007 | +0.021 | supporting_fit |
| ZU | REV | +0.006 | +0.006 | +0.007 | +0.022 | supporting_fit |
| ZC | RAD | +0.007 | +0.015 | +0.008 | +0.023 | supporting_fit |
| NR | RAD | +0.007 | +0.023 | +0.008 | +0.022 | supporting_fit |
| ZR | REV | +0.007 | +0.025 | +0.008 | +0.023 | supporting_fit |
| ZN | REV | +0.008 | +0.036 | +0.009 | +0.023 | supporting_fit |
| ZW | REV | +0.009 | +0.020 | +0.009 | +0.025 | supporting_fit |
| KC | RAD | +0.010 | +0.030 | +0.009 | +0.027 | supporting_fit |
| ZP | RAD | +0.011 | +0.021 | +0.010 | +0.029 | supporting_fit |
| GI | RAD_REGEN | +0.013 | +0.016 | +0.011 | +0.033 | supporting_fit |
| ZL | RAD | +0.013 | +0.029 | +0.011 | +0.032 | supporting_fit |
| SB | RAD | +0.013 | +0.041 | +0.011 | +0.031 | supporting_fit |
| KW | REV | +0.014 | +0.034 | +0.011 | +0.033 | supporting_fit |

### Equity Index

#### Active Sources

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

#### Realized E(R) Contributions

| Ticker | Source | Trade contrib | Signal contrib | TC contrib | Obs |
|---|---|---|---|---|---|
| LX | RAD | +0.022 | +0.029 | +0.006 | 2279 |
| XU | RAD | +0.028 | +0.033 | +0.005 | 2284 |
| ER | RAD | +0.031 | +0.035 | +0.005 | 2302 |
| MD | RAD | +0.037 | +0.042 | +0.005 | 2259 |
| XX | RAD | +0.038 | +0.044 | +0.006 | 2284 |
| CA | RAD | +0.038 | +0.044 | +0.005 | 2298 |
| SC | RAD_REGEN | +0.065 | +0.072 | +0.007 | 2269 |
| ES | RAD_REGEN | +0.065 | +0.072 | +0.007 | 2269 |
| SP | RAD_REGEN | +0.065 | +0.072 | +0.007 | 2269 |
| YM | RAD | +0.077 | +0.085 | +0.007 | 2269 |
| EN | RAD_REGEN | +0.079 | +0.085 | +0.006 | 2269 |

#### Leave-One-Out Diagnostics

| Ticker | Source | Δ asset |E(R) gap| | Δ asset |Sharpe gap| | Δ All |E(R) gap| | Δ All |Sharpe gap| | Class |
|---|---|---|---|---|---|---|
| EN | RAD_REGEN | -0.034 | -0.040 | -0.017 | -0.049 | pressure_source |
| YM | RAD | -0.032 | -0.033 | -0.017 | -0.048 | pressure_source |
| ES | RAD_REGEN | -0.018 | -0.015 | -0.014 | -0.039 | pressure_source |
| SC | RAD_REGEN | -0.018 | -0.015 | -0.014 | -0.040 | pressure_source |
| SP | RAD_REGEN | -0.018 | -0.015 | -0.014 | -0.040 | pressure_source |
| MD | RAD | +0.014 | +0.017 | -0.008 | -0.022 | supporting_fit |
| ER | RAD | +0.016 | +0.016 | -0.004 | -0.012 | supporting_fit |
| XX | RAD | +0.018 | +0.013 | -0.006 | -0.018 | supporting_fit |
| CA | RAD | +0.019 | +0.013 | -0.005 | -0.014 | supporting_fit |
| LX | RAD | +0.027 | +0.017 | -0.005 | -0.013 | supporting_fit |
| XU | RAD | +0.030 | +0.035 | -0.003 | -0.006 | supporting_fit |

### Fixed Income

#### Active Sources

| Ticker | Source |
|---|---|
| DT | RAD |
| FB | RAD |
| TY | RAD |
| UB | RAD |
| US | RAD |

#### Realized E(R) Contributions

| Ticker | Source | Trade contrib | Signal contrib | TC contrib | Obs |
|---|---|---|---|---|---|
| FB | RAD | +0.019 | +0.080 | +0.061 | 2268 |
| TY | RAD | +0.064 | +0.100 | +0.036 | 2268 |
| US | RAD | +0.086 | +0.105 | +0.019 | 2268 |
| UB | RAD | +0.121 | +0.199 | +0.079 | 2284 |
| DT | RAD | +0.197 | +0.232 | +0.035 | 2284 |

#### Leave-One-Out Diagnostics

| Ticker | Source | Δ asset |E(R) gap| | Δ asset |Sharpe gap| | Δ All |E(R) gap| | Δ All |Sharpe gap| | Class |
|---|---|---|---|---|---|---|
| FB | RAD | -0.080 | -0.087 | -0.002 | -0.009 | pressure_source |
| TY | RAD | -0.023 | -0.031 | -0.007 | -0.023 | pressure_source |
| US | RAD | +0.002 | +0.004 | -0.009 | -0.029 | supporting_fit |
| UB | RAD | +0.006 | +0.018 | -0.013 | -0.041 | supporting_fit |
| DT | RAD | +0.098 | +0.114 | -0.020 | -0.059 | supporting_fit |

### Forex

#### Active Sources

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

#### Realized E(R) Contributions

| Ticker | Source | Trade contrib | Signal contrib | TC contrib | Obs |
|---|---|---|---|---|---|
| JN | REV | -0.070 | -0.057 | +0.014 | 2269 |
| FN | RAD | -0.057 | -0.046 | +0.011 | 2268 |
| CN | RAD | -0.052 | -0.039 | +0.013 | 2269 |
| BN | RAD | -0.038 | -0.027 | +0.012 | 2269 |
| AN | RAD | -0.037 | -0.028 | +0.009 | 2269 |
| MP | RAD_REGEN | -0.031 | -0.022 | +0.009 | 2262 |
| SN | REV | -0.011 | +0.004 | +0.015 | 2269 |
| DX | RAD | +0.031 | +0.046 | +0.015 | 2313 |
| NK | RAD_REGEN | +0.086 | +0.092 | +0.006 | 2326 |

#### Leave-One-Out Diagnostics

| Ticker | Source | Δ asset |E(R) gap| | Δ asset |Sharpe gap| | Δ All |E(R) gap| | Δ All |Sharpe gap| | Class |
|---|---|---|---|---|---|---|
| SN | REV | -0.012 | +0.024 | +0.002 | +0.005 | mixed |
| MP | RAD_REGEN | +0.007 | +0.004 | +0.005 | +0.018 | supporting_fit |
| AN | RAD | +0.017 | +0.010 | +0.007 | +0.021 | supporting_fit |
| BN | RAD | +0.019 | +0.027 | +0.007 | +0.020 | supporting_fit |
| DX | RAD | +0.022 | +0.007 | -0.003 | -0.013 | supporting_fit |
| CN | RAD | +0.034 | +0.058 | +0.010 | +0.030 | supporting_fit |
| FN | RAD | +0.039 | +0.076 | +0.010 | +0.030 | supporting_fit |
| JN | REV | +0.054 | +0.141 | +0.013 | +0.034 | supporting_fit |
| NK | RAD_REGEN | +0.082 | +0.201 | -0.008 | -0.023 | supporting_fit |

## Interpretation

- This is the current attribution authority for the repo.
- Any report mentioning `memory_5` or `0.0618 / 0.0627 / 0.0630` should be treated as archive-only.
- Search and data-quality decisions should now cite the deltas in this sigma-band report, not the historical sweep logs.
