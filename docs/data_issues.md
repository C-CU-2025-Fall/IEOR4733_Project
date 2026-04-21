# Data Issues Log

This note records the current practical view of contract-level data issues under
the working trade-world interpretation.

It is intentionally separate from reporting-world questions (`MDD`, `Calmar`)
and separate from paper-internal anomaly notes.

## Working Modified Structural-38 Variant

This note refers to the following modified variant discussed in-session:

```python
overrides = dict(SOURCE_OVERRIDES)
overrides.update(
    {
        "EN": "REV",
        "DT": "RAD",
        "CC": "RAD_REGEN",
        "LB": "RAD",
        "JO": "RAD_REGEN",
        "ZH": "RAD_REGEN",
    }
)
excluded = {"ZA", "FB"}
```

The evaluation context here is:

- Table 3
- four asset classes only:
  - Commodity
  - Equity Index
  - Fixed Income
  - Forex
- trade-world metrics only:
  - `E(R), std(R), DD, Sharpe, Sortino, % +ve, Ave P/L`
- no `MDD`
- no `Calmar`
- no `All`

## Trade-World Alignment Under This Variant

Using the modified structural-38 variant above:

- total trade-world score:
  - `<=10: 22/28`
  - `<=15: 28/28`

Per asset class:

| Asset | `<=10 / 7` | `<=15 / 7` |
| --- | ---: | ---: |
| Commodity | 5 | 7 |
| Equity Index | 6 | 7 |
| Fixed Income | 6 | 7 |
| Forex | 5 | 7 |

Remaining trade-world misses above `10%` but still within `15%`:

| Asset | Metric | Ours | Paper | %Err |
| --- | --- | ---: | ---: | ---: |
| Commodity | `E(R)` | `-0.254` | `-0.298` | `14.7%` |
| Commodity | `Sortino` | `-1.009` | `-1.152` | `12.4%` |
| Equity Index | `Sharpe` | `0.624` | `0.543` | `14.8%` |
| Fixed Income | `Sortino` | `0.962` | `1.081` | `11.0%` |
| Forex | `E(R)` | `-0.173` | `-0.198` | `12.6%` |
| Forex | `std(R)` | `0.423` | `0.472` | `10.4%` |

Interpretation:

- under this modified variant, the trade world is already highly aligned
- the remaining tails are small
- this makes it useful as a data-issues / source-choice reference point, even
  if it is not the final promoted frontier

## Problematic Contract Table

The table below lists historically flagged contracts and the effective choice in
the modified structural-38 variant.

| Contract | Problem batch | Current choice | Comment |
| --- | --- | --- | --- |
| `CC` | negative-`REV` | `RAD_REGEN` | clean fallback from invalid `REV` |
| `LB` | negative-`REV` | `RAD_RAW` | cleaner than `REV`; still sensitive |
| `JO` | negative-`REV` | `RAD_REGEN` | clean fallback |
| `ZH` | negative-`REV`, bad-`RAD` family | `RAD_REGEN` | clean fallback |
| `ZO` | negative-`REV`, structural pressure | `RAD_REGEN` | kept in, not excluded here |
| `ZU` | bad-`RAD` / `v2` family | `REV` | historical source-quality issue remains |
| `US` | bad-`RAD` / `v2` family | `RAD_V2` | explicit `v2` fallback |
| `ZN` | bad-`RAD` / `v2` family | `REV` | historical source-quality issue remains |
| `SB` | commodity distortion | `RAD_RAW` | still included |
| `KC` | commodity distortion | `RAD_RAW` | still included |
| `ZL` | commodity distortion | `RAD_RAW` | still included |
| `NR` | commodity distortion | `RAD_RAW` | still included |
| `ZC` | commodity distortion | `RAD_RAW` | still included |
| `FB` | structural exclusion pressure | `EXCLUDE` | excluded |
| `ZA` | structural exclusion pressure | `EXCLUDE` | excluded |
| `EN` | equity structural pressure | `REV` | added back under this working assumption |
| `ES` | equity structural pressure | `RAD_REGEN` | added back |
| `ZG` | current drop-pressure set | `RAD_REGEN` | still included |
| `ZI` | current drop-pressure set | `REV` | still included |
| `ZZ` | current drop-pressure set | `RAD_RAW` | still included |
| `XU` | current drop-pressure set | `RAD_RAW` | still included |
| `YM` | current drop-pressure set | `RAD_RAW` | still included |
| `TY` | current drop-pressure set | `RAD_RAW` | still included |

## Main Takeaway

Under this modified structural-38 variant:

- the trade-world alignment is already strong
- most remaining uncertainty is no longer “many metrics are broken”
- instead, the remaining questions are:
  - whether some paper cells are themselves anomalous
  - whether reporting-world metrics should be handled in a separate backtest
    module
  - whether a few source-sensitive contracts should be audited further
