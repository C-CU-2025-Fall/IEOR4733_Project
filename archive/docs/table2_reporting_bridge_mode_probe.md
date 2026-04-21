# Table 2 Reporting Bridge Mode Probe

- fixed setup:
  - `sigma = 0.058`
  - `port_vol_target = 0.97`
  - `bridges = constant_posthoc, rolling252_lagged`
  - `report_source = RISK_PRICE_SIGMA0`

This report preserves the explicit side-by-side experiment instead of only a prose summary.

## Forex — constant_posthoc — split_world

- contracts: `9`
- `<=10`: `6/9`
- `<=15`: `6/9`

| Metric | Ours | Paper | %Err |
| --- | ---: | ---: | ---: |
| E(R) | -0.397 | -0.344 | 15.3% |
| std(R) | +0.970 | +0.973 | 0.3% |
| DD | +0.625 | +0.583 | 7.2% |
| Sharpe | -0.409 | -0.353 | 15.9% |
| Sortino | -0.635 | -0.590 | 7.6% |
| MDD | +0.220 | +0.423 | 48.0% |
| Calmar | -0.090 | -0.097 | 7.5% |
| % +ve | +0.490 | +0.491 | 0.1% |
| Ave P/L | +0.972 | +0.979 | 0.8% |

## Forex — constant_posthoc — same_as_port_contract

- contracts: `9`
- `<=10`: `5/9`
- `<=15`: `6/9`

| Metric | Ours | Paper | %Err |
| --- | ---: | ---: | ---: |
| E(R) | -0.397 | -0.344 | 15.3% |
| std(R) | +0.970 | +0.973 | 0.3% |
| DD | +0.625 | +0.583 | 7.2% |
| Sharpe | -0.409 | -0.353 | 15.9% |
| Sortino | -0.635 | -0.590 | 7.6% |
| MDD | +0.374 | +0.423 | 11.7% |
| Calmar | -0.075 | -0.097 | 23.1% |
| % +ve | +0.490 | +0.491 | 0.1% |
| Ave P/L | +0.972 | +0.979 | 0.8% |

## Forex — rolling252_lagged — split_world

- contracts: `9`
- `<=10`: `8/9`
- `<=15`: `8/9`

| Metric | Ours | Paper | %Err |
| --- | ---: | ---: | ---: |
| E(R) | -0.366 | -0.344 | 6.4% |
| std(R) | +0.980 | +0.973 | 0.8% |
| DD | +0.635 | +0.583 | 9.0% |
| Sharpe | -0.373 | -0.353 | 5.8% |
| Sortino | -0.576 | -0.590 | 2.3% |
| MDD | +0.220 | +0.423 | 48.0% |
| Calmar | -0.090 | -0.097 | 7.5% |
| % +ve | +0.490 | +0.491 | 0.1% |
| Ave P/L | +0.977 | +0.979 | 0.2% |

## Forex — rolling252_lagged — same_as_port_contract

- contracts: `9`
- `<=10`: `8/9`
- `<=15`: `8/9`

| Metric | Ours | Paper | %Err |
| --- | ---: | ---: | ---: |
| E(R) | -0.366 | -0.344 | 6.4% |
| std(R) | +0.980 | +0.973 | 0.8% |
| DD | +0.635 | +0.583 | 9.0% |
| Sharpe | -0.373 | -0.353 | 5.8% |
| Sortino | -0.576 | -0.590 | 2.3% |
| MDD | +0.385 | +0.423 | 8.9% |
| Calmar | -0.074 | -0.097 | 23.3% |
| % +ve | +0.490 | +0.491 | 0.1% |
| Ave P/L | +0.977 | +0.979 | 0.2% |

## Four-Asset Summary

| Bridge | Mode | Commodity <=15/9 | Equity <=15/9 | Fixed Income <=15/9 | Forex <=15/9 | Total <=10/36 | Total <=15/36 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| constant_posthoc | split_world | 6 | 6 | 4 | 6 | 19 | 22 |
| constant_posthoc | same_as_port_contract | 5 | 5 | 4 | 6 | 18 | 20 |
| rolling252_lagged | split_world | 6 | 6 | 4 | 8 | 20 | 24 |
| rolling252_lagged | same_as_port_contract | 5 | 5 | 4 | 8 | 20 | 22 |

## Reproduce

```bash
python archive/tests/table2_reporting_bridge_mode_probe.py
python baseline_run.py --table 2 --asset Forex --all-metrics --sigma 0.058 --port-vol-target 0.97 --port-bridge constant_posthoc --report-bridge-mode split_world
python baseline_run.py --table 2 --asset Forex --all-metrics --sigma 0.058 --port-vol-target 0.97 --port-bridge constant_posthoc --report-bridge-mode same_as_port_contract
python baseline_run.py --table 2 --asset Forex --all-metrics --sigma 0.058 --port-vol-target 0.97 --port-bridge rolling252_lagged --report-bridge-mode split_world
python baseline_run.py --table 2 --asset Forex --all-metrics --sigma 0.058 --port-vol-target 0.97 --port-bridge rolling252_lagged --report-bridge-mode same_as_port_contract
```
