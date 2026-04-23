# PROJECT_MEMORY.md
# Last updated: 2026-04-23

Read this first when resuming work on this repo.

## 0. Current Truth

The repo now has one active interpretation:

1. baseline
- `tests/run_structural_38.py`
- this is the true reproducible baseline
- it locks the `structural_38` source overrides and exclusions

2. unified backtester
- one portfolio/metric stack for baseline and DRL
- all final metrics, including `MDD` and `Calmar`, come from the same simulated
  portfolio path

3. DRL mainline
- DQN uses the same `structural_38` data doctrine as the baseline
- one current shared feature/state convention
- no active `v0 / v2 / v2.1 / v3` story in the mainline

Old versioned model families may remain on disk, but they are archive artifacts
and are not part of the active default path.

## 1. Baseline

Primary commands:
- `python baseline_run.py --table 3 --all-metrics --sigma 0.058`
- `python tests/run_structural_38.py --table 3`
- `python tests/run_structural_38.py --table both`
- `python tests/run_structural_38.py --table 3 --with-path-metrics`

Current interpretation:
- the ideal paper/data world is useful reference only
- the actual local baseline is the structural-38 line
- DRL is judged relative to this baseline, not relative to an unattainable
  ideal-data line

## 2. DRL Mainline

Current DRL structure:
- `drl_shared/`
  - shared feature/state construction
- `drl/dqn/`
  - DQN-only model/training/inference code
- `run_strategy_backtest.py`
  - the single public evaluation entrypoint

Current feature/state convention:
- `seq_len = 60`
- `feature_dim = 8`
- close feature:
  - `(p_t - EMA60(p)_t) / (EWMA60(r)_t * sqrt(60))`
- return features:
  - `21 / 42 / 63 / 252`
- MACD normalization:
  - 63-day price volatility
- action space:
  - `{-1, 0, +1}`

Current active artifact layout:
- features:
  - `drl/features/<ticker>/r<round>.npz`
- DQN bundles:
  - `drl/dqn/models/<ticker>/r<round>/<run_id>/`

Archive-only artifact families:
- `drl/dqn/models/walkforward/`
- `drl/dqn/models/v2.1/`

Do not treat archive artifacts as active truth in new code or docs.

## 3. Eq.4 / Sleeve / Unified Path

Do not collapse these three layers:

- Eq.4 trade-return layer
  - additive `r_t = p_t - p_{t-1}`
  - volatility-scaled positions
- sleeve/reporting wealth layer
  - `capital0 = p0 * sigma_tgt / sigma0`
  - not part of Eq.4 itself
- current unified backtest layer
  - one simulated portfolio path
  - final `MDD / Calmar` come from that path

## 4. Current References

- `docs/data_issues.md`
- `docs/paper_table_suspicious_cells.md`
- `docs/drl_pipeline.md`
- `drl/dqn/README.md`
- `docs/structural38_trade_tables_paper_style_a4.png`

## 5. Alignment Snapshot

Current retained structural-38 snapshot:

- trade-world metrics only:
  - `Table 3 <=10: 23/28`
  - `Table 3 <=15: 28/28`
  - `Table 2 <=10: 24/28`
  - `Table 2 <=15: 25/28`
- long-only with unified path metrics:
  - trade metrics `28/28`
  - total `30/36`

Most suspicious paper-side cells still flagged:
- `Equity Index / Table 2 / DD: 0.606`
- `Equity Index / Table 2 / Sortino: 1.102`
- `Fixed Income / Table 2 / Sortino: 1.180`

## 6. 41/45 Status

`41/45` remains only as an experimental upper-bound reproducer:
- `python tests/run_legacy_41.py`

It is not part of the active baseline or the active DRL mainline.

## 7. Archive Notes

Past searches, compatibility lanes, and older versioned DRL interpretations were
useful during diagnosis, but they are no longer the active mainline story.

If a future task explicitly asks for archive behavior:
- treat it as archive work
- do not silently reintroduce version selectors into the active CLI
- do not make old `walkforward` or `models/v2.1` directories the default again
