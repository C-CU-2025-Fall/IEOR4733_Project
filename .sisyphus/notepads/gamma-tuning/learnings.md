## 2026-05-05 13:42 Wave 1 Started
- 4 parallel training jobs launched via tmux (gamma-wave1 session)
- T1: g0.5_s42 | T2: g0.5_s43 | T3: g0.5_s44 | T4: g0.5_s45
- All on Forex r1, 100 episodes, CUDA (NVIDIA GB10)
- Preflight checks passed for all 4
- Expected completion: ~15-22 min each (due to early stopping)

## 2026-05-05 Aggregate Results (Task 18)
- Wrote scripts/aggregate_gamma_results.py — stdlib only, manual quartile interpolation
- summary.csv: 27 rows (9 metrics × 3 gammas), r1/r2 median+Q1+Q3
- per_seed.csv: 135 rows (9 metrics × 15 seeds), individual r1/r2 values
- topk_models.json: best_gamma=0.6 (median r2 Sharpe=0.088), decision=clear_winner
- Validation: all 30 values finite, all IQR > 0
- median_r1_sharpe for top3 seeds: g0.5=-1.047, g0.6=0.088, g0.7=-0.749
- All gammas show negative Sharpe; g0.6 significantly less negative (near zero)

## 2026-05-05 Bug Fix (Task 18b)
- Bug: write_topk() computed median from top-3 seeds only, not all 5 seeds
- Fixed: now reports both all_seed_median_* (correct) and top_3_median_* (supplementary)
- best_gamma = 0.7 (all-seed median r2 Sharpe = -1.036, best among all 3)
- g0.6 all-seed median = -1.285, g0.5 = -1.733
- decision=clear_winner (gap = -1.036 - (-1.285) = 0.249 > 0.05)
- Verified: all all_seed_median_r2_sharpe values match summary.csv exactly
