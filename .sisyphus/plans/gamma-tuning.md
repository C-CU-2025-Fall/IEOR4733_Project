# DQN Gamma Discount Factor Tuning

## TL;DR

> **Quick Summary**: Systematically compare gamma=0.5, 0.6, 0.7 with 5 seeds each on Forex, using r1-trained model to evaluate both r1 and r2 test periods (apple-to-apple). Select best gamma, expand to 10 seeds, generate rolling Sharpe comparison.
>
> **Deliverables**:
> - 15 freshly trained DQN models (3 gamma × 5 seeds) on Forex r1 — zero reuse
> - 30 backtest results (15 models × 2 test periods)
> - Winner gamma selected → full-scale backtest with top seeds only
> - Summary comparison table (median ± IQR per gamma, per metric)
> - Top-k seed selection for best gamma
> - Rolling Sharpe ratio chart comparing best gamma vs baseline vs gamma=0.3
>
> **Estimated Effort**: Medium (~4h training + 1h backtest + 1h analysis)
> **Parallel Execution**: YES — 4 concurrent training jobs
> **Critical Path**: Train all models → Backtest all → Aggregate metrics → Top-k → Rolling Sharpe

---

## Context

### Original Request
User wants to test gamma=0.5, 0.6, 0.7 to determine which discount factor yields better DQN trading performance. Time-constrained, wants 4 parallel jobs, train r1 only and test on both r1+r2 for fair comparison.

### Interview Summary

**Key Discussions**:
- **Gamma status**: spec.py default is 0.7, but all 8 existing checkpoints (2026-05-05, 4 asset classes × 2 rounds) used `--gamma 0.5`. So gamma=0.5 has 1 seed (42) of results.
- **Epsilon mechanism**: 2-phase — (1) first 5000 transitions pure random (buffer fill), (2) then EPS_SCHEDULE based on `step/total_steps` fraction. Identical across all gamma experiments.
- **Gamma conduction**: Confirmed correct — `model.py:learn()` uses `_spec.GAMMA` at runtime (line 325), `train_asset_round()` mutates `_spec.GAMMA` before agent creation (lines 557-559). Python module singleton ensures all code sees the same value.
- **Apple-to-apple comparison**: Train on r1 data only, backtest same checkpoint on both r1 (2011-2015) and r2 (2016-2019) test data. Backtest engine supports this via `--checkpoint-bundle <r1_dir> --round 2`.
- **Parallel execution**: 4 concurrent `train_dqn_walkforward.py` processes. GB10 GPU (130.7GB VRAM) should handle 4× 32K-param models easily.

**Research Findings**:
- **Epsilon complexity**: `buffer_fill_threshold = agent.replay.capacity (=5000)`. First 5000 transitions are `np.random.randint()` — epsilon not even consulted. After buffer fills, `epsilon_for_step(global_step)` interpolates EPS_SCHEDULE. For Forex with 1.1M total_steps, buffer fills at ~0.4% of training — barely any random-only phase relative to total.
- **MEMORY_RATIO bug**: `train_asset_round()` lines 615-621 hardcode `buffer_sizes = {'Forex': 5000, ...}`, bypassing spec.py's `MEMORY_RATIO=0.5`. Low impact for gamma tuning but documented.
- **Feature dimension**: Code uses 9D features (`structural_38_close_norm_9d`), not 12D from PROJECT_MEMORY. Training features are pre-generated and consistent.

### Code Analysis Summary

| Component | File:Line | Status |
|-----------|-----------|--------|
| Gamma definition | `drl/dqn/spec.py:43` | `GAMMA = 0.7` (default) |
| Gamma override | `train_dqn_walkforward.py:557-559` | `_spec.GAMMA = gamma` ✅ |
| Gamma in Q-target | `model.py:325` | `_spec.GAMMA * next_q` ✅ runtime access |
| Epsilon schedule | `spec.py:80-86` | 5-waypoint linear interpolation |
| Buffer fill | `train_dqn_walkforward.py:471-477` | Pure random first 5000 transitions |
| Backtest r1→r2 | `engine.py:164` | `--checkpoint-bundle` bypasses round |
| Reward (Eq.4) | `state_space.py:109-141` | `σ_tgt/σ_{t-1}` scaling, BP=0.002 TC |

---

## Work Objectives

### Core Objective
Determine the optimal DQN discount factor gamma ∈ {0.5, 0.6, 0.7} for Forex futures trading, measured by out-of-sample portfolio metrics on both r1 (2011-2015) and r2 (2016-2019) test periods.

### Concrete Deliverables
- `results/gamma_tuning/` — all experiment outputs
- `results/gamma_tuning/summary.csv` — per-gamma summary (median ± IQR)
- `results/gamma_tuning/topk_models.json` — best k seeds per gamma
- `results/gamma_tuning/rolling_sharpe.png` — comparison chart
- Updated `PROJECT_MEMORY.md` with gamma finding

### Definition of Done
- [ ] 15 models trained (3 gamma × 5 seeds, gamma=0.5/s42 reused)
- [ ] 30 backtests executed (15 × r1 + 15 × r2)
- [ ] `summary.csv` shows clear winner or tie
- [ ] Rolling Sharpe chart generated with matplotlib
- [ ] Top-k seeds identified for best gamma
- [ ] `python -m pytest tests/ -x -q` passes

### Must Have
- Exact same training config for all experiments (only gamma varies)
- Apple-to-apple comparison: r1 model tested on both r1 and r2
- Median ± IQR statistics across seeds per gamma
- Rolling Sharpe ratio visualization

### Must NOT Have (Guardrails)
- No retraining of r2 (defeats apple-to-apple purpose)
- No changes to epsilon schedule, learning rate, or batch size
- No changes to feature set (keep 9D structural_38)
- No cherry-picking seeds — report all seeds, select top-k transparently
- No over-interpretation from single-seed results

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES (pytest)
- **Automated tests**: Tests-after — run existing test suite before/after to confirm no regressions
- **Framework**: pytest (existing in tests/)

### QA Policy
Every task includes agent-executed QA scenarios. Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.
- **CLI training**: interactive_bash (tmux) — submit training command, capture output logs
- **Backtest**: Bash — run backtest commands, assert output format and metric counts
- **Analysis**: Bash (python) — run aggregation scripts, assert file outputs exist

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Training batch A — 4 parallel):
├── T1: Train gamma=0.5, seed=42 [Forex r1]
├── T2: Train gamma=0.5, seed=43 [Forex r1]
├── T3: Train gamma=0.5, seed=44 [Forex r1]
└── T4: Train gamma=0.5, seed=45 [Forex r1]

Wave 2 (Training batch B — 4 parallel):
├── T5: Train gamma=0.5, seed=46 [Forex r1]
├── T6: Train gamma=0.6, seed=42 [Forex r1]
├── T7: Train gamma=0.6, seed=43 [Forex r1]
└── T8: Train gamma=0.6, seed=44 [Forex r1]

Wave 3 (Training batch C — 4 parallel):
├── T9:  Train gamma=0.6, seed=45 [Forex r1]
├── T10: Train gamma=0.6, seed=46 [Forex r1]
├── T11: Train gamma=0.7, seed=42 [Forex r1]
└── T12: Train gamma=0.7, seed=43 [Forex r1]

Wave 4 (Training batch D — 3 parallel):
├── T13: Train gamma=0.7, seed=44 [Forex r1]
├── T14: Train gamma=0.7, seed=45 [Forex r1]
└── T15: Train gamma=0.7, seed=46 [Forex r1]

Wave 5 (Backtest — MAX PARALLEL, all independent):
├── T16: Backtest all 15 models on r1 test period (2011-2015)
└── T17: Backtest all 15 models on r2 test period (2016-2019) using r1 checkpoint

Wave 6 (Analysis — sequential, depends on Wave 5):
├── T18: Aggregate metrics → summary.csv (median ± IQR per gamma)
├── T19: Select best gamma + top-k seeds
├── T20: Generate rolling Sharpe comparison chart

Wave 7 (Conditional — expand best gamma to 10 seeds if winner unclear):
├── T21: Train 5 additional seeds (47-51) for best gamma
└── T22: Backtest + re-aggregate with 10 seeds

Wave 8 (Final full-scale backtest — best gamma only):
├── T23: Full asset-class backtest on all 4 asset classes with best gamma, top seeds
└── T24: Generate final report comparing best gamma vs baseline vs paper DQN

Wave FINAL (4 parallel reviews):
├── F1: Plan compliance audit (oracle)
├── F2: Code quality review (unspecified-high)
├── F3: Real manual QA (unspecified-high)
└── F4: Scope fidelity check (deep)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| T1-T4 | None | T15 | 1 |
| T5-T8 | None | T15 | 2 |
| T9-T12 | None | T15 | 3 |
| T13-T14 | None | T15 | 4 |
| T15 | T1-T14 | T18 | 5 |
| T16 | T1-T14 | T18 | 5 |
| T17 | T1 (existing s42) | T18 | 5 |
| T18 | T15, T16, T17 | T19, T20 | 6 |
| T19 | T18 | F1-F4 | 6 |
| T20 | T18 | F1-F4 | 6 |
| T21 | T19 (conditional) | T22 | 7 |
| T22 | T21 | F1-F4 | 7 |
| F1-F4 | T18-T20 (or T22) | — | FINAL |

**Critical Path**: T1 → T15 → T18 → T19/T20 → F1-F4

---

## TODOs

- [x] 1. Train — gamma=0.5, seed=42, Forex r1

  **Command**: `python drl/dqn/train/train_dqn_walkforward.py --asset Forex --round 1 --gamma 0.5 --seed 42 --episodes 100 --device cuda`
  **Agent**: `quick` | **Wave**: 1 (with T2,T3,T4) | **Blocks**: T16 | **Verification**: exit 0, manifest gamma=0.5/seed=42, best_val_reward finite
  **Evidence**: `task-1-train.log`

- [x] 2. Train — gamma=0.5, seed=43, Forex r1

  **Command**: `--gamma 0.5 --seed 43` (rest same as T1)
  **Agent**: `quick` | **Wave**: 1 | **Blocks**: T16
  **Evidence**: `task-2-train.log`

- [x] 3. Train — gamma=0.5, seed=44, Forex r1

  **Command**: `--gamma 0.5 --seed 44`
  **Agent**: `quick` | **Wave**: 1 | **Blocks**: T16
  **Evidence**: `task-3-train.log`

- [x] 4. Train — gamma=0.5, seed=45, Forex r1

  **Command**: `--gamma 0.5 --seed 45`
  **Agent**: `quick` | **Wave**: 1 | **Blocks**: T16
  **Evidence**: `task-4-train.log`

- [x] 5. Train — gamma=0.5, seed=46, Forex r1

  **Command**: `--gamma 0.5 --seed 46`
  **Agent**: `quick` | **Wave**: 2 (with T6,T7,T8) | **Blocks**: T16
  **Evidence**: `task-5-train.log`

- [x] 6. Train — gamma=0.6, seed=42, Forex r1

  **Command**: `--gamma 0.6 --seed 42`
  **Agent**: `quick` | **Wave**: 2 | **Blocks**: T16
  **Evidence**: `task-6-train.log`

- [x] 7. Train — gamma=0.6, seed=43, Forex r1
- [x] 8. Train — gamma=0.6, seed=44, Forex r1
- [x] 9. Train — gamma=0.6, seed=45, Forex r1
- [x] 10. Train — gamma=0.6, seed=46, Forex r1
- [x] 11. Train — gamma=0.7, seed=42, Forex r1
- [x] 12. Train — gamma=0.7, seed=43, Forex r1
- [x] 13. Train — gamma=0.7, seed=44, Forex r1
- [x] 14. Train — gamma=0.7, seed=45, Forex r1
- [x] 15. Train — gamma=0.7, seed=46, Forex r1

  **Command**: `--gamma 0.7 --seed 46`
  **Agent**: `quick` | **Wave**: 4 | **Blocks**: T16
  **Evidence**: `task-15-train.log`

- [x] 16. Backtest r1 — all 15 models on 2011-2015 test period

  **What to do**:
  - For each of the 15 trained models, run:
    ```bash
    python drl/dqn/backtest/backtest_dqn_walkforward.py \
      --strategy DQN --asset Forex --round 1 \
      --checkpoint-bundle drl/dqn/models/Forex/r1/{run_id} \
      --device cuda
    ```
  - Save each output as `results/gamma_tuning/backtest_r1_{gamma}_{seed}.json`
  - Extract 9 metrics: E(R), std(R), DD, Sharpe, Sortino, MDD, Calmar, %+ve, Ave P/L

  **Must NOT do**: No retraining, no model changes. Pure inference.

  **Agent**: `quick` | **Wave**: 5 (with T17, 30 independent jobs) | **Blocks**: T18 | **Blocked By**: T1-T15

  **References**: `backtest_dqn_walkforward.py`, `engine.py:134-145` (checkpoint_bundle bypass), `engine.py:220-262` (portfolio_metrics)

  **Acceptance Criteria**:
  - [ ] 15 JSON files in `results/gamma_tuning/backtest_r1_*.json`
  - [ ] Each file contains all 9 metrics with finite values
  - [ ] No FileNotFoundError or checkpoint mismatch errors

  **QA**:
  ```
  Scenario: Backtest produces valid metrics for one model
    Tool: Bash
    Steps:
      1. python drl/dqn/backtest/backtest_dqn_walkforward.py --strategy DQN --asset Forex --round 1 --checkpoint-bundle <bundle> --device cuda 2>&1 | tee /tmp/btest_r1.log
      2. grep -c "err=" /tmp/btest_r1.log  # should be 9
    Expected: 9 metric lines, no NaN
    Evidence: .sisyphus/evidence/task-16-sample.log
  ```

  **Evidence**: `task-16-sample.log`, `results/gamma_tuning/backtest_r1_*.json`
  **Commit**: NO

- [x] 17. Backtest r2 — all 15 models on 2016-2019 test period using r1 checkpoint

  **What to do**:
  - Same as T16 but with `--round 2`:
    ```bash
    python drl/dqn/backtest/backtest_dqn_walkforward.py \
      --strategy DQN --asset Forex --round 2 \
      --checkpoint-bundle drl/dqn/models/Forex/r1/{run_id} \
      --device cuda
    ```
  - **Key insight**: `--round 2` loads r2.npz features; `--checkpoint-bundle` loads r1 model → true out-of-sample test
  - Save as `results/gamma_tuning/backtest_r2_{gamma}_{seed}.json`

  **Agent**: `quick` | **Wave**: 5 | **Blocks**: T18 | **Blocked By**: T1-T15

  **References**: `engine.py:164` (checkpoint_bundle bypasses round), `engine.py:127-143` (r2 feature loading)

  **Acceptance Criteria**:
  - [ ] 15 JSON files in `results/gamma_tuning/backtest_r2_*.json`
  - [ ] Non-zero positions generated (not all-flat policy)

  **QA**: Same pattern as T16, verify `--round 2` produces valid metrics.
  **Evidence**: `task-17-sample.log`, `results/gamma_tuning/backtest_r2_*.json`
  **Commit**: NO

- [x] 18. Aggregate metrics → summary.csv (median ± IQR per gamma)

  **What to do**:
  - Write `scripts/aggregate_gamma_results.py`:
    1. Read all `results/gamma_tuning/backtest_r*.json`
    2. Group by gamma value
    3. For each gamma: compute median and IQR (Q1, Q3) of each metric across 5 seeds
    4. Output `results/gamma_tuning/summary.csv` (27 rows: 9 metrics × 3 gammas)
    5. Output `results/gamma_tuning/per_seed.csv` (135 rows: 9 metrics × 15 seeds)

  **Must NOT do**: No cherry-picking — include ALL 5 seeds per gamma.

  **Agent**: `quick` | **Wave**: 6 | **Blocks**: T19,T20 | **Blocked By**: T16,T17

  **Acceptance Criteria**:
  - [ ] `summary.csv`: 27 rows, all values finite, IQR > 0
  - [ ] `per_seed.csv`: 135 rows, all 15 seeds represented

  **QA**: `python3 -c "import pandas as pd; df=pd.read_csv('results/gamma_tuning/summary.csv'); assert len(df)==27; assert set(df.gamma)=={0.5,0.6,0.7}"`

  **Commit**: YES — `feat(gamma): aggregate results with median ± IQR`

- [x] 19. Select best gamma + top-k seeds

  **What to do**:
  - From `per_seed.csv`, rank seeds within each gamma by r2 Sharpe (primary, true OOS) + r1 Sharpe (secondary)
  - Identify best gamma: highest median r2 Sharpe
  - If winner is clear (>0.05 Sharpe above 2nd place AND IQR < 0.1) → proceed to T20-T23
  - If winner unclear (all within 0.03 Sharpe) → trigger T21 expansion to 10 seeds
  - Output `results/gamma_tuning/topk_models.json` with rankings

  **Agent**: `quick` | **Wave**: 6 | **Blocks**: T20,T21 | **Blocked By**: T18

  **Commit**: YES — `feat(gamma): best gamma identified, top-k seeds`

- [x] 20. Generate rolling Sharpe comparison chart

  **What to do**:
  - Write `scripts/plot_rolling_sharpe.py`:
    1. Load top-5 seeds for best gamma
    2. Compute rolling 252-day Sharpe on r1+r2 test data (2011-2019)
    3. Ensemble median + IQR band across seeds
    4. Also plot: Long baseline, gamma=0.3 reference (if historical data available)
    5. Output `results/gamma_tuning/rolling_sharpe.png` (≥800×600px)
    6. Vertical line at 2016-01-01 (r1/r2 boundary)

  **Agent**: `visual-engineering` | **Wave**: 6 | **Blocked By**: T19

  **Acceptance**:
  - [ ] PNG exists, ≥ 800×600px
  - [ ] r1/r2 boundary clearly marked
  - [ ] Best gamma line with IQR band visible

  **QA**: `python3 -c "from PIL import Image; img=Image.open('results/gamma_tuning/rolling_sharpe.png'); assert img.size[0]>=800; print('OK')"`

  **Commit**: YES — `feat(gamma): rolling Sharpe chart + script`

- [ ] 21. (CONDITIONAL) Expand best gamma to 10 seeds

  **Trigger**: Winner unclear (Sharpe gap < 0.03 between best and 2nd, OR IQR > 0.2)
  **Skip if**: Clear winner from 5-seed results

  Train seeds 47-51 for best gamma: `--gamma {BEST} --seed {47..51}`
  **Agent**: `quick` | **Wave**: 7 | **Blocked By**: T19

- [ ] 22. (CONDITIONAL) Re-aggregate with 10 seeds

  **Trigger**: T21 completed. Re-run T18+T19 with 10 seeds. Update charts.
  **Agent**: `quick` | **Wave**: 7 | **Blocked By**: T21

- [ ] 23. Full-scale backtest — best gamma on all 4 asset classes

  **What to do**:
  - With best gamma selected + top seeds identified from T19 (or T22):
    1. Train best gamma on Commodity, Equity Index, Fixed Income r1 (reuse Forex from Waves 1-4)
    2. Backtest all 4 asset classes on r1+r2
    3. Compare against paper Table 3 DQN baseline
  - Commands per asset class:
    ```bash
    python drl/dqn/train/train_dqn_walkforward.py --asset {class} --round 1 --gamma {BEST} --seed {TOP_SEED} --episodes 100 --device cuda
    ```

  **Agent**: `quick` | **Wave**: 8 | **Blocked By**: T19

- [ ] 24. Final comparison report

  **What to do**:
  - Aggregate all 4 asset class results with best gamma
  - Compare vs paper Table 3 DQN metrics (E(R), Sharpe, Sortino, etc.)
  - Output `results/gamma_tuning/final_report.json` with per-asset and all-asset metrics
  - Update PROJECT_MEMORY.md with findings (optional, user discretion)

  **Agent**: `quick` | **Wave**: 8 | **Blocked By**: T23



---

## Final Verification Wave

- [ ] F1. **Plan Compliance Audit** — `oracle`

  Read the plan end-to-end. Verify:
  - 15 training tasks executed (T1-T15, all fresh, zero reuse)
  - All backtest JSONs exist in results/gamma_tuning/ (15 × r1 + 15 × r2 = 30)
  - summary.csv has correct structure
  - topk_models.json identifies best gamma
  - rolling_sharpe.png exists and is valid
  - No "Must NOT do" violated (check no r2 training, no changed hyperparams, no reuse of old checkpoints)
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`

  Verify:
  - `scripts/aggregate_gamma_results.py` passes `python -m py_compile`
  - `scripts/plot_rolling_sharpe.py` passes `python -m py_compile`
  - No `as any`/`@ts-ignore` in new scripts
  - Existing test suite: `python -m pytest tests/ -x -q` passes
  - All 15 training manifests have correct gamma/seed recorded
  Output: `Compile [PASS/FAIL] | Tests [N pass/N fail] | Manifests [N/N] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`

  Execute:
  - Pick one trained model, verify checkpoint loads and gamma matches manifest
  - Run a single backtest and confirm output format has all 9 metrics
  - Verify rolling_sharpe.png opens without corruption
  - Check r1/r2 boundary is clearly marked at 2016-01-01
  - Verify no old model reuse — all 15 bundles have fresh timestamps
  Output: `Checks [N/N pass] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`

  Verify:
  - Only Forex asset class used for gamma tuning (no contamination from other classes)
  - Only r1 training (no r2 checkpoint created in any bundle)
  - All 15 models accounted for in results (no missing, no extras)
  - No feature dimension change (all use 9D structural_38_close_norm_9d)
  - Gamma conduction verified: each model's manifest gamma matches training command
  - If T23 executed: all 4 asset classes trained with same best gamma
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | VERDICT`

---

## Commit Strategy

- **T18**: `feat(gamma): aggregate gamma tuning results with median ± IQR` — `scripts/aggregate_gamma_results.py`, `results/gamma_tuning/summary.csv`, `results/gamma_tuning/per_seed.csv`
- **T19**: `feat(gamma): top-k seed selection, best gamma identified` — `results/gamma_tuning/topk_models.json`
- **T20**: `feat(gamma): rolling Sharpe comparison chart + script` — `scripts/plot_rolling_sharpe.py`, `results/gamma_tuning/rolling_sharpe.png`
- **T22** (if triggered): `feat(gamma): expand best gamma to 10 seeds, final results` — updated CSVs, JSON, PNG
- **T24** (if triggered): `feat(gamma): full-scale backtest on all asset classes with best gamma` — final report

---

## Success Criteria

### Verification Commands
```bash
# Verify 15 fresh checkpoints exist (no reuse)
find drl/dqn/models/Forex/r1 -name "checkpoint.pt" -newer /tmp/gamma_tuning_start -type f | wc -l
# Expected: 15

# Verify all manifests have correct gamma
for d in drl/dqn/models/Forex/r1/*/; do
  python3 -c "import json; m=json.load(open('${d}manifest.json')); print(m['hyperparameters']['gamma'], m['seed'])"
done | sort

# Verify summary structure
python3 -c "
import pandas as pd
df = pd.read_csv('results/gamma_tuning/summary.csv')
assert set(df['gamma'].unique()) == {0.5, 0.6, 0.7}
assert len(df) == 27  # 9 metrics × 3 gammas
print('summary.csv: OK')
"

# Verify existing tests still pass
python -m pytest tests/ -x -q

# Verify rolling Sharpe image
python3 -c "
from PIL import Image
img = Image.open('results/gamma_tuning/rolling_sharpe.png')
assert img.size[0] >= 800
print(f'rolling_sharpe.png: {img.size} OK')
"
```

### Final Checklist
- [ ] All 15 training tasks completed fresh (zero reuse)
- [ ] All 30 backtest JSONs generated (15 × r1 + 15 × r2)
- [ ] `summary.csv` shows clear best gamma or documents tie
- [ ] `topk_models.json` identifies top-3 seeds per gamma
- [ ] `rolling_sharpe.png` exists with r1/r2 demarcation
- [ ] Best gamma selected based on r2 Sharpe (true OOS)
- [ ] If winner clear: proceed to T23-T24 (full-scale backtest)
- [ ] If winner unclear: expand to 10 seeds (T21-T22)
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent (no r2 training, no reuse, no hyperparam changes)
- [ ] Existing test suite passes
- [ ] PROJECT_MEMORY.md updated with gamma finding (optional)


  **What to do**:
  - Run: `python drl/dqn/train/train_dqn_walkforward.py --asset Forex --round 1 --gamma 0.6 --seed 42 --episodes 100 --device cuda`
  - Record run_id for backtest

  **Must NOT do**: Same constraints as Task 1.

  **Recommended Agent Profile**: `quick`, Skills: []

  **Parallelization**: Wave 2 (with Tasks 6, 7, 8). Blocks Task 15. Blocked By: None.

  **References**:
  - Same as Task 1 for training mechanics
  - `drl/dqn/model.py:325` — `_spec.GAMMA` in Q-target computation

  **Acceptance Criteria**:
  - [ ] Training exit code 0
  - [ ] manifest records `"gamma": 0.6`, `"seed": 42`
  - [ ] `best_val_reward` finite

  **QA Scenarios**: Same pattern — verify exit code, checkpoint, manifest gamma/seed.

  **Evidence**: `task-5-train-complete.log`

  **Commit**: NO

- [ ] 6. Train DQN — gamma=0.6, seed=43, Forex r1

  **What to do**: `python drl/dqn/train/train_dqn_walkforward.py --asset Forex --round 1 --gamma 0.6 --seed 43 --episodes 100 --device cuda`

  **Recommended Agent Profile**: `quick`, Skills: []

  **Parallelization**: Wave 2. Blocks Task 15.

  **Acceptance Criteria**: Exit code 0, gamma=0.6/seed=43 in manifest.

  **Evidence**: `task-6-train-complete.log`

  **Commit**: NO

- [ ] 7. Train DQN — gamma=0.6, seed=44, Forex r1

  **What to do**: `python drl/dqn/train/train_dqn_walkforward.py --asset Forex --round 1 --gamma 0.6 --seed 44 --episodes 100 --device cuda`

  **Recommended Agent Profile**: `quick`, Skills: []

  **Parallelization**: Wave 2. Blocks Task 15.

  **Acceptance Criteria**: Exit code 0, gamma=0.6/seed=44.

  **Evidence**: `task-7-train-complete.log`

  **Commit**: NO

- [ ] 8. Train DQN — gamma=0.6, seed=45, Forex r1

  **What to do**: `python drl/dqn/train/train_dqn_walkforward.py --asset Forex --round 1 --gamma 0.6 --seed 45 --episodes 100 --device cuda`

  **Recommended Agent Profile**: `quick`, Skills: []

  **Parallelization**: Wave 2. Blocks Task 15.

  **Acceptance Criteria**: Exit code 0, gamma=0.6/seed=45.

  **Evidence**: `task-8-train-complete.log`

  **Commit**: NO

- [ ] 9. Train DQN — gamma=0.6, seed=46, Forex r1

  **What to do**: `python drl/dqn/train/train_dqn_walkforward.py --asset Forex --round 1 --gamma 0.6 --seed 46 --episodes 100 --device cuda`

  **Recommended Agent Profile**: `quick`, Skills: []

  **Parallelization**: Wave 3 (with Tasks 10, 11, 12). Blocks Task 15.

  **Acceptance Criteria**: Exit code 0, gamma=0.6/seed=46.

  **Evidence**: `task-9-train-complete.log`

  **Commit**: NO

- [ ] 10. Train DQN — gamma=0.7, seed=42, Forex r1

  **What to do**: `python drl/dqn/train/train_dqn_walkforward.py --asset Forex --round 1 --gamma 0.7 --seed 42 --episodes 100 --device cuda`

  **Recommended Agent Profile**: `quick`, Skills: []

  **Parallelization**: Wave 3. Blocks Task 15.

  **Acceptance Criteria**: Exit code 0, gamma=0.7/seed=42.

  **Evidence**: `task-10-train-complete.log`

  **Commit**: NO

- [ ] 11. Train DQN — gamma=0.7, seed=43, Forex r1

  **What to do**: `python drl/dqn/train/train_dqn_walkforward.py --asset Forex --round 1 --gamma 0.7 --seed 43 --episodes 100 --device cuda`

  **Recommended Agent Profile**: `quick`, Skills: []

  **Parallelization**: Wave 3. Blocks Task 15.

  **Acceptance Criteria**: Exit code 0, gamma=0.7/seed=43.

  **Evidence**: `task-11-train-complete.log`

  **Commit**: NO

- [ ] 12. Train DQN — gamma=0.7, seed=44, Forex r1

  **What to do**: `python drl/dqn/train/train_dqn_walkforward.py --asset Forex --round 1 --gamma 0.7 --seed 44 --episodes 100 --device cuda`

  **Recommended Agent Profile**: `quick`, Skills: []

  **Parallelization**: Wave 3. Blocks Task 15.

  **Acceptance Criteria**: Exit code 0, gamma=0.7/seed=44.

  **Evidence**: `task-12-train-complete.log`

  **Commit**: NO

- [ ] 13. Train DQN — gamma=0.7, seed=45, Forex r1

  **What to do**: `python drl/dqn/train/train_dqn_walkforward.py --asset Forex --round 1 --gamma 0.7 --seed 45 --episodes 100 --device cuda`

  **Recommended Agent Profile**: `quick`, Skills: []

  **Parallelization**: Wave 4 (with Task 14). Blocks Task 15.

  **Acceptance Criteria**: Exit code 0, gamma=0.7/seed=45.

  **Evidence**: `task-13-train-complete.log`

  **Commit**: NO

- [ ] 14. Train DQN — gamma=0.7, seed=46, Forex r1

  **What to do**: `python drl/dqn/train/train_dqn_walkforward.py --asset Forex --round 1 --gamma 0.7 --seed 46 --episodes 100 --device cuda`

  **Recommended Agent Profile**: `quick`, Skills: []

  **Parallelization**: Wave 4. Blocks Task 15.

  **Acceptance Criteria**: Exit code 0, gamma=0.7/seed=46.

  **Evidence**: `task-14-train-complete.log`

  **Commit**: NO


  **What to do**:
  - Run: `python drl/dqn/train/train_dqn_walkforward.py --asset Forex --round 1 --gamma 0.5 --seed 45 --episodes 100 --device cuda`

  **Must NOT do**: Same as Task 1.

  **Recommended Agent Profile**: `quick`, Skills: []

  **Parallelization**: Wave 1 (with Tasks 1, 2, 4). Blocks Task 15.

  **References**: Same as Task 1.

  **Acceptance Criteria**: Exit code 0, checkpoint exists, manifest records gamma=0.5/seed=45.

  **QA Scenarios**: Same pattern as Task 1.

  **Evidence**: `task-3-train-complete.log`

  **Commit**: NO

- [ ] 4. Train DQN — gamma=0.5, seed=46, Forex r1

  **What to do**:
  - Run: `python drl/dqn/train/train_dqn_walkforward.py --asset Forex --round 1 --gamma 0.5 --seed 46 --episodes 100 --device cuda`

  **Must NOT do**: Same as Task 1.

  **Recommended Agent Profile**: `quick`, Skills: []

  **Parallelization**: Wave 1 (with Tasks 1, 2, 3). Blocks Task 15.

  **References**: Same as Task 1.

  **Acceptance Criteria**: Exit code 0, checkpoint exists, manifest records gamma=0.5/seed=46.

  **QA Scenarios**: Same pattern as Task 1.

  **Evidence**: `task-4-train-complete.log`

  **Commit**: NO

