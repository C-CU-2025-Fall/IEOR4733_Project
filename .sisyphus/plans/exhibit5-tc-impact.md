# Exhibit 5: Transaction Cost Impact on DQN Learning

## TL;DR

> **Quick Summary**: Retrain DQN at 9 new BP levels (paper: 1/5/10/15/25/30/35/40/45 bps) × 4 assets × 2 rounds × 10 seeds, reuse existing BP=20 models. 2-phase execution: Phase 1 (1/10/30/45) to prove cost effect, Phase 2 (5/15/25/35/40) for full paper reproduction. Panel A (Sharpe vs BP) + Panel B (Avg Daily Cost Per Contract vs BP).

> **Deliverables**:
> - BP-aware training pipeline (`--tc-bp` CLI flag, bp-prefixed run_id)
> - ~500 retrained models (560 total minus ~60 existing BP=20)
> - Backtest/ensemble pipeline reading BP from manifest
> - Exhibit 5 figure (`figures/exhibit5_tc_impact.png`)
> - Training completion tracker

> **Estimated Effort**: Large (720 training jobs: 4 assets × 2 rounds × 9 BP × 10 seeds in 2 phases)
> **Parallel Execution**: YES — 8 waves, max 7 parallel tasks in Wave 1
> **Critical Path**: Task 1 (BP plumbing) → Task 5 (migrate BP=20 models) → Task 8 (train Commodity r1 Phase 1) → Task 16 (Commodity ensemble) → Task 20 (Sharpe) → Task 22 (Exhibit 5 figure) → F1-F4 (reviews)
> **2-Phase Strategy**: Phase 1 (1/10/30/45 bps, 320 jobs) proves cost effect. Phase 2 (5/15/25/35/40 bps, 400 jobs after Phase 1 validated) for full paper reproduction.

---

## Context

### Original Request
"根据目前本地代码最新的情况，dqn的readme，制定最后复现exhibit5 的计划 panel A sharpe ratio, panel B average cost per contract"

### Interview Summary
**Key Decisions**:
- Asset classes: ALL 4 (Commodity, Equity Index, Fixed Income, Forex)
- **BP levels**: 10 levels matching paper — 1, 5, 10, 15, 20, 25, 30, 35, 40, 45 bps
   (config values: 0.0001, 0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040, 0.0045)
  - **Phase 1** (priority): 1, 10, 30, 45 bps — prove cost changes agent behavior
  - **Phase 2**: 5, 15, 25, 35, 40 bps — fill in for full paper reproduction
  - BP=20 already done — reuse existing models
- Seeds: full 10 seeds (42-51), top-5 validation-selected ensemble
- Walk-forward: r1 (2005-2010 train / 2011-2015 test), r2 (2005-2015 train / 2016-2019 test)
- Total: 4 × 2 × 7 × 10 = 560 training jobs, minus ~60 existing BP=20 models = ~500 new
- Existing BP=20 models MUST be reused (not retrained)
- Ensemble: per-(asset, round, BP) top-5 by validation reward, Q-value averaging (not simple mean)

**Research Findings**:
- `BP = 0.0020` hardcoded in `config.py` line 10 — no CLI/YAML override exists
- Training: `ContractEnv.step()` → `compute_eq4_reward(bp=BP)` in `drl_shared/state_space.py:220`
- Backtest: `compute_contract_returns_from_positions()` in `baseline_run.py:478` uses `BP * prices[...] * np.abs(sp - spp)`
- Current run_id format: `{timestamp}_s{seed}` (e.g., `20260505T180235_s42`)
- Models stored at: `drl/dqn/models/{asset}/r{round}/{timestamp}_s{seed}/`
- `generate_ensemble_table2.py` discovers models by scanning `models/{asset}/r{round}/` for `*_s{seed}` pattern
- `compute_eq4_reward` signature has `bp` parameter with default `BP` — already parameter-friendly

### Metis Review
**Identified Gaps** (addressed):
- **Model directory collision** (CRITICAL): Encode BP in run_id prefix: `bp{XX}_{timestamp}_s{seed}`
- **Backtest BP consistency**: Read BP from manifest.json, never from CLI or config.py
- **BP=0 edge case**: Division by zero? Not in current formula — `bp * prices * abs(diff)` correctly returns 0 when bp=0
- **Sharpe formula definition**: annualized, risk-free=0, Sharpe = mean(R_daily) / std(R_daily) × √252
- **Avg cost per contract definition**: total_TC / (n_contracts × n_days_in_test_period) — daily average dollar cost per contract
- **Ensemble selection per-BP**: Top-5 validation reward per (asset, round, BP) — NEVER cross-BP
- **Job tracking**: Required for 560 jobs — completion manifest with per-job status

---

## Work Objectives

### Core Objective
Reproduce Exhibit 5 from Zhang et al. (2019) showing how transaction cost levels impact DQN learning behavior, measured via Sharpe ratio and average daily transaction cost per contract.

### Concrete Deliverables
- `drl/dqn/spec.py` — `TC_BP_LEVELS` constant, `make_run_id(bp)` parameterized
- `drl/dqn/train/train_dqn_walkforward.py` — `--tc-bp` CLI argument
- `drl/dqn/reports/generate_ensemble_table2.py` — BP-aware model discovery + backtest
- `drl/dqn/reports/ensemble_table2_bp/` — R data and metrics per BP level
- `drl/dqn/models/{asset}/r{round}/bp{XX}_{timestamp}_s{seed}/` — all trained models
- `drl/dqn/figures/exhibit5_tc_impact.py` — figure generation script
- `drl/dqn/figures/exhibit5_tc_impact.png` — final 2-panel figure

### Definition of Done
- [ ] `--tc-bp` CLI flag functional: `python train_dqn_walkforward.py --tc-bp 0.0010 --asset Commodity --round 1 --seed 42` completes
- [ ] All 560 models trained (or confirmed existing) — training completion tracker shows 560/560
- [ ] Ensemble backtest produces Sharpe + cost metrics at all 7 BP levels × 4 assets
- [ ] Exhibit 5 figure generated: Panel A (4 lines, Sharpe vs BP), Panel B (4 lines, Cost vs BP)
- [ ] BP=20 results match existing Table 2 within tolerance (std(R)≈0.97, Sharpe values match)

### Must Have
- BP level encoded in model directory (no collision between BP levels)
- Backtest uses model's trained BP (from manifest.json, not config.py)
- Ensemble selection per-(asset, round, BP) — never cross-BP
- All other hyperparameters IDENTICAL across BP levels

### Must NOT Have (Guardrails)
- **MUST NOT** modify reward function logic (`compute_eq4_reward` formula) — only parameterize BP
- **MUST NOT** change gamma, episodes, epsilon schedule, LSTM architecture, or any other hyperparameter
- **MUST NOT** add new asset classes
- **MUST NOT** change walk-forward date ranges
- **MUST NOT** create new ensemble selection method
- **MUST NOT** add visualization panels beyond the 2 specified
- **MUST NOT** refactor unrelated code paths
- **MUST NOT** allow backtest BP to differ from training BP
- **MUST NOT** mix different BP levels in the same ensemble

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed.

### Test Decision
- **Infrastructure exists**: YES (pytest in project root)
- **Automated tests**: Tests-after (validate plumbing, then train, then backtest)
- **Framework**: pytest
- **No TDD** — training is too heavy for TDD; tests validate pipeline before mass training

### QA Policy
Every task includes Agent-Executed QA Scenarios. Evidence saved to `.sisyphus/evidence/`:
- **CLI/API**: Bash (python/bun/node) — run scripts, assert exit codes, check output
- **Data validation**: Bash (python) — load NPZ/JSON, assert shapes, assert values
- **Figure verification**: Bash (python) — load PNG, assert dimensions, count subplots

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — plumbing + validation):
├── Task 1: BP parameter plumbing (spec.py, state_space, train_dqn_walkforward)
├── Task 2: Model directory BP encoding (make_run_id, find_latest_bundle)
├── Task 3: Manifest BP recording (checkpoint_metadata, train_config)
├── Task 4: Backtest BP-aware pipeline (generate_ensemble_table2)
├── Task 5: Migrate existing BP=20 models (rename dirs, validate)
├── Task 6: BP=0 edge case test
└── Task 7: Orchestration script + job tracker

Wave 1 (Start Immediately — plumbing + validation + FIGURE PROTOTYPE):
├── Task 1: BP parameter plumbing (spec.py, state_space, train_dqn_walkforward)
├── Task 2: Model directory BP encoding (make_run_id, find_latest_bundle)
├── Task 3: Manifest BP recording (checkpoint_metadata, train_config)
├── Task 4: Backtest BP-aware pipeline (generate_ensemble_table2)
├── Task 5: DRY-RUN migration test (rename 1 model → verify → rollback)
├── Task 5b: Execute migration of all BP=20 models (after dry-run confirmed)
├── Task 6: BP edge case + equation validation tests
├── Task 7: Orchestration script + job tracker (with nohup/tmux support)
├── Task 7b: EXHIBIT5 FIGURE PROTOTYPE — write figure script NOW, test with BP=20 data

Wave 2 (After Wave 1 — Phase 1 training, 4 BP × 4 assets × 2 rounds, MAX PARALLEL):
├── Task 8: Train Commodity r1 Phase 1 (1/10/30/45 bps × 10 seeds = 40 jobs)
├── Task 9: Train Commodity r2 Phase 1 (40 jobs)
├── Task 10: Train Equity Index r1 Phase 1 (40 jobs)
├── Task 11: Train Equity Index r2 Phase 1 (40 jobs)

Wave 3 (After Wave 2 — Phase 1 for remaining assets, MAX PARALLEL):
├── Task 12: Train Fixed Income r1 Phase 1 (40 jobs)
├── Task 13: Train Fixed Income r2 Phase 1 (40 jobs)
├── Task 14: Train Forex r1 Phase 1 (40 jobs)
├── Task 15: Train Forex r2 Phase 1 (40 jobs)

Wave 3.5 (After Wave 3 — Phase 1 ensemble + preliminary Exhibit 5, validate before Phase 2):
├── Task 16: Ensemble Commodity Phase 1 (1/10/30/45 bps)
├── Task 17: Ensemble Equity Index Phase 1
├── Task 18: Ensemble Fixed Income Phase 1
├── Task 19: Ensemble Forex Phase 1
├── Task 20: Compute Sharpe + Cost Phase 1
├── Task 21: UPDATE Exhibit 5 with Phase 1 data (4 points + BP=20)
├── Task 22: VALIDATION GATE — user confirms cost effect is visible before Phase 2

Wave 4 (After Gate — Phase 2 training, 5 BP × 4 assets × 2 rounds, MAX PARALLEL):
├── Task 23-26: Train Commodity + Equity r1+r2 Phase 2 (5/15/25/35/40 bps, 50 jobs each)
├── Task 27-30: Train FI + Forex r1+r2 Phase 2 (50 jobs each)

Wave 5 (After Wave 4 — Phase 2 ensemble, MAX PARALLEL):
├── Task 31-34: Ensemble all 4 assets Phase 2 (5/15/25/35/40 bps)

Wave 6 (After Wave 5 — FINAL figure):
├── Task 35: Compute Sharpe + Cost ALL 10 BP levels
└── Task 36: Generate FINAL Exhibit 5 (10 data points per asset)

Wave 4 (After Waves 2-3 — ensemble backtest per BP, parallel per asset):
├── Task 16: Ensemble Commodity (7 BP levels)
├── Task 17: Ensemble Equity Index (7 BP levels)
├── Task 18: Ensemble Fixed Income (7 BP levels)
├── Task 19: Ensemble Forex (7 BP levels)

Wave 5 (After Wave 4 — metrics computation):
├── Task 20: Compute Sharpe per (asset, BP)
├── Task 21: Compute Avg Cost per (asset, BP)

Wave 6 (After Wave 5 — figure generation):
└── Task 22: Generate Exhibit 5 figure (Panel A + Panel B)

Wave FINAL (After ALL tasks — 4 parallel reviews):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
→ Present results → Get explicit user okay
```

**Critical Path**: Task 1 → Task 5 → Task 8-9 → Task 16 → Task 20 → Task 22 → F1-F4
**Parallel Speedup**: ~75% faster than sequential (4-asset parallel training)
**Max Concurrent**: 7 (Wave 1), 4 (Waves 2-4)

### Agent Dispatch Summary

- **Wave 1**: 7 tasks — plumbing, migration, testing
- **Wave 2**: 4 tasks — Commodity + Equity training (280 jobs)
- **Wave 3**: 4 tasks — FI + Forex training (280 jobs)
- **Wave 4**: 4 tasks — ensemble backtest per asset
- **Wave 5**: 2 tasks — metrics
- **Wave 6**: 1 task — figure
- **FINAL**: 4 tasks — reviews

---

## TODOs

- [ ] 1. BP Parameter Plumbing — `spec.py`, `state_space.py`, `train_dqn_walkforward.py`

  **What to do**:
  - Add `TC_BP_LEVELS = [0.0001, 0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040, 0.0045]` constant to `spec.py` (maps to 1/5/10/15/20/25/30/35/40/45 bps — matches paper Exhibit 5 exactly)
  - Add `TC_BP_PHASE1 = [0.0001, 0.0010, 0.0030, 0.0045]` (1/10/30/45 bps) for priority batch
  - Add `TC_BP_PHASE2 = [0.0005, 0.0015, 0.0025, 0.0035, 0.0040]` (5/15/25/35/40 bps) for fill-in batch
  - Modify `ContractEnv.__init__()` in `drl_shared/state_space.py` to accept optional `bp` parameter (default `BP` from config, preserves backward compat)
  - Modify `ContractEnv.step()` line 220: use `self.bp` instead of hardcoded `BP`
  - Add `--tc-bp` CLI argument to `train_dqn_walkforward.py` (float, default=BP from config)
  - Pass `--tc-bp` value through to `ContractEnv(bp=tc_bp)` when creating the environment
  - Pass `--tc-bp` value into `checkpoint_metadata()` via `extra` dict as `reward_spec.bp`

  **Must NOT do**:
  - Do NOT change `compute_eq4_reward()` formula — already accepts `bp` parameter
  - Do NOT change `config.py` — BP stays at 0.0020 as the paper default
  - Do NOT break existing training paths that don't use `--tc-bp`

  **Recommended Agent Profile**:
  > Select category + skills based on task domain.
  - **Category**: `quick`
    - Reason: Scoped plumbing across 3 files, no new logic, just adding parameter propagation
  - **Skills**: `[]`
    - All changes are straightforward Python parameter passing — no special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4, 5, 6, 7)
  - **Blocks**: Tasks 2, 3, 4, 5, 6, 7
  - **Blocked By**: None (can start immediately)

  **References**:
  - `drl_shared/state_space.py:179-226` — `ContractEnv.__init__` and `step` where `bp=BP` is hardcoded at line 220; change to `self.bp`
  - `drl_shared/state_space.py:109-141` — `compute_eq4_reward(bp=BP)` — function already accepts `bp` parameter, just needs caller to pass it
  - `drl/dqn/train/train_dqn_walkforward.py:600-650` — CLI argument parsing and model training loop where environment is created
  - `drl/dqn/spec.py:173-258` — `checkpoint_metadata()` where `reward_spec` is assembled — add `bp` field

  **Acceptance Criteria**:

  **QA Scenarios**:

  ```
  Scenario: --tc-bp flag accepted and recorded in manifest
    Tool: Bash
    Steps:
      1. Run: python drl/dqn/train/train_dqn_walkforward.py --tc-bp 0.0010 --help
      2. Assert: exit code 0, output contains "--tc-bp"
      3. Run dry training (1 episode): python drl/dqn/train/train_dqn_walkforward.py --tc-bp 0.0010 --asset Commodity --round 1 --seed 42 --episodes 1 --device cpu
      4. Assert: exit code 0, model directory created
      5. Read the generated manifest.json: assert reward_spec.bp == 0.0010
    Expected Result: Training launches with bp=0.0010, manifest records it
    Failure Indicators: Crash, wrong bp in manifest, directory not created
    Evidence: .sisyphus/evidence/task-1-cli-manifest.txt

  Scenario: Default (no --tc-bp) uses config BP=0.0020
    Tool: Bash
    Steps:
      1. Run: python drl/dqn/train/train_dqn_walkforward.py --asset Commodity --round 1 --seed 99 --episodes 1 --device cpu
      2. Assert: exit code 0
      3. Read manifest.json: assert reward_spec.bp == 0.0020
    Expected Result: Backward compat preserved — default uses config BP
    Evidence: .sisyphus/evidence/task-1-default-bp.txt
  ```

  **Commit**: NO (not yet — part of Wave 1 batch commit)

- [ ] 2. Model Directory BP Encoding — `make_run_id`, `find_latest_bundle`

  **What to do**:
  - Modify `make_run_id()` in `logging_utils.py` to accept optional `bp` parameter: `make_run_id(bp=None)` → returns `f"bp{int(bp*10000)}_{timestamp}"` when bp is provided, else `timestamp`
  - Modify `train_dqn_walkforward.py` line 642: `run_id = make_run_id(bp=tc_bp)` then `run_id = f"{run_id}_s{seed}"`
  - Result: `bp20_20260506T120000_s42` format
  - Update `find_latest_bundle()` in `generate_ensemble_table2.py` (line 65-84) to accept optional `bp` filter parameter
  - Update `generate_ensemble_table2.py` model discovery to scan for `bp{XX}_*_s{seed}` pattern
  - Update `spec.py::model_bundle_root()` if needed for bp-prefixed paths

  **Must NOT do**:
  - Do NOT break existing model loading for BP=20 non-prefixed dirs (handle both formats during migration)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: String manipulation and directory scanning changes — straightforward
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: Tasks 5, 8-19
  - **Blocked By**: Task 1 (needs BP plumbing first)

  **References**:
  - `drl/dqn/logging_utils.py:13-14` — `make_run_id()` function to modify
  - `drl/dqn/train/train_dqn_walkforward.py:642-646` — where run_id is constructed with seed suffix
  - `drl/dqn/reports/generate_ensemble_table2.py:65-84` — `find_latest_bundle()` scanning logic
  - `drl/dqn/spec.py:102-122` — `model_bundle_root()`, `resolve_model_bundle()`, `_valid_bundle_dirs()`

  **Acceptance Criteria**:

  **QA Scenarios**:

  ```
  Scenario: bp-aware run_id generates correct directory
    Tool: Bash
    Steps:
      1. python3 -c "from drl.dqn.logging_utils import make_run_id; rid = make_run_id(bp=0.0015); assert rid.startswith('bp15_'), f'got {rid}'"
      2. python3 -c "from drl.dqn.logging_utils import make_run_id; rid = make_run_id(bp=0.0020); assert rid.startswith('bp20_'), f'got {rid}'"
      3. python3 -c "from drl.dqn.logging_utils import make_run_id; rid = make_run_id(); assert '_' not in rid or rid.startswith('20'), f'got {rid}'"
    Expected Result: bp prefix correctly encodes BP in bps × 100
    Evidence: .sisyphus/evidence/task-2-runid.txt

  Scenario: find_latest_bundle filters by BP
    Tool: Bash
    Steps:
      1. After migrating existing models (Task 5), run: python3 -c "
         from drl.dqn.reports.generate_ensemble_table2 import find_latest_bundle
         bundle = find_latest_bundle('Commodity', 'r1', 42)
         assert bundle is not None, 'bundle not found'
         assert 'bp20_' in str(bundle), f'expected bp20_ prefix in {bundle}'
         print('OK:', bundle)
         "
    Expected Result: Bundle found with bp20_ prefix
    Evidence: .sisyphus/evidence/task-2-find-bundle.txt
  ```

  **Commit**: NO (batch with Wave 1)

- [ ] 3. Manifest BP Recording — `checkpoint_metadata`, `train_config.json`

  **What to do**:
  - Update `spec.py::checkpoint_metadata()` to include `bp` in `reward_spec` dict (currently only has formula string at line 242)
  - Add `"bp": float_value` to the `reward_spec` dict
  - Add `"bp_bps": int_value` for human readability (e.g., 20)
  - Update `train_dqn_walkforward.py` to pass `tc_bp` through to `checkpoint_metadata(extra={'tc_bp': tc_bp})` 
  - Verify `manifest.json` written at training time includes the bp value
  - Update `spec.py::maybe_load_manifest_for_checkpoint()` to return bp value from manifest

  **Must NOT do**:
  - Do NOT change existing manifest.json schema compatibility — add fields, don't remove

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: JSON schema update — adding fields to existing dict
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: Tasks 4, 16-19
  - **Blocked By**: Task 1

  **References**:
  - `drl/dqn/spec.py:238-243` — `reward_spec` dict to augment with bp field
  - `drl/dqn/spec.py:148-162` — `load_manifest()`, `maybe_load_manifest_for_checkpoint()`
  - `drl/dqn/train/train_dqn_walkforward.py:647-660` — where manifest/metadata is written

  **Acceptance Criteria**:

  **QA Scenarios**:

  ```
  Scenario: manifest.json contains bp after training
    Tool: Bash
    Steps:
      1. Run 1-episode training: python drl/dqn/train/train_dqn_walkforward.py --tc-bp 0.0010 --asset Commodity --round 1 --seed 42 --episodes 1 --device cpu
      2. Find the generated manifest.json in the model directory
      3. python3 -c "import json; m=json.load(open('...')); assert m['reward_spec']['bp']==0.0010; assert m['reward_spec']['bp_bps']==10"
    Expected Result: bp=0.0010 and bp_bps=10 recorded in manifest
    Evidence: .sisyphus/evidence/task-3-manifest-bp.json
  ```

  **Commit**: NO (batch with Wave 1)

- [ ] 4. Backtest BP-Aware Pipeline — `generate_ensemble_table2.py`

  **What to do**:
  - Add `--tc-bp` CLI argument to `generate_ensemble_table2.py` (float) to select which BP level to backtest
  - Modify `find_latest_bundle()` to accept `bp` filter parameter
  - Modify model discovery to scan for `bp{XX}_*_s{seed}` pattern
  - Modify `backtest_contract_ensemble()` — read `bp` from model's `manifest.json`, pass to `compute_contract_returns_from_positions(prepared, positions, sigma_tgt, bp=bp_value)`
  - SAVE ensemble R data to `ensemble_table2_bp/{asset}/bp{XX}/top5_ensemble_R.npz` (new directory per BP)
  - SAVE metrics to `ensemble_table2_bp/{asset}/bp{XX}/metrics.json`
  - Add `--all-bp` flag to run all 7 BP levels for a single asset sequentially

  **Must NOT do**:
  - Do NOT hardcode `BP` in any new code path — always read from manifest or CLI
  - Do NOT overwrite existing `ensemble_table2/` results (BP=20 goes to new `bp20/` subdir)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: CLI flag + directory routing + manifest reading — straightforward
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: Tasks 16-21
  - **Blocked By**: Tasks 1, 2, 3

  **References**:
  - `drl/dqn/reports/generate_ensemble_table2.py:65-84` — `find_latest_bundle()`
  - `drl/dqn/reports/generate_ensemble_table2.py:157-272` — `backtest_contract_ensemble()` where `compute_contract_returns_from_positions` is called at line 260
  - `drl/dqn/reports/generate_ensemble_table2.py:319-376` — `run_asset_ensemble_backtest()` — add BP-aware directory routing
  - `baseline_run.py:434-492` — `compute_contract_returns_from_positions()` — add `bp` parameter
  - `baseline_run.py:478` — where `BP * prices[...]` is used — change to use `bp` parameter

  **Acceptance Criteria**:

  **QA Scenarios**:

  ```
  Scenario: Backtest with --tc-bp reads manifest bp
    Tool: Bash
    Steps:
      1. After BP=20 models migrated (Task 5): python generate_ensemble_table2.py --tc-bp 0.0020 --asset Commodity --round 1
      2. Assert: exit code 0
      3. Check: python3 -c "import json; m=json.load(open('ensemble_table2_bp/Commodity/bp20/metrics.json')); print(m['metrics']['Sharpe'])"
      4. Assert: Sharpe value matches existing Table 2 Commodity Sharpe (-0.964) within ±0.05
    Expected Result: BP=20 backtest matches existing results
    Evidence: .sisyphus/evidence/task-4-bp20-match.txt

  Scenario: Backtest enforces bp from manifest (not CLI)
    Tool: Bash
    Steps:
      1. Try: python generate_ensemble_table2.py --tc-bp 0.0030 --asset Commodity (but models are BP=20)
      2. Assert: warning or error about missing BP=30 models
      3. Assert: does NOT silently use BP=20 models with BP=30 cost
    Expected Result: Model bp and backtest bp must match
    Evidence: .sisyphus/evidence/task-4-bp-mismatch.txt
  ```

  **Commit**: NO (batch with Wave 1)

- [ ] 5. Migrate Existing BP=20 Models — DRY-RUN FIRST, then execute

  **What to do**:
  - **Step 5a (DRY-RUN)**: Test migration on ONE model (e.g., Commodity r1 s42)
    - Copy directory: `cp -r {timestamp}_s42 bp20_{timestamp}_s42_test`
    - Update manifest.json in copy: add `"bp": 0.0020, "bp_bps": 20`
    - Verify: `find_latest_bundle('Commodity', 'r1', 42)` finds the new path
    - If OK → delete test copy, proceed to full migration
    - If FAIL → fix code, retry
  - **Step 5b (EXECUTE)**: After dry-run confirmed:
    - Enumerate ALL existing model dirs under `models/{Commodity,Equity_Index,Fixed_Income,Forex}/r{1,2}/`
    - Validate each: read `manifest.json` → verify gamma=0.6, episodes=100
    - Rename dirs: `{timestamp}_s{seed}` → `bp20_{timestamp}_s{seed}`
    - Update `manifest.json` in each to add `"bp": 0.0020, "bp_bps": 20`
    - Generate migration log: `drl/dqn/models/migration_bp20.log` listing old→new paths
    - Verify NO training jobs are running (would corrupt)
  - Count: expect 60 models (3 assets × 2 rounds × 10 seeds). Handle partial Forex.

  **Must NOT do**:
  - Do NOT retrain any BP=20 model — only rename and update manifest
  - Do NOT execute full migration before dry-run test passes
  - Do NOT change checkpoint.pt files

  **Recommended Agent Profile**: `quick`, `[]`

  **Parallelization**: Wave 1 | **Blocks**: Tasks 4, 8-30 | **Blocked By**: Task 2

  **QA Scenarios**:

  ```
  Scenario: Dry-run migration succeeds on one model
    Tool: Bash
    Steps:
      1. Copy one model dir: cp -r models/Commodity/r1/{timestamp}_s42 models/Commodity/r1/bp20_{timestamp}_s42_test
      2. Update its manifest.json with bp=0.0020, bp_bps=20
      3. python3 -c "
         from drl.dqn.reports.generate_ensemble_table2 import find_latest_bundle
         b = find_latest_bundle('Commodity', 'r1', 42)
         assert b is not None and 'bp20_' in str(b)
         "
      4. Cleanup: rm -rf models/Commodity/r1/bp20_*_test
    Expected Result: find_latest_bundle discovers bp20_ prefixed path
    Evidence: .sisyphus/evidence/task-5a-dry-run.txt

  Scenario: Full migration produces correct count
    Tool: Bash
    Steps:
      1. Run migration script
      2. ls models/Commodity/r1/bp20_*/checkpoint.pt | wc -l → assert 10
      3. ls models/Equity_Index/r1/bp20_*/checkpoint.pt | wc -l → assert 10
      4. ls models/Fixed_Income/r1/bp20_*/checkpoint.pt | wc -l → assert 10
      5. python3 -c "import json; m=json.load(open('models/Commodity/r1/bp20_*/manifest.json')); assert m['reward_spec']['bp']==0.0020"
    Expected Result: All models renamed, manifests updated
    Evidence: .sisyphus/evidence/task-5b-migration.log
  ```

  **Commit**: YES — `refactor(dqn): migrate existing models to bp20_ prefix`

- [ ] 6. BP=0 Edge Case Test + `compute_eq4_reward` Validation

  **What to do**:
  - Write unit test: `compute_eq4_reward(bp=0.0, ...)` → `tc=0`, `reward=gross`
  - Write unit test: `compute_eq4_reward(bp=0.001, ...)` → `tc > 0` when position changes, `tc=0` when position unchanged
  - Write unit test: `compute_contract_returns_from_positions(..., bp=0.0)` → `tc_cost` array all zeros
  - Verify no division by zero at any BP level (0-30 bps)
  - Verify formula: `tc = bp * prices[idx-1] * abs(action*vol_scale - prev_action*vol_scale_prev)` works correctly at all BP levels
  - Save tests to `tests/test_tc_bp_edge_cases.py`

  **Must NOT do**:
  - Do NOT modify the formula — only test it

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Unit tests for existing function — no implementation changes
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: None (validation only)
  - **Blocked By**: Task 1

  **References**:
  - `drl_shared/state_space.py:109-141` — `compute_eq4_reward()` function to test
  - `baseline_run.py:434-492` — `compute_contract_returns_from_positions()` to test
  - `tests/test_drl_v2.py:350-407` — existing tests for compute_eq4_reward (line 367, 388, 401)

  **Acceptance Criteria**:

  **QA Scenarios**:

  ```
  Scenario: bp=0 produces zero transaction cost
    Tool: Bash (pytest)
    Steps:
      1. python3 -c "
         from drl_shared.state_space import compute_eq4_reward
         import numpy as np
         prices = np.array([100.0, 101.0, 102.0])
         returns = np.array([0.0, 1.0, 1.0])
         sigma = np.array([0.5, 0.6, 0.7])
         reward, gross, tc, vol = compute_eq4_reward(prices, returns, sigma, idx=2, action=1.0, prev_action=0.0, sigma_tgt=0.058, bp=0.0)
         assert tc == 0.0, f'tc should be 0, got {tc}'
         assert reward == gross, f'reward should equal gross when bp=0'
         print('PASS: bp=0, tc=0, reward=gross')
         "
      2. Assert: prints PASS, exit code 0
    Expected Result: Zero cost produces zero TC, reward equals gross PnL
    Evidence: .sisyphus/evidence/task-6-bp0.txt

  Scenario: bp=0.001 produces positive tc when position changes
    Tool: Bash (pytest)
    Steps:
      1. python3 -c "
         from drl_shared.state_space import compute_eq4_reward
         import numpy as np
         prices = np.array([100.0, 101.0])
         returns = np.array([0.0, 1.0])
         sigma = np.array([0.5, 0.6])
         # Position changes from 0 to 1 → TC > 0
         _, _, tc1, _ = compute_eq4_reward(prices, returns, sigma, idx=1, action=1.0, prev_action=0.0, sigma_tgt=0.058, bp=0.001)
         # Position stays at 1 → TC = 0
         _, _, tc2, _ = compute_eq4_reward(prices, returns, sigma, idx=1, action=1.0, prev_action=1.0, sigma_tgt=0.058, bp=0.001)
         assert tc1 > 0, f'tc1 should be > 0, got {tc1}'
         assert tc2 == 0, f'tc2 should be 0 when position unchanged, got {tc2}'
         print(f'PASS: tc_change={tc1:.6f}, tc_nochange={tc2}')
         "
      2. Assert: PASS, tc_change > 0, tc_nochange == 0
    Expected Result: TC correctly detects position changes
    Evidence: .sisyphus/evidence/task-6-tc-change.txt
  ```

  **Commit**: NO (batch with Wave 1)

- [ ] 7. Orchestration Script + Job Tracker (with Background Execution)

  **What to do**:
  - Create `drl/dqn/train/train_exhibit5.py` — thin orchestration script
  - Loop over: 4 assets × 2 rounds × 10 BP levels × 10 seeds = 800 potential jobs
  - For each job: check if model dir already exists with valid checkpoint.pt → skip if exists
  - **BACKGROUND EXECUTION**: Launch training via `nohup` + `&` or `tmux`:
    ```bash
    nohup python drl/dqn/train/train_dqn_walkforward.py \
      --tc-bp {bp} --asset {asset} --round {r} --seed {seed} \
      --device cuda > logs/train_bp{bp}_{asset}_r{r}_s{seed}.log 2>&1 &
    ```
  - Write job status to `drl/dqn/reports/ensemble_table2_bp/training_jobs.json`:
    ```json
    {
      "asset": "Commodity", "round": 1, "bp": 30, "seed": 42,
      "status": "running", "pid": 12345, "log": "logs/train_bp30_Commodity_r1_s42.log",
      "started_at": "2026-05-06T12:00:00"
    }
    ```
  - Statuses: `pending`, `running`, `complete` (checkpoint.pt exists), `failed` (log has error)
  - Support resuming: re-run script, skips `complete`, retries `failed`
  - GPU management: limit to 4 concurrent per GPU (use `CUDA_VISIBLE_DEVICES` or semaphore)
  - After all jobs: run `python training_status.py` to show "X/Y complete, Z failed, W running"

  **Must NOT do**:
  - Do NOT overwrite checkpoint.pt of completed jobs
  - Do NOT launch without nohup (training dies when terminal closes)

  **Recommended Agent Profile**: `quick`, `[]`

  **Parallelization**: Wave 1 | **Blocks**: Tasks 8-30 | **Blocked By**: Tasks 1,2,3,5b

  **QA Scenarios**:

  ```
  Scenario: Orchestration reports status, nohup jobs survive disconnect
    Tool: Bash
    Steps:
      1. Run: python drl/dqn/train/train_exhibit5.py --phase 1 --dry-run
      2. Assert: exit 0, output "320 jobs (Phase 1), XX skip (BP=20 exists), YY new"
      3. Check training_jobs.json has 320 entries with status "pending"
      4. Run 1 real job: python train_exhibit5.py --phase 1 --asset Commodity --dry-run=False
      5. Assert: nohup process launched, log file created
      6. Wait 30s, check: process still running (status="running" in JSON)
    Expected Result: Jobs tracked, survive terminal disconnect
    Evidence: .sisyphus/evidence/task-7-nohup-test.txt
  ```

  **Commit**: NO (batch with Wave 1)

- [ ] 7b. Exhibit 5 Figure Prototype — Write + Test with BP=20 Data NOW

  **What to do**:
  - Create `drl/dqn/figures/exhibit5_tc_impact.py` IMMEDIATELY (don't wait for training)
  - Use BP=20 data from existing `ensemble_table2/Commodity/bp20/top5_ensemble_R.npz` (after migration)
  - Panel A: Sharpe Ratio vs BP — plot BP=20 as single data point with 4 asset lines (placeholder)
  - Panel B: Average Daily Cost Per Contract vs BP — same
  - The figure should work with 1 data point first, then auto-update as more BP levels complete
  - Design: reads `exhibit5_metrics_summary.json` and `exhibit5_cost_summary.json` 
  - Auto-detects: if only BP=20 data exists, plots single point + "Phase 1 in progress" watermark
  - As Phase 1 completes, re-run to get 5 data points (1/10/20/30/45)
  - Final: 10 data points after Phase 2

  **Why now**: We already have BP=20 results. No need to wait 2 weeks of training to see if the figure code works. This de-risks the plotting and validates the data pipeline.

  **Must NOT do**:
  - Do NOT add grid lines
  - Do NOT hardcode data — always read from JSON

  **Recommended Agent Profile**: `visual-engineering`, `[]`

  **Parallelization**: Wave 1 (runs in parallel with Task 7 — no dependency on training)
  | **Blocks**: None | **Blocked By**: Task 5b (need BP=20 data migrated)

  **Acceptance Criteria**:

  **QA Scenarios**:

  ```
  Scenario: Prototype figure renders with BP=20 data only
    Tool: Bash
    Steps:
      1. Run: python drl/dqn/figures/exhibit5_tc_impact.py
      2. Assert: exit 0, figures/exhibit5_tc_impact.png exists
      3. python3 -c "
         from PIL import Image
         img = Image.open('figures/exhibit5_tc_impact.png')
         assert img.size[0] > 800
         print(f'OK: {img.size}')
         "
      4. Assert: figure has 2 panels (Panel A: Sharpe, Panel B: Cost)
      5. Assert: 4 lines visible (one per asset), each has at least 1 data point (BP=20)
    Expected Result: Working 2-panel figure with BP=20 data
    Evidence: .sisyphus/evidence/task-7b-prototype.png
  ```

  **Commit**: YES — `feat(dqn): Exhibit 5 figure prototype (tested with BP=20 data)`

- [ ] 8. Train Commodity Round 1 Phase 1 — 1/10/30/45 bps × 10 seeds = 40 jobs

  **What to do**:
  - Run `python drl/dqn/train/train_exhibit5.py --phase 1 --asset Commodity --round 1`
  - Each job launched via **nohup** (survives terminal disconnect):
    ```
    nohup python train_dqn_walkforward.py --tc-bp 0.0030 --asset Commodity \
      --round 1 --seed 42 --device cuda > logs/train_bp30_Commodity_r1_s42.log 2>&1 &
    ```
  - BP=20 jobs skipped (reuse existing models from Task 5b)
  - Expected output: ~40 new models (4 non-20 BP levels × 10 seeds)
  - Monitor: tail training logs at `logs/train_bp{XX}_{asset}_r{round}_s{seed}.log`
  - After each job: verify checkpoint.pt exists, update `training_jobs.json` status to `complete`
  - If a job fails: status → `failed`, error recorded in log → retry on next run

  **Background Execution Protocol** (ALL training tasks):
  - ALWAYS use `nohup ... &` or `tmux new-session -d`
  - NEVER run training in foreground (dies when terminal closes)
  - Log every job to `logs/train_bp{XX}_{asset}_r{round}_s{seed}.log`
  - PID recorded in `training_jobs.json` for monitoring/killing

  **Must NOT do**:
  - Do NOT retrain BP=20
  - Do NOT run in foreground

  **Recommended Agent Profile**: `unspecified-low`, `[]`
  **Parallelization**: Wave 2 | **Blocks**: Task 16 | **Blocked By**: Tasks 1,2,3,5b,7

  **QA Scenarios**:
  ```
  Scenario: Phase 1 training job survives disconnect
    Tool: Bash
    Steps:
      1. Launch 1 test job with nohup
      2. Verify PID recorded in training_jobs.json
      3. Wait 30s, check: ps -p {pid} shows process still running
      4. tail logs/train_bp30_Commodity_r1_s42.log → shows training progress
    Expected Result: Process survives, log file grows
    Evidence: .sisyphus/evidence/task-8-nohup-survive.txt
  ```

  **Commit**: NO (model artifacts)

- [ ] 9. Train Commodity Round 2 Phase 1 — 40 jobs
  Same as Task 8 for Commodity r2. **nohup required**.
  **Parallel Group**: Wave 2 | **Blocks**: Task 16 | **Agent**: `unspecified-low`, `[]`

- [ ] 10. Train Equity Index Round 1 Phase 1 — 40 jobs
  Same as Task 8 for Equity Index r1. **nohup required**.
  **Parallel Group**: Wave 2 | **Blocks**: Task 17 | **Agent**: `unspecified-low`, `[]`

- [ ] 11. Train Equity Index Round 2 Phase 1 — 40 jobs
  Same as Task 8 for Equity Index r2. **nohup required**.
  **Parallel Group**: Wave 2 | **Blocks**: Task 17 | **Agent**: `unspecified-low`, `[]`

- [ ] 12. Train Fixed Income Round 1 Phase 1 — 40 jobs
  Same as Task 8. **nohup required**. **Wave 3** | **Blocks**: Task 18 | **Agent**: `unspecified-low`, `[]`

- [ ] 13. Train Fixed Income Round 2 Phase 1 — 40 jobs
  Same as Task 8. **nohup required**. **Wave 3** | **Blocks**: Task 18 | **Agent**: `unspecified-low`, `[]`

- [ ] 14. Train Forex Round 1 Phase 1 — 40 jobs
  Same as Task 8. **nohup required**. **Wave 3** | **Blocks**: Task 19 | **Agent**: `unspecified-low`, `[]`

- [ ] 15. Train Forex Round 2 Phase 1 — 40 jobs
  Same as Task 8. **nohup required**. **Wave 3** | **Blocks**: Task 19 | **Agent**: `unspecified-low`, `[]`

- [ ] 16. Ensemble Backtest — Commodity Phase 1 (1/10/30/45 bps)
  **Wave 3.5** | **Blocks**: Task 20,21 | **Blocked By**: Tasks 8,9
  Same ensemble pattern as original Task 16 (Q-value top-5). 
  Save to `ensemble_table2_bp/Commodity/bp{XX}/`. **Agent**: `unspecified-low`, `[]`

- [ ] 17-19. Ensemble Phase 1 — Equity, FI, Forex
  Same as Task 16 for remaining assets. **Wave 3.5**. **Agent**: `unspecified-low`, `[]`

- [ ] 20. Compute Sharpe + Cost — Phase 1 (1/10/20/30/45 bps)
  **Wave 3.5** | **Blocked By**: Tasks 16-19
  Merge BP=20 data (from Task 5b) with Phase 1 results. Save `exhibit5_metrics_phase1.json`.
  Test: BP=20 Sharpe matches Table 2. **Agent**: `quick`, `[]`

- [ ] 21. Update Exhibit 5 Figure — Phase 1
  **Wave 3.5** | **Blocked By**: Task 20
  Re-run `exhibit5_tc_impact.py` (built in Task 7b). Now has 5 data points per asset (1/10/20/30/45).
  Verify: cost monotonically increases, Sharpe generally decreases.

- [ ] 22. VALIDATION GATE — Phase 1 Review
  **BLOCKING**: User confirms cost effect is visible BEFORE Phase 2 training starts.
  Check: Exhibit 5 figure shows clear trend (cost impacts learning behavior).
  If effect invisible → debug before spending compute on Phase 2.

- [ ] 23-30. Phase 2 Training — 5/15/25/35/40 bps × 4 assets × 2 rounds (400 jobs)
  Same pattern as Tasks 8-15. **nohup required**. **Waves 4-5**. **Agent**: `unspecified-low`, `[]`

- [ ] 31-34. Ensemble Phase 2
  Same pattern as Tasks 16-19. **Wave 6**. **Agent**: `unspecified-low`, `[]`

- [ ] 35. Compute Sharpe + Cost — ALL 10 BP levels
  Merge Phase 1, Phase 2, BP=20. Save `exhibit5_metrics_full.json`. **Wave 7**. **Agent**: `quick`, `[]`

- [ ] 36. Generate FINAL Exhibit 5
  Re-run figure script with all 10 data points. 
  4 asset lines per panel, 10 x-axis ticks (1/5/10/15/20/25/30/35/40/45).
  Save final `exhibit5_tc_impact.png`. **Wave 7**. **Agent**: `visual-engineering`, `[]`

---

## Final Verification Wave

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. Verify: BP encoded in model paths, manifest contains bp, backtest reads bp from manifest, existing BP=20 models reused, Exhibit 5 figure has 2 panels × 4 asset lines. Check evidence files exist.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Verify no broken imports, `--tc-bp` flag works, training launches successfully, backtest runs without errors. Check for: hardcoded BP references remaining, `from config import BP` in new code paths.
  Output: `Build [PASS/FAIL] | Flag [PASS/FAIL] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Verify: (1) Train 1 seed at BP=10 → model dir exists with bp10_ prefix. (2) Backtest that seed → manifest.bp matches. (3) BP=20 results match existing Table 2. (4) Exhibit 5 PNG has correct layout.
  Output: `Scenarios [N/N pass] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  Verify: no hyperparameter changes, no new asset classes, no refactoring of unrelated code, BP plumbing only touches necessary files.
  Output: `Tasks [N/N compliant] | VERDICT`

---

## Commit Strategy

- **1**: `feat(dqn): add --tc-bp parameter to training pipeline` — spec.py, state_space.py, train_dqn_walkforward.py
- **5**: `refactor(dqn): migrate existing models to bp20_ prefix` — model directory rename
- **7b**: `feat(dqn): Exhibit 5 figure prototype (tested with BP=20 data)` — exhibit5_tc_impact.py
- **8-15, 23-30**: training jobs (model artifacts, not committed code)
- **16-19, 31-34**: ensemble backtest — generate_ensemble_table2.py updates
- **36**: `feat(dqn): final Exhibit 5 with full 10 BP levels` — exhibit5_tc_impact.py update

---

## Success Criteria

### Verification Commands
```bash
# 1. BP plumbing functional + nohup training survives
python drl/dqn/train/train_dqn_walkforward.py --tc-bp 0.0010 --asset Commodity --round 1 --seed 42 --episodes 1 --device cpu 2>&1 | grep "bp=0.001"
nohup python drl/dqn/train/train_dqn_walkforward.py --tc-bp 0.0030 --asset Commodity --round 1 --seed 42 --episodes 1 --device cpu > /tmp/test_nohup.log 2>&1 &
# After disconnect/reconnect: ps aux | grep train_dqn  → process still running

# 2. Model directory includes bp prefix (after migration)
ls drl/dqn/models/Commodity/r1/bp20_*/checkpoint.pt | wc -l  # → 10

# 3. Manifest contains bp
python3 -c "import json; m=json.load(open('drl/dqn/models/Commodity/r1/bp20_*/manifest.json')); assert m['reward_spec']['bp']==0.0020"

# 4. Backtest uses model bp (BP=20 matches existing)
python3 -c "
import numpy as np
d = np.load('drl/dqn/reports/ensemble_table2_bp/Commodity/bp20/top5_ensemble_R.npz')
assert abs(np.std(d['portfolio_returns']) - 0.97) < 0.05
"

# 5. Exhibit 5 prototype renders with BP=20 data
python3 -c "
from PIL import Image
img = Image.open('drl/dqn/figures/exhibit5_tc_impact.png')
assert img.size[0] > 800
"

# 6. Phase 1 figure shows 5 data points (1/10/20/30/45) per asset
python3 -c "
import json
m = json.load(open('drl/dqn/reports/ensemble_table2_bp/exhibit5_metrics_phase1.json'))
assert len(m['Commodity']) == 5  # 5 BP levels
"

# 7. Final figure shows 10 data points (full paper range)
python3 -c "
import json
m = json.load(open('drl/dqn/reports/ensemble_table2_bp/exhibit5_metrics_full.json'))
assert len(m['Commodity']) == 10
"
```

### Final Checklist
- [ ] All "Must Have" present (BP in path, BP in manifest, backtest uses manifest BP)
- [ ] All "Must NOT Have" absent (no hyperparameter changes, no formula changes)
- [ ] 560 training jobs tracked, all complete or accounted for
- [ ] Exhibit 5 figure has Panel A (Sharpe vs BP) + Panel B (Cost vs BP)
- [ ] BP=20 results match existing Table 2 (within tolerance)
