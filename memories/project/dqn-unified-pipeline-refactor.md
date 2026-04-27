# DQN Unified Data Pipeline Refactor (2026-04-27)

## Summary
Refactored DQN training/backtest to use a single data pipeline with date-validated feature loading.

## Changes
1. **`drl_shared/data_loader.py`** (NEW) — unified feature loader
   - `load_npz()`: raw load with validation (NaN/Inf/duplicates/dates)
   - `get_train_slice()`: train period, 90/10 split
   - `get_test_slice()`: test period slice
   - All date-based, raises on mismatch

2. **`drl_shared/prepare_features.py`** — npz now covers train+test (was train only)

3. **`_train_single_contract.py`** — uses `get_train_slice()` instead of manual date slicing

4. **`train_dqn_walkforward.py`** — added train/eval mode switching, removed hardcoded 7→FEATURE_DIM, run_id 加 seed 后缀防覆盖

5. **`backtest_test_period.py`** — uses `get_test_slice()` instead of `load_clc_full`

6. **Unit tests**: `drl/dqn/tests/test_data_loader.py` — 18/18 Forex contracts pass
   - 路径: `python3 drl/dqn/tests/test_data_loader.py`
   - 验证: train/test 无重叠、日期范围正确、无 NaN/Inf、无重复日期、feature dim=9、非零 returns/sigma、WARMUP offset 正确

## Deprecated → archive/deprecated/
- prepare_dqn_data.py, prepare_dqn_walkforward.py
- train_dqn_paper_aligned.py, strategy_dqn.py, train_all_assets.py
- pipeline.py, backtest_test_period_fast.py, backtest_test_parallel.py
- run_all_percontract.py
- 13+ old scripts → archive/old_scripts/

## Key Decisions
- Feature preset "structural_38" = 9-dim features (not 38)
- npz covers full range (burn-in + train + test), code slices by date
- Train/test split via RETRAIN_ROUNDS dates, validated at load time
- FEATURE_DIM=9 (1 price norm + 4 returns + 3 MACD + 1 RSI)

## 性能瓶颈
- **训练慢的根因是 CPU 串行循环，不是 GPU**
- 每个 cycle 遍历 9 合约 × ~2500 steps，每步 Python `env.step()` + `agent.act()`
- GPU 只做前向/反向传播，模型小（32K params），GPU 利用率 86% 但显存只用 1.3%
- 5 seed 并行时 CPU load ~5，每个 seed 跑完 100 cycles 需 ~3 小时
- 可能优化：将 env simulation 批量化/C 化，或减少合约/episodes
