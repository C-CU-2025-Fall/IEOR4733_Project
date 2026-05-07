"""
reeval_route_b_sigma064.py
==========================
使用已有的 Route B checkpoints (.pt)，以 sigma_target=0.064 重新评估，
覆盖 regime_detection/results/ 下的 pnl_routeB_*.csv。

说明
----
sigma_target 只影响仓位缩放幅度（scaled_pos = sigma_target / sigma_t * action），
不影响 actor 学到的信号方向。因此无需重训练，直接重新推理即可得到
与 A2C（sigma=0.064）同量纲的 PnL 序列。

用法
----
    cd /path/to/IEOR4733_Project
    source .venv/bin/activate
    python regime_detection/reeval_route_b_sigma064.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "rl_models"))

from config import ASSET_CLASSES
from rl_models.a2c_model import PaperA2CTrainer, DEVICE
from regime_detection.route_b_train import (
    PERIODS,
    FEATURE_COLS_12,
    _load_rad_data_for_a2c,
    _build_augmented_state_dict,
    _evaluate,
    predict_regime_soft_probs,
    detect_regimes_for_asset_class_timeseries,
)
from regime_detection.timeseries_fft_regime import (
    detect_regimes_for_asset_class_timeseries,
    predict_regime_soft_probs,
)
from data_loader import load_clc_full

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── 修改后的 sigma_target ────────────────────────────────────────────────────
NEW_SIGMA_TARGET = 0.064          # 对齐 A2C 基准

CKPT_DIR  = PROJECT_ROOT / "rl_models"
PNL_DIR   = PROJECT_ROOT / "regime_detection" / "results"
PNL_DIR.mkdir(exist_ok=True)

TARGET_PERIODS = ["period_1", "period_2"]

all_tickers = [t for v in ASSET_CLASSES.values() for t in v]
ASSET_CLASSES_WITH_ALL = dict(ASSET_CLASSES)
ASSET_CLASSES_WITH_ALL["All"] = all_tickers

TARGET_CLASSES = ["Commodity", "Equity Index", "Fixed Income", "Forex", "All"]


def _load_clc(tickers, data_start):
    clc = {}
    for t in tickers:
        try:
            df = load_clc_full(t, start_date=data_start)
            if df is not None:
                clc[t] = df
        except Exception as e:
            logger.debug(f"skip {t}: {e}")
    return clc


def main():
    print(f"\n{'='*70}")
    print(f"Route B Re-evaluation  |  sigma_target = {NEW_SIGMA_TARGET}")
    print(f"{'='*70}\n")

    summary = []

    for period_name in TARGET_PERIODS:
        cfg = PERIODS[period_name]
        train_start, train_end = cfg["train"]
        test_start,  test_end  = cfg["test"]
        data_start             = cfg["data_start"]

        for asset_class in TARGET_CLASSES:
            tickers = ASSET_CLASSES_WITH_ALL[asset_class]
            ac_slug = asset_class.replace(" ", "_")

            ckpt_path = CKPT_DIR / f"a2c_rb_{ac_slug}_{period_name}.pt"
            if not ckpt_path.exists():
                logger.warning(f"  ⚠️  Checkpoint not found: {ckpt_path.name}  — skipping")
                continue

            logger.info(f"\n[{period_name} | {asset_class}]")
            logger.info(f"  Loading checkpoint: {ckpt_path.name}")

            # ── 1. 加载模型 ──────────────────────────────────────────────────
            trainer = PaperA2CTrainer(n_features=12, device=DEVICE)
            trainer.load_checkpoint(str(ckpt_path))

            # ── 2. 重建 regime soft probs ────────────────────────────────────
            clc_data = _load_clc(tickers, data_start)
            train_regime_result = detect_regimes_for_asset_class_timeseries(
                clc_data=clc_data,
                asset_class_tickers=tickers,
                asset_class_name=asset_class,
                n_regimes=3,
                date_range=(train_start, train_end),
            )
            if train_regime_result is None:
                logger.error(f"  ❌ Regime detection failed, skipping {asset_class} {period_name}")
                continue

            full_regime_df = predict_regime_soft_probs(
                clc_data=clc_data,
                asset_class_tickers=tickers,
                asset_class_name=asset_class,
                trained_gmm=train_regime_result["gmm_model"],
                trained_scaler=train_regime_result["fft_scaler"],
                date_range=(data_start, test_end),
            )
            if full_regime_df is None:
                logger.error(f"  ❌ Soft probs failed, skipping {asset_class} {period_name}")
                continue

            # ── 3. 重建 12-dim 状态 ──────────────────────────────────────────
            feature_dict = _load_rad_data_for_a2c(tickers, data_start, test_end)
            _, test_sd = _build_augmented_state_dict(
                feature_dict=feature_dict,
                regime_df=full_regime_df,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )

            # ── 4. 用新 sigma_target 推理 ────────────────────────────────────
            metrics = _evaluate(
                actor=trainer.actor,
                test_state_dict=test_sd,
                sigma_target=NEW_SIGMA_TARGET,
            )

            sharpe  = metrics.get("portfolio_sharpe",  float("nan"))
            calmar  = metrics.get("portfolio_calmar",  float("nan"))
            ann_ret = metrics.get("portfolio_ann_ret", float("nan"))
            ann_std = metrics.get("portfolio_ann_std", float("nan"))
            logger.info(
                f"  Sharpe={sharpe:.3f}  Calmar={calmar:.3f}  "
                f"AnnRet={ann_ret:.4f}  AnnStd={ann_std:.4f}"
            )

            # ── 5. 保存 PnL CSV ──────────────────────────────────────────────
            portfolio_pnl = metrics.get("portfolio_pnl")
            if portfolio_pnl is not None and len(portfolio_pnl) > 0:
                csv_path = PNL_DIR / f"pnl_routeB_{period_name}_{ac_slug}.csv"
                pnl_df = pd.DataFrame({
                    "date":       portfolio_pnl.index,
                    "net_pnl":    portfolio_pnl.values,
                    "cum_wealth": 1.0 + portfolio_pnl.cumsum().values,
                })
                pnl_df.to_csv(csv_path, index=False)
                logger.info(f"  💾 Saved: {csv_path.name}  ({len(pnl_df)} rows)")
                summary.append((period_name, asset_class, sharpe, calmar, ann_ret))
            else:
                logger.warning(f"  ⚠️  portfolio_pnl is None for {asset_class} {period_name}")

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Summary  (sigma_target = {NEW_SIGMA_TARGET})")
    print(f"{'='*70}")
    print(f"{'Period':<12} {'Asset Class':<18} {'Sharpe':>8} {'Calmar':>8} {'AnnRet':>9}")
    print("-" * 58)
    for period, ac, sh, ca, ar in summary:
        print(f"{period:<12} {ac:<18} {sh:>8.3f} {ca:>8.3f} {ar:>9.4f}")

    print(f"\n✅ 完成。CSV 已保存至: {PNL_DIR}")
    print(f"   回到 strategies_comparison.ipynb 重新执行 Section 8 + 绘图 cell 即可。")


if __name__ == "__main__":
    main()
