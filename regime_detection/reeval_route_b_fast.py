"""
reeval_route_b_fast.py
======================
快速重新评估 Route B 模型，用 sigma_target=0.064 覆盖 PnL CSV。

关键优化：**直接加载已有 regime CSV**（不重新计算），大幅加速。

用法
----
    cd /path/to/IEOR4733_Project
    source .venv/bin/activate
    python regime_detection/reeval_route_b_fast.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "rl_models"))

from config import ASSET_CLASSES
from rl_models.a2c_model import PaperA2CTrainer, DEVICE
from rl_models.loaddata import (
    load_paper_rad_data,
    build_paper_features,
    make_state_tensor_single,
)
from regime_detection.route_b_train import PERIODS, FEATURE_COLS_12

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NEW_SIGMA_TARGET = 0.064

CKPT_DIR  = PROJECT_ROOT / "rl_models"
REGIME_DIR = PROJECT_ROOT / "regime_detection" / "results"
PNL_DIR   = REGIME_DIR
PNL_DIR.mkdir(exist_ok=True)

all_tickers = [t for v in ASSET_CLASSES.values() for t in v]
ASSET_CLASSES_WITH_ALL = dict(ASSET_CLASSES)
ASSET_CLASSES_WITH_ALL["All"] = all_tickers

TARGET_PERIODS = ["period_1", "period_2"]
TARGET_CLASSES = ["Commodity", "Equity Index", "Fixed Income", "Forex", "All"]


def _build_augmented_state_from_csv(
    tickers: list,
    period_name: str,
    asset_class: str,
    data_start: str,
    test_end: str,
):
    """快速构建 12-dim 状态：加载 RAD 特征 + 直接读取已有 regime CSV"""
    root_path = str(PROJECT_ROOT)
    data_dict, _, missing = load_paper_rad_data(root_path, start=data_start, end=test_end)
    data_dict = {t: v for t, v in data_dict.items() if t in tickers}
    feature_dict, _ = build_paper_features(data_dict, dropna=False)

    # 加载已有 regime CSV
    ac_slug = asset_class.replace(" ", "_")
    regime_files = list(REGIME_DIR.glob(f"regime_routeB_*_{ac_slug}.csv"))

    # 合并所有 period 的 regime 数据
    regimes = []
    for rf in regime_files:
        if not rf.exists():
            continue
        df = pd.read_csv(rf, parse_dates=["date"])
        regimes.append(df[["date", "regime_prob_0", "regime_prob_1", "regime_prob_2"]])

    if not regimes:
        logger.warning(f"  No regime CSV found for {asset_class}")
        return None

    regime_df = pd.concat(regimes, ignore_index=True).drop_duplicates(subset=["date"])

    # 为每个 ticker 构建 12-dim 状态
    state_dict = {}
    for ticker, feature_df in feature_dict.items():
        if feature_df is None or len(feature_df) == 0:
            continue

        # 合并 regime probs
        fd = feature_df.copy()
        rd = regime_df[["date", "regime_prob_0", "regime_prob_1", "regime_prob_2"]].copy()

        fd["_date_key"] = pd.to_datetime(fd["date"]).dt.normalize()
        rd["_date_key"] = pd.to_datetime(rd["date"]).dt.normalize()

        merged = fd.merge(rd, on="_date_key", how="left", suffixes=("", "_r"))
        merged = merged.drop(columns=["_date_key", "date_r"], errors="ignore")

        # 填充缺失的 regime probs
        for col in ["regime_prob_0", "regime_prob_1", "regime_prob_2"]:
            merged[col] = merged[col].ffill().bfill().fillna(1.0 / 3)

        # 构建 12-dim 特征张量
        feature_cols = [
            "close_norm",
            "ret_1m", "ret_2m", "ret_3m", "ret_1y",
            "macd_8_24", "macd_16_48", "macd_32_96",
            "rsi_30",
            "regime_prob_0", "regime_prob_1", "regime_prob_2",
        ]
        feature_data = merged[feature_cols].fillna(0).values  # (T, 12)

        # 构建滑动窗口 (T, 60, 12)
        X = []
        for i in range(len(feature_data) - 60 + 1):
            X.append(feature_data[i : i + 60])
        X = np.array(X)

        if len(X) == 0:
            continue

        # 提取对齐数据（日期、收益、波动率）
        al_cols = ["date", "close", "ret_1d", "ewm_vol_60"]
        al_df = merged[al_cols].iloc[59:].reset_index(drop=True)

        state_dict[ticker] = {
            "X": X,  # (T-60+1, 60, 12)
            "aligned_df": al_df,
        }

    return state_dict


def _inference_deterministic(
    actor, state_dict: dict, sigma_target: float
) -> dict:
    """快速推理：actor(state) → scaled_pos → PnL"""
    from rl_models.a2c_model import BP, MU

    actor.eval()
    daily_pnl = []

    for ticker, sd in state_dict.items():
        X = sd["X"]
        al_df = sd["aligned_df"].reset_index(drop=True)

        if len(X) == 0:
            continue

        pnl_rows = []
        prev_action = 0.0

        for i in range(len(X) - 1):
            row_t = al_df.iloc[i]
            row_tp1 = al_df.iloc[i + 1]

            sigma_t = row_t["ewm_vol_60"]
            if pd.isna(sigma_t) or sigma_t == 0:
                pnl_rows.append((row_tp1["date"], np.nan))
                continue

            state = torch.tensor(X[i : i + 1], dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                action = actor.deterministic_action(state).item()

            action = float(np.clip(action, -1.0, 1.0))
            scaled_pos = sigma_target / sigma_t * action

            prev_scaled = 0.0
            if i > 0:
                prev_sigma = al_df.iloc[i - 1]["ewm_vol_60"]
                if not pd.isna(prev_sigma) and prev_sigma > 0:
                    prev_scaled = sigma_target / prev_sigma * prev_action

            r_tp1 = row_tp1["ret_1d"]
            price_t = row_t["close"]

            if pd.isna(r_tp1) or pd.isna(price_t):
                pnl_rows.append((row_tp1["date"], np.nan))
            else:
                gross = MU * scaled_pos * r_tp1
                cost = MU * BP * price_t * abs(scaled_pos - prev_scaled)
                pnl_rows.append((row_tp1["date"], gross - cost))

            prev_action = action

        if pnl_rows:
            dates, pnls = zip(*pnl_rows)
            daily_pnl.append(
                pd.Series(pnls, index=pd.DatetimeIndex(dates), name=ticker)
            )

    if not daily_pnl:
        return {}

    pnl_df = pd.DataFrame(daily_pnl).dropna(how="all")
    portfolio = pnl_df.mean(axis=1).dropna()

    if len(portfolio) == 0:
        return {}

    ann = np.sqrt(252)
    mean_ret = portfolio.mean() * 252
    std_ret = portfolio.std() * ann
    sharpe = mean_ret / std_ret if std_ret > 0 else 0.0

    cum = portfolio.cumsum()
    rolling_max = cum.cummax()
    drawdown = cum - rolling_max
    mdd = drawdown.min()
    calmar = mean_ret / abs(mdd) if mdd < 0 else 0.0

    return {
        "portfolio_ann_ret": mean_ret,
        "portfolio_ann_std": std_ret,
        "portfolio_sharpe": sharpe,
        "portfolio_calmar": calmar,
        "portfolio_mdd": mdd,
        "portfolio_pnl": portfolio,
    }


def main():
    print(f"\n{'='*70}")
    print(f"Route B Fast Re-evaluation  |  sigma_target = {NEW_SIGMA_TARGET}")
    print(f"{'='*70}\n")

    summary = []

    for period_name in TARGET_PERIODS:
        cfg = PERIODS[period_name]
        data_start = cfg["data_start"]
        test_end = cfg["test"][-1]

        for asset_class in TARGET_CLASSES:
            tickers = ASSET_CLASSES_WITH_ALL[asset_class]
            ac_slug = asset_class.replace(" ", "_")

            ckpt_path = CKPT_DIR / f"a2c_rb_{ac_slug}_{period_name}.pt"
            if not ckpt_path.exists():
                logger.warning(f"⚠️  Checkpoint not found: {ckpt_path.name}")
                continue

            logger.info(f"[{period_name} | {asset_class}]")

            # 加载 checkpoint
            trainer = PaperA2CTrainer(n_features=12, device=DEVICE)
            trainer.load_checkpoint(str(ckpt_path))

            # 构建状态（直接读 regime CSV，不重新计算）
            state_dict = _build_augmented_state_from_csv(
                tickers, period_name, asset_class, data_start, test_end
            )
            if state_dict is None or len(state_dict) == 0:
                logger.warning(
                    f"  ⚠️  Failed to build state for {asset_class} {period_name}"
                )
                continue

            logger.info(f"  State dict: {len(state_dict)} tickers")

            # 推理
            metrics = _inference_deterministic(
                trainer.actor, state_dict, sigma_target=NEW_SIGMA_TARGET
            )

            if not metrics:
                logger.warning(
                    f"  ⚠️  Inference failed for {asset_class} {period_name}"
                )
                continue

            sharpe = metrics.get("portfolio_sharpe", float("nan"))
            calmar = metrics.get("portfolio_calmar", float("nan"))
            ann_ret = metrics.get("portfolio_ann_ret", float("nan"))

            logger.info(
                f"  ✅ Sharpe={sharpe:.3f}  Calmar={calmar:.3f}  AnnRet={ann_ret:.4f}"
            )

            # 覆盖 CSV
            portfolio_pnl = metrics.get("portfolio_pnl")
            if portfolio_pnl is not None and len(portfolio_pnl) > 0:
                csv_path = PNL_DIR / f"pnl_routeB_{period_name}_{ac_slug}.csv"
                pnl_df = pd.DataFrame(
                    {
                        "date": portfolio_pnl.index,
                        "net_pnl": portfolio_pnl.values,
                        "cum_wealth": 1.0 + portfolio_pnl.cumsum().values,
                    }
                )
                pnl_df.to_csv(csv_path, index=False)
                logger.info(f"  💾 Saved: {csv_path.name}  ({len(pnl_df)} rows)")
                summary.append((period_name, asset_class, sharpe, calmar, ann_ret))

    # 汇总
    print(f"\n{'='*70}")
    print(f"Summary  (sigma_target = {NEW_SIGMA_TARGET})")
    print(f"{'='*70}")
    print(
        f"{'Period':<12} {'Asset Class':<18} {'Sharpe':>8} {'Calmar':>8} {'AnnRet':>9}"
    )
    print("-" * 58)
    for period, ac, sh, ca, ar in summary:
        print(f"{period:<12} {ac:<18} {sh:>8.3f} {ca:>8.3f} {ar:>9.4f}")

    print(f"\n✅ Done. CSV files overwritten in: {PNL_DIR}")
    print(f"   Now go to strategies_comparison.ipynb Section 8 and re-run.")


if __name__ == "__main__":
    main()
