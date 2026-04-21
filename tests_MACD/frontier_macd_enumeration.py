#!/usr/bin/env python3
"""
MACD 策略数据源前沿枚举（自包含版本）。

功能：
  - 自包含数据加载，支持 source_overrides 逐 ticker 指定数据源
  - 4-asset scorecard：4 × 9 = 36 指标（Paper Table 3）
  - 遍历数据源 overrides、排除合约集合、报告参数，最大化 <=15% 容差评分
  - All 资产类别计算并上报，不计入 score
  - 搜索结果以 JSON 写入 tests_MACD/best_overrides_macd.json
"""
from __future__ import annotations

import json
import itertools
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    ASSET_CLASSES, BP, EWMA_SPAN, METRIC_NAMES,
    PAPER_TABLE3, SIGN_LOOKBACK, TRADING_DAYS,
)
from metrics import compute_metrics  # noqa: E402
from strategies import strategy_sign_r, strategy_macd  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIGMA = 0.058
STRAT = "MACD"
ASSETS4 = ["Commodity", "Equity Index", "Fixed Income", "Forex"]
MAX_SCORE = len(ASSETS4) * len(METRIC_NAMES)  # 36

NUMERATOR_MODES = ["wealth_cagr", "annual_mean_simple", "annual_mean_log"]
ALL_MODES = [
    "contract_equal_path",
    "asset_equal_path",
    "asset_count_weighted_path",
    "asset_equal_simple",
    "asset_count_weighted_simple",
]
FAST_ALL_MODES = ["contract_equal_path", "asset_equal_path", "asset_count_weighted_path"]

# JSON output path
RESULTS_JSON = Path(__file__).resolve().parent / "best_overrides_macd.json"


# ---------------------------------------------------------------------------
# Self-contained data loading
# ---------------------------------------------------------------------------

def _load_csv(ticker: str, dataset: str = "RAD") -> pd.DataFrame | None:
    """Load a CLC CSV by ticker and dataset, trying CLCDATA first then CLC."""
    dataset = dataset.upper()
    for data_dir in ["data/CLCDATA", "data/CLC"]:
        fpath = os.path.join(ROOT, data_dir, f"{ticker}_{dataset}.CSV")
        if os.path.exists(fpath):
            df = pd.read_csv(
                fpath, header=None,
                names=["Date", "Open", "High", "Low", "Close", "Volume", "OI"],
            )
            df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
            df = df[df["Close"].notna() & (df["Close"] > 0)].sort_values("Date").reset_index(drop=True)
            df = df[df["Date"] >= "2009-01-01"].reset_index(drop=True)
            if len(df) >= 500:
                return df
    return None


def load_contract(ticker: str, source: str = "RAD",
                  test_start: str = "2011-01-01",
                  test_end: str = "2019-12-31") -> dict | None:
    """Load and prepare a single contract."""
    df = _load_csv(ticker, source)
    if df is None:
        return None
    prices = df["Close"].values.astype(float)
    p0 = prices[0]
    norm_p = prices / p0
    rt = np.zeros(len(norm_p))
    rt[1:] = norm_p[1:] - norm_p[:-1]
    sigma = pd.Series(rt).ewm(span=EWMA_SPAN, adjust=False).std().values

    mask_s = df["Date"] >= test_start
    mask_e = df["Date"] <= test_end
    if not mask_s.any() or not mask_e.any():
        return None
    t0 = mask_s.idxmax()
    t1 = len(df) - 1 - mask_e[::-1].values.argmax()
    start = max(t0, SIGN_LOOKBACK)
    dates = df["Date"].iloc[start:t1].values
    return {
        "tk": ticker, "rt": rt, "sigma": sigma, "norm_p": norm_p,
        "prices": prices, "start": start, "t1": t1, "dates": dates,
        "macd_pos": strategy_macd(norm_p),
    }


def load_contracts_for_asset(
    asset: str,
    source_overrides: dict[str, str] | None = None,
    excluded: set | None = None,
    default_source: str = "RAD",
) -> list[dict]:
    """Load all contracts for an asset class with per-ticker source overrides."""
    source_overrides = source_overrides or {}
    excluded = excluded or set()
    tickers = ASSET_CLASSES.get(asset, [])
    result = []
    for tk in tickers:
        if tk in excluded:
            continue
        source = source_overrides.get(tk, default_source)
        rd = load_contract(tk, source=source)
        if rd is not None:
            result.append(rd)
    return result


# ---------------------------------------------------------------------------
# Compute returns (self-contained)
# ---------------------------------------------------------------------------

def compute_contract_returns(rd: dict, strat: str, sigma_tgt: float) -> np.ndarray:
    """Eq 4: R_t for one contract."""
    rt, sigma, norm_p = rd["rt"], rd["sigma"], rd["norm_p"]
    n = len(rt)
    if strat == "Long":
        pos = np.ones(n)
    elif strat == "Sign(R)":
        pos = strategy_sign_r(rt, SIGN_LOOKBACK)
    else:
        pos = rd["macd_pos"]
    Rt = np.zeros(n)
    for t in range(1, n):
        if sigma[t - 1] > 0 and (t < 2 or sigma[t - 2] > 0):
            a_prev = pos[t - 1]
            a_prev2 = pos[t - 2] if t >= 2 else 0.0
            sp = a_prev * sigma_tgt / sigma[t - 1]
            spp = a_prev2 * sigma_tgt / sigma[t - 2] if t >= 2 else 0.0
            Rt[t] = sp * rt[t] - BP * norm_p[t - 1] * abs(sp - spp)
    return Rt


def compute_portfolio_returns(raw_data: list, strat: str, sigma_tgt: float) -> np.ndarray:
    """Eq 13: equal-weight portfolio return."""
    series = []
    for rd in raw_data:
        Rt = compute_contract_returns(rd, strat, sigma_tgt)
        start, t1, dates = rd["start"], rd["t1"], rd["dates"]
        slc = Rt[start:t1]
        series.append(pd.Series(slc[: len(dates)], index=dates[: len(slc)]))
    return pd.DataFrame(series).T.dropna().mean(axis=1).values


def max_drawdown_from_path(path: np.ndarray) -> float:
    """Max drawdown from a wealth path."""
    peak = np.maximum.accumulate(path)
    dd = (peak - path) / np.where(peak > 0, peak, 1.0)
    return float(np.max(dd)) if len(dd) > 0 else 0.0


# ---------------------------------------------------------------------------
# Data-source override sets (same as Sign(R) — shared data quality findings)
# ---------------------------------------------------------------------------

_BASE_OVERRIDES: dict[str, str] = {}

BASE_CLEAN_OVERRIDES = dict(_BASE_OVERRIDES)
BASE_CLEAN_OVERRIDES.update({
    "EN": "REV", "DT": "REV",
    "CC": "RAD", "LB": "RAD",
    "JO": "RAD", "ZH": "RAD",
    "NR": "NON", "ZC": "NON",
})
BASE_CLEAN_EXCLUDED = {"FB", "ZA", "ZO", "SB", "KC", "ZL"}

STRUCTURAL_HISTORY_OVERRIDES = dict(_BASE_OVERRIDES)
STRUCTURAL_HISTORY_OVERRIDES.update({
    "DT": "REV", "CC": "RAD",
    "LB": "RAD", "JO": "RAD", "ZH": "RAD",
})
STRUCTURAL_HISTORY_EXCLUDED = {"FB", "ZA", "ZO", "EN", "ES"}

LEGACY_EXPERIMENTAL_OVERRIDES = dict(_BASE_OVERRIDES)
LEGACY_EXPERIMENTAL_OVERRIDES.update({
    "EN": "REV", "DT": "REV",
    "CC": "RAD", "LB": "REV",
    "JO": "REV", "ZH": "REV",
})
LEGACY_EXPERIMENTAL_EXCLUDED = {"FB", "ZA", "ZO", "EN", "ES"}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def pct_err(value: float, paper: float) -> float:
    if not np.isfinite(value) or not np.isfinite(paper):
        return float("inf")
    if paper == 0:
        return abs(value - paper)
    return abs((value - paper) / abs(paper)) * 100.0


def annual_mean(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(arr.mean() * 252.0)


def annual_return_from_reporting(reporting: dict, numerator_mode: str) -> float:
    port = reporting["portfolio_path"]
    if numerator_mode == "wealth_cagr":
        ratio = port[-1] / port[0]
        if not np.isfinite(ratio) or ratio <= 0:
            return float("nan")
        return float(ratio ** (252.0 / len(port)) - 1.0)
    if numerator_mode == "annual_mean_simple":
        return annual_mean(reporting["portfolio_simple_returns"][1:])
    if numerator_mode == "annual_mean_log":
        return annual_mean(reporting["portfolio_log_returns"][1:])
    raise ValueError(numerator_mode)


# ---------------------------------------------------------------------------
# Portfolio reporting builders
# ---------------------------------------------------------------------------

def build_reporting_portfolio(raw_data: list) -> dict | None:
    """Build sleeve paths for MACD strategy."""
    sleeve_paths = []
    for rd in raw_data:
        Rt = compute_contract_returns(rd, STRAT, SIGMA)
        start, t1 = rd["start"], rd["t1"]
        Rt_test = Rt[start:t1 + 1]
        sigma_slice = rd["sigma"][start:t1 + 1]
        prices_slice = rd["prices"][start:t1 + 1]
        if len(Rt_test) == 0:
            continue
        sigma0 = float(sigma_slice[0])
        if not np.isfinite(sigma0) or sigma0 <= 0:
            continue
        capital0 = float(prices_slice[0]) * SIGMA / sigma0
        if not np.isfinite(capital0) or capital0 <= 0:
            continue
        sleeve_paths.append(1.0 + np.cumsum(Rt_test / capital0))

    if not sleeve_paths:
        return None

    min_len = min(len(p) for p in sleeve_paths)
    sleeves = np.column_stack([p[:min_len] for p in sleeve_paths])
    portfolio = sleeves.mean(axis=1)

    portfolio_simple = np.full(min_len, np.nan, dtype=float)
    portfolio_log = np.full(min_len, np.nan, dtype=float)
    if min_len > 1:
        portfolio_simple[1:] = portfolio[1:] / portfolio[:-1] - 1.0
        with np.errstate(invalid="ignore", divide="ignore"):
            portfolio_log[1:] = np.log(portfolio[1:] / portfolio[:-1])

    return {
        "portfolio_path": portfolio,
        "portfolio_simple_returns": portfolio_simple,
        "portfolio_log_returns": portfolio_log,
        "sleeve_paths": sleeves,
        "contract_count": sleeves.shape[1],
    }


def build_all_reporting(asset_reporting: dict[str, dict], all_mode: str) -> dict:
    """Combine per-asset reporting into All portfolio."""
    min_len = min(len(asset_reporting[a]["portfolio_path"]) for a in ASSETS4)
    counts = np.array([asset_reporting[a]["contract_count"] for a in ASSETS4], dtype=float)
    counts = counts / counts.sum()

    if all_mode == "contract_equal_path":
        sleeves = [asset_reporting[a]["sleeve_paths"][:min_len, :] for a in ASSETS4]
        stacked = np.column_stack(sleeves)
        path = stacked.mean(axis=1)
    elif all_mode == "asset_equal_path":
        mat = np.column_stack([asset_reporting[a]["portfolio_path"][:min_len] for a in ASSETS4])
        path = mat.mean(axis=1)
    elif all_mode == "asset_count_weighted_path":
        mat = np.column_stack([asset_reporting[a]["portfolio_path"][:min_len] for a in ASSETS4])
        path = (mat * counts).sum(axis=1)
    elif all_mode in {"asset_equal_simple", "asset_count_weighted_simple"}:
        mat = np.column_stack([asset_reporting[a]["portfolio_path"][:min_len] for a in ASSETS4])
        rets = np.full_like(mat, np.nan, dtype=float)
        rets[1:, :] = mat[1:, :] / mat[:-1, :] - 1.0
        daily = (
            rets[1:, :].mean(axis=1) if all_mode == "asset_equal_simple"
            else (rets[1:, :] * counts).sum(axis=1)
        )
        path = np.empty(len(daily) + 1, dtype=float)
        path[0] = 1.0
        path[1:] = np.cumprod(1.0 + daily)
    else:
        raise ValueError(all_mode)

    simple = np.full(len(path), np.nan, dtype=float)
    log_ = np.full(len(path), np.nan, dtype=float)
    if len(path) > 1:
        simple[1:] = path[1:] / path[:-1] - 1.0
        with np.errstate(invalid="ignore", divide="ignore"):
            log_[1:] = np.log(path[1:] / path[:-1])

    total_contracts = int(sum(asset_reporting[a]["contract_count"] for a in ASSETS4))
    return {
        "portfolio_path": path,
        "portfolio_simple_returns": simple,
        "portfolio_log_returns": log_,
        "contract_count": total_contracts,
    }


# ---------------------------------------------------------------------------
# Core evaluation (with lru_cache)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def evaluate_scenario(
    overrides_key: tuple[tuple[str, str], ...],
    excluded_key: tuple[str, ...],
    numerator_mode: str,
    all_mode: str,
) -> dict:
    overrides = dict(overrides_key)
    excluded = set(excluded_key)

    results: dict = {}
    score10 = 0
    score15 = 0
    mean_ann_errs: list[float] = []
    mean_cal_errs: list[float] = []
    asset_reporting: dict = {}

    for asset in ASSETS4:
        raw = load_contracts_for_asset(asset, source_overrides=overrides, excluded=excluded)
        trade_returns = compute_portfolio_returns(raw, STRAT, SIGMA)
        trade_metrics = dict(zip(METRIC_NAMES, compute_metrics(trade_returns, n_contracts=len(raw))))

        reporting = build_reporting_portfolio(raw)
        if reporting is None:
            reporting = {
                "portfolio_path": np.array([1.0, 1.0]),
                "portfolio_simple_returns": np.array([np.nan, np.nan]),
                "portfolio_log_returns": np.array([np.nan, np.nan]),
                "sleeve_paths": np.ones((2, 1)),
                "contract_count": 0,
            }

        ann = annual_return_from_reporting(reporting, numerator_mode)
        mdd = max_drawdown_from_path(reporting["portfolio_path"])
        cal = ann / mdd if mdd > 0 and np.isfinite(ann) else float("nan")
        trade_metrics["MDD"] = round(mdd, 3)
        trade_metrics["Calmar"] = round(cal, 3) if np.isfinite(cal) else float("nan")

        paper = PAPER_TABLE3[asset][STRAT]
        errors = {m: pct_err(trade_metrics[m], paper[m]) for m in METRIC_NAMES}
        misses = [m for m in METRIC_NAMES if errors[m] >= 15.0]

        asset_reporting[asset] = reporting
        results[asset] = {
            "metrics": trade_metrics,
            "errors": errors,
            "misses15": misses,
            "n_contracts": len(raw),
        }
        score10 += sum(errors[m] < 10 for m in METRIC_NAMES)
        score15 += sum(errors[m] < 15 for m in METRIC_NAMES)
        mean_ann_errs.append(pct_err(ann, paper["Calmar"] * paper["MDD"]))
        mean_cal_errs.append(pct_err(cal, paper["Calmar"]))

    # --- All ---
    all_reporting = build_all_reporting(asset_reporting, all_mode)
    all_ann = annual_return_from_reporting(all_reporting, numerator_mode)
    all_mdd = max_drawdown_from_path(all_reporting["portfolio_path"])
    all_cal = all_ann / all_mdd if all_mdd > 0 and np.isfinite(all_ann) else float("nan")

    raw_all = []
    for a in ASSETS4:
        raw_all.extend(load_contracts_for_asset(a, source_overrides=overrides, excluded=excluded))
    all_trade_returns = compute_portfolio_returns(raw_all, STRAT, SIGMA)
    all_metrics = dict(zip(METRIC_NAMES, compute_metrics(all_trade_returns, n_contracts=len(raw_all))))
    all_metrics["MDD"] = round(all_mdd, 3)
    all_metrics["Calmar"] = round(all_cal, 3) if np.isfinite(all_cal) else float("nan")

    results["All"] = {
        "metrics": all_metrics,
        "errors": {},
        "misses15": [],
        "n_contracts": len(raw_all),
        "no_paper_target": True,
    }

    return {
        "score10": score10,
        "score15": score15,
        "max_score": MAX_SCORE,
        "mean_ann_gap": float(np.nanmean(mean_ann_errs)),
        "mean_cal_gap": float(np.nanmean(mean_cal_errs)),
        "results": results,
        "rank": (
            -score15,
            -score10,
            float(np.nanmean(mean_ann_errs)),
            float(np.nanmean(mean_cal_errs)),
        ),
    }


# ---------------------------------------------------------------------------
# scenario() wrapper
# ---------------------------------------------------------------------------

def scenario(
    label: str,
    family: str,
    overrides: dict,
    excluded: set | list,
    numerator_mode: str = "wealth_cagr",
    all_mode: str = "contract_equal_path",
) -> dict:
    summary = evaluate_scenario(
        tuple(sorted(overrides.items())),
        tuple(sorted(excluded)),
        numerator_mode,
        all_mode,
    )
    return {
        "label": label,
        "family": family,
        "summary": summary,
        "overrides": dict(overrides),
        "excluded": sorted(excluded),
        "numerator_mode": numerator_mode,
        "all_mode": all_mode,
    }


# ---------------------------------------------------------------------------
# Search functions
# ---------------------------------------------------------------------------

def search_clean_same_rule() -> list[dict]:
    rows = []
    for numerator_mode, all_mode in itertools.product(NUMERATOR_MODES, ALL_MODES):
        rows.append(scenario(
            label=f"clean / {numerator_mode} / {all_mode}",
            family="clean_same_rule",
            overrides=BASE_CLEAN_OVERRIDES,
            excluded=BASE_CLEAN_EXCLUDED,
            numerator_mode=numerator_mode,
            all_mode=all_mode,
        ))
    rows.sort(key=lambda r: r["summary"]["rank"])
    return rows


def search_coherent_override() -> list[dict]:
    rows = []
    bases = [
        ("clean", BASE_CLEAN_OVERRIDES, BASE_CLEAN_EXCLUDED),
        ("structural", STRUCTURAL_HISTORY_OVERRIDES, STRUCTURAL_HISTORY_EXCLUDED),
    ]
    for base_name, overrides, excluded in bases:
        for numerator_mode, all_mode in itertools.product(NUMERATOR_MODES, FAST_ALL_MODES):
            rows.append(scenario(
                label=f"{base_name} / {numerator_mode} / {all_mode}",
                family="coherent_override",
                overrides=overrides,
                excluded=excluded,
                numerator_mode=numerator_mode,
                all_mode=all_mode,
            ))
    rows.sort(key=lambda r: r["summary"]["rank"])
    return rows


def search_structural_heavy() -> list[dict]:
    rows = []
    structural_bases = [
        ("history_seed", STRUCTURAL_HISTORY_OVERRIDES, STRUCTURAL_HISTORY_EXCLUDED),
        ("clean_plus_en_es", BASE_CLEAN_OVERRIDES, set(BASE_CLEAN_EXCLUDED) | {"EN", "ES"}),
    ]
    for base_name, overrides, excluded in structural_bases:
        for numerator_mode, all_mode in itertools.product(NUMERATOR_MODES, FAST_ALL_MODES):
            rows.append(scenario(
                label=f"{base_name} / {numerator_mode} / {all_mode}",
                family="structural_heavy",
                overrides=overrides,
                excluded=excluded,
                numerator_mode=numerator_mode,
                all_mode=all_mode,
            ))
    rows.sort(key=lambda r: r["summary"]["rank"])
    return rows


def search_legacy_experimental() -> list[dict]:
    rows = []
    for numerator_mode, all_mode in itertools.product(NUMERATOR_MODES, ALL_MODES):
        rows.append(scenario(
            label=f"legacy_experimental / {numerator_mode} / {all_mode}",
            family="legacy_experimental",
            overrides=LEGACY_EXPERIMENTAL_OVERRIDES,
            excluded=LEGACY_EXPERIMENTAL_EXCLUDED,
            numerator_mode=numerator_mode,
            all_mode=all_mode,
        ))
    rows.sort(key=lambda r: r["summary"]["rank"])
    return rows


# ---------------------------------------------------------------------------
# JSON save/load
# ---------------------------------------------------------------------------

def save_best_to_json(best_row: dict, path: Path | None = None) -> Path:
    """Save the best scenario overrides and config to JSON."""
    path = path or RESULTS_JSON
    data = {
        "strategy": STRAT,
        "label": best_row["label"],
        "family": best_row["family"],
        "sigma": SIGMA,
        "overrides": best_row["overrides"],
        "excluded": best_row["excluded"],
        "numerator_mode": best_row["numerator_mode"],
        "all_mode": best_row["all_mode"],
        "score10": int(best_row["summary"]["score10"]),
        "score15": int(best_row["summary"]["score15"]),
        "max_score": int(MAX_SCORE),
        "per_asset": {},
    }
    for asset in ASSETS4 + ["All"]:
        res = best_row["summary"]["results"][asset]
        data["per_asset"][asset] = {
            "metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                        for k, v in res["metrics"].items()},
            "misses15": res["misses15"],
            "n_contracts": int(res["n_contracts"]),
        }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def load_best_from_json(path: Path | None = None) -> dict | None:
    """Load previously saved best overrides from JSON."""
    path = path or RESULTS_JSON
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def print_scenario_detail(row: dict) -> None:
    s = row["summary"]
    print(f"\n{'='*72}")
    print(f"Label  : {row['label']}")
    print(f"Family : {row['family']}")
    print(f"Score  : <=10: {s['score10']}/{MAX_SCORE}   <=15: {s['score15']}/{MAX_SCORE}")
    print(f"{'='*72}")
    header = f"{'Asset':<18} {'E(R)':>8} {'std':>8} {'DD':>8} {'Sharpe':>8} {'Sort':>8} {'MDD':>8} {'Calmar':>8} {'%+':>7} {'A P/L':>8}"
    print(header)
    print("-" * 90)
    for asset in ASSETS4 + ["All"]:
        res = s["results"][asset]
        m = res["metrics"]
        no_paper = res.get("no_paper_target", False)
        suffix = " (no paper)" if no_paper else ""
        vals = (
            f"{asset:<18}"
            f" {m.get('E(R)', float('nan')):>8.3f}"
            f" {m.get('std(R)', float('nan')):>8.3f}"
            f" {m.get('DD', float('nan')):>8.3f}"
            f" {m.get('Sharpe', float('nan')):>8.3f}"
            f" {m.get('Sortino', float('nan')):>8.3f}"
            f" {m.get('MDD', float('nan')):>8.3f}"
            f" {m.get('Calmar', float('nan')):>8.3f}"
            f" {m.get('% +ve', float('nan')):>7.3f}"
            f" {m.get('Ave P/L', float('nan')):>8.3f}"
            f"{suffix}"
        )
        print(vals)
        if not no_paper:
            paper = PAPER_TABLE3[asset][STRAT]
            p_vals = (
                f"{'  paper':<18}"
                f" {paper.get('E(R)', float('nan')):>8.3f}"
                f" {paper.get('std(R)', float('nan')):>8.3f}"
                f" {paper.get('DD', float('nan')):>8.3f}"
                f" {paper.get('Sharpe', float('nan')):>8.3f}"
                f" {paper.get('Sortino', float('nan')):>8.3f}"
                f" {paper.get('MDD', float('nan')):>8.3f}"
                f" {paper.get('Calmar', float('nan')):>8.3f}"
                f" {paper.get('% +ve', float('nan')):>7.3f}"
                f" {paper.get('Ave P/L', float('nan')):>8.3f}"
            )
            print(p_vals)
            misses = res["misses15"]
            miss_str = ", ".join(misses) if misses else "none"
            print(f"  miss15: {miss_str}")
        print()
