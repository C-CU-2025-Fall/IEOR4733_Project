#!/usr/bin/env python3
"""
Sign(R) 策略数据源前沿枚举。

将 tests_long/frontier_40plus_enumeration.py 中的 Long 枚举框架适配为 Sign(R) 版本：
  - 4-asset scorecard：4 × 9 = 36 指标（Paper Table 3 无 Sign(R) All 行）
  - 遍历数据源 overrides、排除合约集合、报告参数，最大化 <=15% 容差评分
  - All 资产类别依然计算并上报，但不计入 score（无 paper 目标）
"""
from __future__ import annotations

from functools import lru_cache
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_run import compute_contract_returns, compute_portfolio_returns, load_contracts  # noqa: E402
from config import METRIC_NAMES, PAPER_TABLE3, SOURCE_OVERRIDES  # noqa: E402
from data_loader import load_clc_full  # noqa: E402
from metrics import compute_metrics, max_drawdown_from_path  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIGMA = 0.058
STRAT = "Sign(R)"
ASSETS4 = ["Commodity", "Equity Index", "Fixed Income", "Forex"]
MAX_SCORE = len(ASSETS4) * len(METRIC_NAMES)  # 36

NUMERATOR_MODES = ["wealth_cagr", "annual_mean_simple", "annual_mean_log", "annual_mean_sleeve"]
ASSET_PATH_MODES = ["contract_equal_path", "sleeve_first_simple_path"]
ALL_MODES = [
    "contract_equal_path",
    "asset_equal_path",
    "asset_count_weighted_path",
    "asset_equal_simple",
    "asset_count_weighted_simple",
]
CLEAN_NUMERATORS = ["wealth_cagr", "annual_mean_simple", "annual_mean_log"]
FAST_ALL_MODES = ["contract_equal_path", "asset_equal_path", "asset_count_weighted_path"]
OVERRIDE_ASSETS = ["Commodity", "Equity Index", "Fixed Income"]

# ---------------------------------------------------------------------------
# Data-source override sets（与 Long 枚举保持相同的出发点；Sign(R) 可能需要不同最优解）
# ---------------------------------------------------------------------------

BASE_CLEAN_OVERRIDES = dict(SOURCE_OVERRIDES)
BASE_CLEAN_OVERRIDES.update(
    {
        "EN": "REV",
        "DT": "REV",
        "CC": "RAD_REGEN",
        "LB": "RAD",
        "JO": "RAD_REGEN",
        "ZH": "RAD_REGEN",
        "NR": "NON",
        "ZC": "NON",
    }
)
BASE_CLEAN_OVERRIDES.pop("ZO", None)
BASE_CLEAN_EXCLUDED = {"FB", "ZA", "ZO", "SB", "KC", "ZL"}

STRUCTURAL_HISTORY_OVERRIDES = dict(SOURCE_OVERRIDES)
STRUCTURAL_HISTORY_OVERRIDES.update(
    {
        "DT": "REV",
        "CC": "RAD_REGEN",
        "LB": "RAD",
        "JO": "RAD_REGEN",
        "ZH": "RAD_REGEN",
    }
)
STRUCTURAL_HISTORY_EXCLUDED = {"FB", "ZA", "ZO", "EN", "ES"}

HYBRID_STRUCTURAL_OVERRIDES = dict(BASE_CLEAN_OVERRIDES)
HYBRID_STRUCTURAL_EXCLUDED = set(BASE_CLEAN_EXCLUDED) | {"EN", "ES"}

# Long 策略最优配置（作为 Sign(R) 枚举的对照起点）
LEGACY_EXPERIMENTAL_OVERRIDES = dict(SOURCE_OVERRIDES)
LEGACY_EXPERIMENTAL_OVERRIDES.update(
    {
        "EN": "REV",
        "DT": "REV",
        "CC": "RAD_REGEN",
        "LB": "REV",
        "JO": "REV",
        "ZH": "REV",
    }
)
LEGACY_EXPERIMENTAL_OVERRIDES.pop("ZO", None)
LEGACY_EXPERIMENTAL_EXCLUDED = {"FB", "ZA", "ZO", "EN", "ES"}

# ---------------------------------------------------------------------------
# Helper utilities（与 Long 版本相同，除了 STRAT 参数）
# ---------------------------------------------------------------------------

def pct_err(value: float, paper: float) -> float:
    if not np.isfinite(value) or not np.isfinite(paper):
        return float("inf")
    if paper == 0:
        return abs(value - paper)
    return abs((value - paper) / abs(paper)) * 100.0


def paper_helper(asset: str) -> float:
    """返回 paper 中 Sign(R) E(R) 的隐含值（Calmar × MDD）。"""
    paper = PAPER_TABLE3[asset][STRAT]
    return paper["Calmar"] * paper["MDD"]


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
    if numerator_mode == "annual_mean_sleeve":
        return annual_mean(reporting["sleeve_simple_returns"][1:, :])
    raise ValueError(numerator_mode)


def aligned_price_p0(ticker: str, date0, source: str) -> float | None:
    df = load_clc_full(ticker, source=source)
    if df is None:
        return None
    row = df[df["Date"] == date0]
    if row.empty:
        row = df[df["Date"] >= date0].head(1)
    if row.empty:
        return None
    p0 = float(row["Close"].iloc[0])
    if not np.isfinite(p0) or p0 <= 0:
        return None
    return p0


# ---------------------------------------------------------------------------
# Portfolio reporting builders
# ---------------------------------------------------------------------------

def build_reporting_portfolio(raw_data: list, capital_mode: str) -> dict | None:
    """
    按 capital_mode 归一化，用 Sign(R) 策略计算每份合约的 sleeve 路径，
    返回等权合约组合的路径字典。
    """
    sleeve_paths = []
    for rd in raw_data:
        detail = compute_contract_returns(rd, STRAT, SIGMA, detail=True)
        start, t1 = rd["start"], rd["t1"]
        Rt = detail["Rt"][start:t1 + 1]
        prices = detail["prices"][start:t1 + 1]
        sigma = detail["sigma"][start:t1 + 1]
        if len(Rt) == 0:
            continue
        sigma0 = float(sigma[0])
        if not np.isfinite(sigma0) or sigma0 <= 0:
            continue

        if capital_mode == "risk_price_source":
            capital0 = float(prices[0]) * SIGMA / sigma0
        elif capital_mode == "risk_price_non":
            date0 = rd["dates"][0]
            p0 = aligned_price_p0(rd["tk"], date0, "NON")
            if p0 is None:
                continue
            capital0 = p0 * SIGMA / sigma0
        else:
            raise ValueError(capital_mode)

        if not np.isfinite(capital0) or capital0 <= 0:
            continue
        sleeve_paths.append(1.0 + np.cumsum(Rt / capital0))

    if not sleeve_paths:
        return None

    min_len = min(len(p) for p in sleeve_paths)
    sleeves = np.column_stack([p[:min_len] for p in sleeve_paths])
    portfolio = sleeves.mean(axis=1)

    sleeve_simple = np.full_like(sleeves, np.nan, dtype=float)
    if min_len > 1:
        sleeve_simple[1:, :] = sleeves[1:, :] / sleeves[:-1, :] - 1.0

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
        "sleeve_simple_returns": sleeve_simple,
        "contract_count": sleeves.shape[1],
    }


def asset_path_variant(reporting: dict, asset_path_mode: str) -> dict:
    if asset_path_mode == "contract_equal_path":
        return reporting
    if asset_path_mode == "sleeve_first_simple_path":
        sleeves = reporting["sleeve_paths"]
        simple = np.full_like(sleeves, np.nan, dtype=float)
        simple[1:, :] = sleeves[1:, :] / sleeves[:-1, :] - 1.0
        daily = np.nanmean(simple[1:, :], axis=1)
        path = np.empty(len(daily) + 1, dtype=float)
        path[0] = 1.0
        path[1:] = np.cumprod(1.0 + daily)
        port_simple = np.full(len(path), np.nan, dtype=float)
        port_log = np.full(len(path), np.nan, dtype=float)
        if len(path) > 1:
            port_simple[1:] = path[1:] / path[:-1] - 1.0
            with np.errstate(invalid="ignore", divide="ignore"):
                port_log[1:] = np.log(path[1:] / path[:-1])
        return {
            "portfolio_path": path,
            "portfolio_simple_returns": port_simple,
            "portfolio_log_returns": port_log,
            "sleeve_simple_returns": simple,
            "sleeve_paths": sleeves,
            "contract_count": reporting["contract_count"],
        }
    raise ValueError(asset_path_mode)


def build_all_reporting(asset_reporting: dict[str, dict], all_mode: str) -> dict:
    min_len = min(len(asset_reporting[a]["portfolio_path"]) for a in ASSETS4)
    counts = np.array([asset_reporting[a]["contract_count"] for a in ASSETS4], dtype=float)
    counts = counts / counts.sum()

    if all_mode == "contract_equal_path":
        sleeves = [asset_reporting[a]["sleeve_paths"][:min_len, :] for a in ASSETS4]
        stacked = np.column_stack(sleeves)
        path = stacked.mean(axis=1)
        sleeve_simple = np.full_like(stacked, np.nan, dtype=float)
        if min_len > 1:
            sleeve_simple[1:, :] = stacked[1:, :] / stacked[:-1, :] - 1.0
        port_simple = np.full(min_len, np.nan, dtype=float)
        port_log = np.full(min_len, np.nan, dtype=float)
        if min_len > 1:
            port_simple[1:] = path[1:] / path[:-1] - 1.0
            with np.errstate(invalid="ignore", divide="ignore"):
                port_log[1:] = np.log(path[1:] / path[:-1])
        return {
            "portfolio_path": path,
            "portfolio_simple_returns": port_simple,
            "portfolio_log_returns": port_log,
            "sleeve_simple_returns": sleeve_simple,
            "sleeve_paths": stacked,
            "contract_count": stacked.shape[1],
        }

    mat = np.column_stack([asset_reporting[a]["portfolio_path"][:min_len] for a in ASSETS4])
    if all_mode == "asset_equal_path":
        path = mat.mean(axis=1)
    elif all_mode == "asset_count_weighted_path":
        path = (mat * counts).sum(axis=1)
    elif all_mode in {"asset_equal_simple", "asset_count_weighted_simple"}:
        rets = np.full_like(mat, np.nan, dtype=float)
        rets[1:, :] = mat[1:, :] / mat[:-1, :] - 1.0
        daily = (
            rets[1:, :].mean(axis=1)
            if all_mode == "asset_equal_simple"
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

    return {
        "portfolio_path": path,
        "portfolio_simple_returns": simple,
        "portfolio_log_returns": log_,
        "sleeve_simple_returns": np.full((len(path), 1), np.nan, dtype=float),
        "sleeve_paths": np.full((len(path), 1), np.nan, dtype=float),
        "contract_count": int(sum(asset_reporting[a]["contract_count"] for a in ASSETS4)),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw(asset: str, overrides: dict, excluded: list) -> list:
    if asset == "All":
        raw = []
        for a in ASSETS4:
            raw.extend(load_contracts(a, excluded_contracts=excluded, source_overrides=overrides))
        return raw
    return load_contracts(asset, excluded_contracts=excluded, source_overrides=overrides)


# ---------------------------------------------------------------------------
# Core evaluation（带 lru_cache，参数均为可哈希类型）
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def evaluate_scenario(
    overrides_key: tuple[tuple[str, str], ...],
    excluded_key: tuple[str, ...],
    default_capital_mode: str,
    asset_capital_overrides_key: tuple[tuple[str, str], ...],
    numerator_mode: str,
    asset_path_mode: str,
    all_mode: str,
) -> dict:
    overrides = dict(overrides_key)
    excluded = list(excluded_key)
    asset_capital_overrides = dict(asset_capital_overrides_key)

    results: dict = {}
    score10 = 0
    score15 = 0
    mean_ann_errs: list[float] = []
    mean_cal_errs: list[float] = []
    asset_reporting: dict = {}

    for asset in ASSETS4:
        raw = load_raw(asset, overrides, excluded)
        trade_returns = compute_portfolio_returns(raw, STRAT, SIGMA, aggregation_mode="variable_n")
        trade_metrics = dict(zip(METRIC_NAMES, compute_metrics(trade_returns, n_contracts=len(raw))))

        capital_mode = asset_capital_overrides.get(asset, default_capital_mode)
        reporting = build_reporting_portfolio(raw, capital_mode)
        if reporting is None:
            # 无法构建 sleeve 路径时，填 NaN 占位
            reporting = {
                "portfolio_path": np.array([1.0, 1.0]),
                "portfolio_simple_returns": np.array([np.nan, np.nan]),
                "portfolio_log_returns": np.array([np.nan, np.nan]),
                "sleeve_paths": np.ones((2, 1)),
                "sleeve_simple_returns": np.full((2, 1), np.nan),
                "contract_count": 0,
            }
        reporting = asset_path_variant(reporting, asset_path_mode)

        ann = annual_return_from_reporting(reporting, numerator_mode)
        mdd = float(max_drawdown_from_path(reporting["portfolio_path"]))
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
            "ann_gap": pct_err(ann, paper_helper(asset)),
            "cal_gap": pct_err(cal, paper["Calmar"]),
            "capital_mode": capital_mode,
            "n_contracts": len(raw),
        }
        s10 = sum(errors[m] < 10 for m in METRIC_NAMES)
        s15 = sum(errors[m] < 15 for m in METRIC_NAMES)
        score10 += s10
        score15 += s15
        mean_ann_errs.append(results[asset]["ann_gap"])
        mean_cal_errs.append(results[asset]["cal_gap"])

    # --- All（无 paper 目标；仅计算，不计分） ---
    raw_all = load_raw("All", overrides, excluded)
    all_trade_returns = compute_portfolio_returns(raw_all, STRAT, SIGMA, aggregation_mode="variable_n")
    all_metrics = dict(zip(METRIC_NAMES, compute_metrics(all_trade_returns, n_contracts=len(raw_all))))
    all_reporting = build_all_reporting(asset_reporting, all_mode)
    all_ann = annual_return_from_reporting(all_reporting, numerator_mode)
    all_mdd = float(max_drawdown_from_path(all_reporting["portfolio_path"]))
    all_cal = all_ann / all_mdd if all_mdd > 0 and np.isfinite(all_ann) else float("nan")
    all_metrics["MDD"] = round(all_mdd, 3)
    all_metrics["Calmar"] = round(all_cal, 3) if np.isfinite(all_cal) else float("nan")

    results["All"] = {
        "metrics": all_metrics,
        "errors": {},   # 无 paper 目标
        "misses15": [],
        "ann_gap": float("nan"),
        "cal_gap": float("nan"),
        "capital_mode": "mixed" if asset_capital_overrides else default_capital_mode,
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
# scenario() — thin wrapper（与 Long 版本接口相同）
# ---------------------------------------------------------------------------

def scenario(
    label: str,
    family: str,
    overrides: dict,
    excluded: set | list,
    default_capital_mode: str = "risk_price_source",
    asset_capital_overrides: dict | None = None,
    numerator_mode: str = "wealth_cagr",
    asset_path_mode: str = "contract_equal_path",
    all_mode: str = "contract_equal_path",
    same_rule: bool = True,
    asset_specific: bool = False,
    structural_heavy: bool = False,
    experimental: bool = False,
) -> dict:
    if asset_capital_overrides is None:
        asset_capital_overrides = {}
    summary = evaluate_scenario(
        tuple(sorted(overrides.items())),
        tuple(sorted(excluded)),
        default_capital_mode,
        tuple(sorted(asset_capital_overrides.items())),
        numerator_mode,
        asset_path_mode,
        all_mode,
    )
    return {
        "label": label,
        "family": family,
        "summary": summary,
        "overrides": dict(overrides),
        "excluded": sorted(excluded),
        "default_capital_mode": default_capital_mode,
        "asset_capital_overrides": dict(asset_capital_overrides),
        "numerator_mode": numerator_mode,
        "asset_path_mode": asset_path_mode,
        "all_mode": all_mode,
        "same_rule": same_rule,
        "asset_specific": asset_specific,
        "structural_heavy": structural_heavy,
        "experimental": experimental,
    }


# ---------------------------------------------------------------------------
# Search functions（遍历报告参数；数据源覆盖集固定在各自 base）
# ---------------------------------------------------------------------------

def search_clean_same_rule() -> list[dict]:
    """干净同规则前沿：BASE_CLEAN_OVERRIDES，遍历 numerator × all_mode。"""
    rows = []
    for numerator_mode, all_mode in itertools.product(CLEAN_NUMERATORS, ALL_MODES):
        rows.append(
            scenario(
                label=f"clean / {numerator_mode} / contract_equal_path / {all_mode}",
                family="clean_same_rule",
                overrides=BASE_CLEAN_OVERRIDES,
                excluded=BASE_CLEAN_EXCLUDED,
                numerator_mode=numerator_mode,
                asset_path_mode="contract_equal_path",
                all_mode=all_mode,
                same_rule=True,
                asset_specific=False,
                structural_heavy=False,
                experimental=False,
            )
        )
    rows.sort(key=lambda r: r["summary"]["rank"])
    return rows


def search_coherent_override() -> list[dict]:
    """允许单资产类别 capital_mode 覆盖，遍历两种 base × 资产 × numerator × all_mode。"""
    rows = []
    bases = [
        ("clean", BASE_CLEAN_OVERRIDES, BASE_CLEAN_EXCLUDED, False),
        ("structural_history", STRUCTURAL_HISTORY_OVERRIDES, STRUCTURAL_HISTORY_EXCLUDED, True),
    ]
    for base_name, overrides, excluded, structural_heavy in bases:
        for asset, numerator_mode, all_mode in itertools.product(OVERRIDE_ASSETS, CLEAN_NUMERATORS, FAST_ALL_MODES):
            rows.append(
                scenario(
                    label=f"{base_name} / {asset}:risk_price_non / {numerator_mode} / {all_mode}",
                    family="coherent_override",
                    overrides=overrides,
                    excluded=excluded,
                    asset_capital_overrides={asset: "risk_price_non"},
                    numerator_mode=numerator_mode,
                    asset_path_mode="contract_equal_path",
                    all_mode=all_mode,
                    same_rule=False,
                    asset_specific=True,
                    structural_heavy=structural_heavy,
                    experimental=True,
                )
            )
    rows.sort(key=lambda r: r["summary"]["rank"])
    return rows


def search_structural_heavy() -> list[dict]:
    """历史数据结构 heavy overrides，遍历 numerator × all_mode。"""
    rows = []
    structural_bases = [
        ("history_seed", STRUCTURAL_HISTORY_OVERRIDES, STRUCTURAL_HISTORY_EXCLUDED),
        ("clean_plus_en_es", BASE_CLEAN_OVERRIDES, set(BASE_CLEAN_EXCLUDED) | {"EN", "ES"}),
    ]
    for base_name, overrides, excluded in structural_bases:
        for numerator_mode, all_mode in itertools.product(CLEAN_NUMERATORS, FAST_ALL_MODES):
            rows.append(
                scenario(
                    label=f"{base_name} / {numerator_mode} / contract_equal_path / {all_mode}",
                    family="structural_heavy",
                    overrides=overrides,
                    excluded=excluded,
                    numerator_mode=numerator_mode,
                    asset_path_mode="contract_equal_path",
                    all_mode=all_mode,
                    same_rule=True,
                    asset_specific=False,
                    structural_heavy=True,
                    experimental=True,
                )
            )
        rows.append(
            scenario(
                label=f"{base_name} / Equity:risk_price_non / wealth_cagr / contract_equal_path",
                family="structural_heavy",
                overrides=overrides,
                excluded=excluded,
                asset_capital_overrides={"Equity Index": "risk_price_non"},
                numerator_mode="wealth_cagr",
                asset_path_mode="contract_equal_path",
                all_mode="contract_equal_path",
                same_rule=False,
                asset_specific=True,
                structural_heavy=True,
                experimental=True,
            )
        )
    rows.sort(key=lambda r: r["summary"]["rank"])
    return rows


def search_legacy_experimental() -> list[dict]:
    """
    使用 Long 策略 41/45 最优数据配置（LEGACY_EXPERIMENTAL_OVERRIDES）
    遍历全部 numerator × all_mode 组合，评估 Sign(R) 表现。
    """
    rows = []
    for numerator_mode, all_mode in itertools.product(NUMERATOR_MODES, ALL_MODES):
        rows.append(
            scenario(
                label=f"legacy_experimental / Equity:risk_price_non / {numerator_mode} / {all_mode}",
                family="legacy_experimental",
                overrides=LEGACY_EXPERIMENTAL_OVERRIDES,
                excluded=LEGACY_EXPERIMENTAL_EXCLUDED,
                asset_capital_overrides={"Equity Index": "risk_price_non"},
                numerator_mode=numerator_mode,
                asset_path_mode="contract_equal_path",
                all_mode=all_mode,
                same_rule=False,
                asset_specific=True,
                structural_heavy=True,
                experimental=True,
            )
        )
    rows.sort(key=lambda r: r["summary"]["rank"])
    return rows


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def render_misses(summary: dict) -> str:
    parts = []
    for asset in ASSETS4:
        misses = summary["results"][asset]["misses15"]
        if misses:
            parts.append(f"{asset}: {', '.join(misses)}")
    return " ; ".join(parts) if parts else "none"


def print_scenario_detail(row: dict) -> None:
    """打印单个 scenario 的完整指标与误差表。"""
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
