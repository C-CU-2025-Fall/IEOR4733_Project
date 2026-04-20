from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / 'tests') not in sys.path:
    sys.path.insert(0, str(ROOT / 'tests'))

import frontier_40plus_enumeration as fe
from baseline_run import compute_contract_returns, compute_portfolio_returns, load_contracts
from config import METRIC_NAMES, PAPER_TABLE3
from metrics import compute_metrics, max_drawdown_from_path
from data_loader import load_clc_full

SIGMA = fe.SIGMA
ASSETS4 = ["Commodity", "Equity Index", "Fixed Income", "Forex"]
STRAT = "Sign(R)"

def annual_mean(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(arr.mean() * 252.0)

def annual_return_from_reporting(reporting, numerator_mode):
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

def aligned_price_p0(ticker, date0, source):
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

def build_reporting_portfolio(raw_data, capital_mode, strat):
    sleeve_paths = []
    for rd in raw_data:
        detail = compute_contract_returns(rd, strat, SIGMA, detail=True)
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
    min_len = min(len(path) for path in sleeve_paths)
    sleeves = np.column_stack([path[:min_len] for path in sleeve_paths])
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

def build_all_reporting(asset_reporting, all_mode):
    min_len = min(len(asset_reporting[a]["portfolio_path"]) for a in ASSETS4)
    if all_mode == "contract_equal_path":
        sleeves = []
        for asset in ASSETS4:
            arr = asset_reporting[asset]["sleeve_paths"]
            sleeves.append(arr[:min_len, :])
        stacked = np.column_stack(sleeves)
        path = stacked.mean(axis=1)
        sleeve_simple = np.full_like(stacked, np.nan, dtype=float)
        if min_len > 1:
            sleeve_simple[1:, :] = stacked[1:, :] / stacked[:-1, :] - 1.0
        port_simple = np.full(min_len, np.nan, dtype=float)
        port_log = np.full(min_len, np.nan, dtype=float)
        if min_len > 1:
            port_simple[1:] = path[1:] / path[:-1] - 1.0
            port_log[1:] = np.log(path[1:] / path[:-1])
        return {
            "portfolio_path": path,
            "portfolio_simple_returns": port_simple,
            "portfolio_log_returns": port_log,
            "sleeve_simple_returns": sleeve_simple,
            "sleeve_paths": stacked,
            "contract_count": stacked.shape[1],
        }
    raise ValueError(all_mode)

def pct_err(value, paper):
    if not np.isfinite(value) or not np.isfinite(paper):
        return float("inf")
    if paper == 0:
        return abs(value - paper)
    return abs((value - paper) / abs(paper)) * 100.0

def main():
    overrides = fe.LEGACY_EXPERIMENTAL_OVERRIDES
    excluded = fe.LEGACY_EXPERIMENTAL_EXCLUDED
    asset_capital_overrides = {"Equity Index": "risk_price_non"}
    numerator_mode = "annual_mean_sleeve"
    all_mode = "contract_equal_path"

    print("Sign(R) under run_legacy_41 data policy")
    print("=" * 80)
    print("overrides =", dict(sorted(overrides.items())))
    print("excluded =", sorted(excluded))
    print("equity reporting override = risk_price_non")
    print("numerator = annual_mean_sleeve")
    print("asset/all mode = contract_equal_path")
    print()

    asset_reporting = {}
    score10 = 0
    score15 = 0
    for asset in ASSETS4:
        raw = load_contracts(asset, excluded_contracts=excluded, source_overrides=overrides)
        trade_returns = compute_portfolio_returns(raw, STRAT, SIGMA, aggregation_mode="variable_n")
        trade_metrics = dict(zip(METRIC_NAMES, compute_metrics(trade_returns, n_contracts=len(raw))))
        capital_mode = asset_capital_overrides.get(asset, "risk_price_source")
        reporting = build_reporting_portfolio(raw, capital_mode, STRAT)
        ann = annual_return_from_reporting(reporting, numerator_mode)
        mdd = float(max_drawdown_from_path(reporting["portfolio_path"]))
        cal = ann / mdd if mdd > 0 and np.isfinite(ann) else float("nan")
        trade_metrics["MDD"] = round(mdd, 3)
        trade_metrics["Calmar"] = round(cal, 3) if np.isfinite(cal) else float("nan")
        asset_reporting[asset] = reporting
        paper = PAPER_TABLE3[asset][STRAT]
        errors = {metric: round(pct_err(trade_metrics[metric], paper[metric]), 2) for metric in METRIC_NAMES}
        misses15 = [metric for metric in METRIC_NAMES if errors[metric] >= 15.0]
        score10 += sum(errors[m] < 10 for m in METRIC_NAMES)
        score15 += sum(errors[m] < 15 for m in METRIC_NAMES)
        print(f"[{asset}] n={len(raw)}")
        print("metrics :", trade_metrics)
        print("paper   :", paper)
        print("err%    :", errors)
        print("miss15  :", misses15 if misses15 else ["none"])
        print()

    all_raw = []
    for asset in ASSETS4:
        all_raw.extend(load_contracts(asset, excluded_contracts=excluded, source_overrides=overrides))
    all_trade_returns = compute_portfolio_returns(all_raw, STRAT, SIGMA, aggregation_mode="variable_n")
    all_metrics = dict(zip(METRIC_NAMES, compute_metrics(all_trade_returns, n_contracts=len(all_raw))))
    all_reporting = build_all_reporting(asset_reporting, all_mode)
    all_ann = annual_return_from_reporting(all_reporting, numerator_mode)
    all_mdd = float(max_drawdown_from_path(all_reporting["portfolio_path"]))
    all_cal = all_ann / all_mdd if all_mdd > 0 and np.isfinite(all_ann) else float("nan")
    all_metrics["MDD"] = round(all_mdd, 3)
    all_metrics["Calmar"] = round(all_cal, 3) if np.isfinite(all_cal) else float("nan")
    print(f"[All] n={len(all_raw)}")
    print("metrics :", all_metrics)
    print("paper   : unavailable in current config for Table 3 / Sign(R) / All")
    print()
    print("4-asset score only (Commodity/Equity/Fixed Income/Forex)")
    print(f"<=10: {score10}/36")
    print(f"<=15: {score15}/36")

if __name__ == "__main__":
    main()
