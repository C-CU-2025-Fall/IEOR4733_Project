"""
eval_utils.py
=============
Evaluation and comparison utilities extracted from `project code.ipynb` (Cell 2).

Exports
-------
- PAPER_METRICS
- PAPER_TABLE2
- PAPER_TABLE3
- PAPER_TICKERS_BY_ASSET_CLASS
- compute_paper_metrics
- apply_portfolio_volatility_targeting
- build_portfolio_pnl
- compute_model_table2_table3_results
- compare_model_to_paper
- compare_many_models_to_paper
- evaluate_models_for_asset_class
- metrics_dict_to_frame
- paper_metrics_to_frame
"""

import numpy as np
import pandas as pd


# ============================================================
# 1) Paper reference values: Table 2 and Table 3
#    Source: Deep Reinforcement Learning for Trading, p.8 and Appendix B p.16.
# ============================================================

PAPER_METRICS = ["E(R)", "Std(R)", "DD", "Sharpe", "Sortino", "MDD", "Calmar", "% +Ret", "Ave.P/Ave.L"]

PAPER_TABLE2 = {
    "Commodity": {
        "Long":    {"E(R)": -0.710, "Std(R)": 0.979, "DD": 0.604, "Sharpe": -0.726, "Sortino": -1.177, "MDD": 0.350, "Calmar": -0.140, "% +Ret": 0.473, "Ave.P/Ave.L": 0.989},
        "Sign(R)": {"E(R)":  0.347, "Std(R)": 0.980, "DD": 0.572, "Sharpe":  0.354, "Sortino":  0.606, "MDD": 0.116, "Calmar":  0.119, "% +Ret": 0.494, "Ave.P/Ave.L": 1.084},
        "MACD":    {"E(R)": -0.171, "Std(R)": 0.978, "DD": 0.584, "Sharpe": -0.175, "Sortino": -0.293, "MDD": 0.190, "Calmar": -0.060, "% +Ret": 0.486, "Ave.P/Ave.L": 1.026},
        "DQN":     {"E(R)":  0.703, "Std(R)": 0.973, "DD": 0.552, "Sharpe":  0.723, "Sortino":  1.275, "MDD": 0.066, "Calmar":  0.501, "% +Ret": 0.498, "Ave.P/Ave.L": 1.135},
        "PG":      {"E(R)":  0.062, "Std(R)": 0.982, "DD": 0.585, "Sharpe":  0.063, "Sortino":  0.106, "MDD": 0.039, "Calmar":  0.023, "% +Ret": 0.495, "Ave.P/Ave.L": 1.029},
        "A2C":     {"E(R)":  0.223, "Std(R)": 0.955, "DD": 0.559, "Sharpe":  0.234, "Sortino":  0.399, "MDD": 0.141, "Calmar":  0.091, "% +Ret": 0.487, "Ave.P/Ave.L": 1.093},
    },
    "Equity Index": {
        "Long":    {"E(R)": 0.668, "Std(R)": 0.970, "DD": 0.606, "Sharpe": 0.688, "Sortino": 1.102, "MDD": 0.132, "Calmar": 0.509, "% +Ret": 0.542, "Ave.P/Ave.L": 0.948},
        "Sign(R)": {"E(R)": 0.228, "Std(R)": 0.966, "DD": 0.610, "Sharpe": 0.236, "Sortino": 0.374, "MDD": 0.344, "Calmar": 0.077, "% +Ret": 0.528, "Ave.P/Ave.L": 0.930},
        "MACD":    {"E(R)": 0.016, "Std(R)": 0.962, "DD": 0.618, "Sharpe": 0.017, "Sortino": 0.027, "MDD": 0.311, "Calmar": 0.006, "% +Ret": 0.519, "Ave.P/Ave.L": 0.927},
        "DQN":     {"E(R)": 0.629, "Std(R)": 0.970, "DD": 0.606, "Sharpe": 0.648, "Sortino": 1.038, "MDD": 0.161, "Calmar": 0.381, "% +Ret": 0.541, "Ave.P/Ave.L": 0.944},
        "PG":      {"E(R)": 0.432, "Std(R)": 0.967, "DD": 0.605, "Sharpe": 0.447, "Sortino": 0.714, "MDD": 0.242, "Calmar": 0.185, "% +Ret": 0.529, "Ave.P/Ave.L": 0.960},
        "A2C":     {"E(R)": 0.473, "Std(R)": 0.929, "DD": 0.593, "Sharpe": 0.510, "Sortino": 0.798, "MDD": 0.124, "Calmar": 0.328, "% +Ret": 0.533, "Ave.P/Ave.L": 0.962},
    },
    "Fixed Income": {
        "Long":    {"E(R)": 0.680, "Std(R)": 0.975, "DD": 0.576, "Sharpe": 0.698, "Sortino": 1.180, "MDD": 0.061, "Calmar": 0.444, "% +Ret": 0.515, "Ave.P/Ave.L": 1.054},
        "Sign(R)": {"E(R)": 0.214, "Std(R)": 0.972, "DD": 0.592, "Sharpe": 0.221, "Sortino": 0.363, "MDD": 0.080, "Calmar": 0.083, "% +Ret": 0.504, "Ave.P/Ave.L": 1.019},
        "MACD":    {"E(R)": 0.219, "Std(R)": 0.967, "DD": 0.579, "Sharpe": 0.228, "Sortino": 0.380, "MDD": 0.065, "Calmar": 0.123, "% +Ret": 0.486, "Ave.P/Ave.L": 1.101},
        "DQN":     {"E(R)": 0.908, "Std(R)": 0.972, "DD": 0.562, "Sharpe": 0.935, "Sortino": 1.617, "MDD": 0.062, "Calmar": 0.543, "% +Ret": 0.515, "Ave.P/Ave.L": 1.098},
        "PG":      {"E(R)": 0.705, "Std(R)": 0.974, "DD": 0.576, "Sharpe": 0.724, "Sortino": 1.225, "MDD": 0.061, "Calmar": 0.436, "% +Ret": 0.517, "Ave.P/Ave.L": 1.052},
        "A2C":     {"E(R)": 0.699, "Std(R)": 0.979, "DD": 0.582, "Sharpe": 0.714, "Sortino": 1.203, "MDD": 0.067, "Calmar": 0.408, "% +Ret": 0.517, "Ave.P/Ave.L": 1.048},
    },
    "FX": {
        "Long":    {"E(R)": -0.344, "Std(R)": 0.973, "DD": 0.583, "Sharpe": -0.353, "Sortino": -0.590, "MDD": 0.423, "Calmar": -0.097, "% +Ret": 0.491, "Ave.P/Ave.L": 0.979},
        "Sign(R)": {"E(R)": -0.297, "Std(R)": 0.973, "DD": 0.592, "Sharpe": -0.306, "Sortino": -0.502, "MDD": 0.434, "Calmar": -0.111, "% +Ret": 0.499, "Ave.P/Ave.L": 0.954},
        "MACD":    {"E(R)":  0.006, "Std(R)": 0.970, "DD": 0.582, "Sharpe":  0.007, "Sortino":  0.011, "MDD": 0.329, "Calmar":  0.002, "% +Ret": 0.493, "Ave.P/Ave.L": 1.029},
        "DQN":     {"E(R)":  0.528, "Std(R)": 0.967, "DD": 0.553, "Sharpe":  0.546, "Sortino":  0.955, "MDD": 0.183, "Calmar":  0.313, "% +Ret": 0.510, "Ave.P/Ave.L": 1.051},
        "PG":      {"E(R)":  0.248, "Std(R)": 0.967, "DD": 0.566, "Sharpe":  0.257, "Sortino":  0.438, "MDD": 0.240, "Calmar":  0.124, "% +Ret": 0.506, "Ave.P/Ave.L": 1.021},
        "A2C":     {"E(R)":  0.316, "Std(R)": 0.963, "DD": 0.563, "Sharpe":  0.328, "Sortino":  0.561, "MDD": 0.165, "Calmar":  0.201, "% +Ret": 0.507, "Ave.P/Ave.L": 1.026},
    },
    "All": {
        "Long":    {"E(R)": 0.055, "Std(R)": 0.975, "DD": 0.598, "Sharpe": 0.058, "Sortino": 0.092, "MDD": 0.071, "Calmar": 0.013, "% +Ret": 0.520, "Ave.P/Ave.L": 0.933},
        "Sign(R)": {"E(R)": 0.429, "Std(R)": 0.972, "DD": 0.582, "Sharpe": 0.441, "Sortino": 0.737, "MDD": 0.038, "Calmar": 0.201, "% +Ret": 0.510, "Ave.P/Ave.L": 1.031},
        "MACD":    {"E(R)": 0.089, "Std(R)": 0.978, "DD": 0.582, "Sharpe": 0.091, "Sortino": 0.153, "MDD": 0.008, "Calmar": 0.035, "% +Ret": 0.493, "Ave.P/Ave.L": 1.043},
        "DQN":     {"E(R)": 1.258, "Std(R)": 0.976, "DD": 0.567, "Sharpe": 1.288, "Sortino": 2.220, "MDD": 0.002, "Calmar": 1.025, "% +Ret": 0.543, "Ave.P/Ave.L": 1.043},
        "PG":      {"E(R)": 0.740, "Std(R)": 0.980, "DD": 0.593, "Sharpe": 0.754, "Sortino": 1.247, "MDD": 0.012, "Calmar": 0.480, "% +Ret": 0.533, "Ave.P/Ave.L": 0.990},
        "A2C":     {"E(R)": 1.024, "Std(R)": 0.975, "DD": 0.573, "Sharpe": 1.050, "Sortino": 1.785, "MDD": 0.007, "Calmar": 0.685, "% +Ret": 0.538, "Ave.P/Ave.L": 1.021},
    }
}

PAPER_TABLE3 = {
    "Commodity": {
        "Long":    {"E(R)": -0.298, "Std(R)": 0.412, "DD": 0.258, "Sharpe": -0.723, "Sortino": -1.152, "MDD": 0.248, "Calmar": -0.130, "% +Ret": 0.473, "Ave.P/Ave.L": 0.987},
        "Sign(R)": {"E(R)":  0.101, "Std(R)": 0.312, "DD": 0.185, "Sharpe":  0.325, "Sortino":  0.548, "MDD": 0.082, "Calmar":  0.115, "% +Ret": 0.494, "Ave.P/Ave.L": 1.081},
        "MACD":    {"E(R)": -0.039, "Std(R)": 0.227, "DD": 0.136, "Sharpe": -0.174, "Sortino": -0.290, "MDD": 0.132, "Calmar": -0.059, "% +Ret": 0.486, "Ave.P/Ave.L": 1.024},
        "DQN":     {"E(R)":  0.187, "Std(R)": 0.301, "DD": 0.173, "Sharpe":  0.623, "Sortino":  1.085, "MDD": 0.039, "Calmar":  0.413, "% +Ret": 0.498, "Ave.P/Ave.L": 1.119},
        "PG":      {"E(R)":  0.013, "Std(R)": 0.287, "DD": 0.172, "Sharpe":  0.047, "Sortino":  0.078, "MDD": 0.072, "Calmar":  0.017, "% +Ret": 0.495, "Ave.P/Ave.L": 1.026},
        "A2C":     {"E(R)":  0.072, "Std(R)": 0.163, "DD": 0.098, "Sharpe":  0.440, "Sortino":  0.729, "MDD": 0.099, "Calmar":  0.161, "% +Ret": 0.487, "Ave.P/Ave.L": 1.151},
    },
    "Equity Index": {
        "Long":    {"E(R)": 0.504, "Std(R)": 0.928, "DD": 0.606, "Sharpe": 0.543, "Sortino": 0.831, "MDD": 0.127, "Calmar": 0.466, "% +Ret": 0.541, "Ave.P/Ave.L": 0.928},
        "Sign(R)": {"E(R)": 0.168, "Std(R)": 0.799, "DD": 0.526, "Sharpe": 0.211, "Sortino": 0.319, "MDD": 0.299, "Calmar": 0.075, "% +Ret": 0.528, "Ave.P/Ave.L": 0.928},
        "MACD":    {"E(R)":-0.068, "Std(R)": 0.586, "DD": 0.385, "Sharpe":-0.117, "Sortino":-0.178, "MDD": 0.351, "Calmar":-0.041, "% +Ret": 0.519, "Ave.P/Ave.L": 0.904},
        "DQN":     {"E(R)": 0.461, "Std(R)": 0.933, "DD": 0.611, "Sharpe": 0.494, "Sortino": 0.754, "MDD": 0.170, "Calmar": 0.308, "% +Ret": 0.541, "Ave.P/Ave.L": 0.922},
        "PG":      {"E(R)": 0.320, "Std(R)": 0.875, "DD": 0.574, "Sharpe": 0.366, "Sortino": 0.558, "MDD": 0.211, "Calmar": 0.183, "% +Ret": 0.529, "Ave.P/Ave.L": 0.949},
        "A2C":     {"E(R)": 0.293, "Std(R)": 0.629, "DD": 0.427, "Sharpe": 0.466, "Sortino": 0.686, "MDD": 0.193, "Calmar": 0.214, "% +Ret": 0.533, "Ave.P/Ave.L": 0.965},
    },
    "Fixed Income": {
        "Long":    {"E(R)": 0.605, "Std(R)": 0.939, "DD": 0.561, "Sharpe": 0.645, "Sortino": 1.081, "MDD": 0.108, "Calmar": 0.455, "% +Ret": 0.515, "Ave.P/Ave.L": 1.048},
        "Sign(R)": {"E(R)": 0.189, "Std(R)": 0.795, "DD": 0.496, "Sharpe": 0.237, "Sortino": 0.381, "MDD": 0.165, "Calmar": 0.103, "% +Ret": 0.504, "Ave.P/Ave.L": 1.024},
        "MACD":    {"E(R)": 0.136, "Std(R)": 0.609, "DD": 0.367, "Sharpe": 0.224, "Sortino": 0.371, "MDD": 0.124, "Calmar": 0.131, "% +Ret": 0.485, "Ave.P/Ave.L": 1.102},
        "DQN":     {"E(R)": 0.734, "Std(R)": 0.862, "DD": 0.508, "Sharpe": 0.851, "Sortino": 1.445, "MDD": 0.118, "Calmar": 0.469, "% +Ret": 0.515, "Ave.P/Ave.L": 1.086},
        "PG":      {"E(R)": 0.624, "Std(R)": 0.938, "DD": 0.561, "Sharpe": 0.665, "Sortino": 1.113, "MDD": 0.109, "Calmar": 0.443, "% +Ret": 0.517, "Ave.P/Ave.L": 1.043},
        "A2C":     {"E(R)": 0.852, "Std(R)": 1.345, "DD": 0.806, "Sharpe": 0.633, "Sortino": 1.057, "MDD": 0.128, "Calmar": 0.397, "% +Ret": 0.517, "Ave.P/Ave.L": 1.039},
    },
    "FX": {
        "Long":    {"E(R)":-0.198, "Std(R)": 0.472, "DD": 0.285, "Sharpe":-0.420, "Sortino":-0.696, "MDD": 0.219, "Calmar":-0.101, "% +Ret": 0.491, "Ave.P/Ave.L": 0.966},
        "Sign(R)": {"E(R)":-0.113, "Std(R)": 0.551, "DD": 0.341, "Sharpe":-0.207, "Sortino":-0.332, "MDD": 0.170, "Calmar":-0.071, "% +Ret": 0.499, "Ave.P/Ave.L": 0.968},
        "MACD":    {"E(R)": 0.016, "Std(R)": 0.424, "DD": 0.259, "Sharpe": 0.037, "Sortino": 0.061, "MDD": 0.156, "Calmar": 0.016, "% +Ret": 0.493, "Ave.P/Ave.L": 1.034},
        "DQN":     {"E(R)": 0.272, "Std(R)": 0.487, "DD": 0.280, "Sharpe": 0.560, "Sortino": 0.972, "MDD": 0.085, "Calmar": 0.326, "% +Ret": 0.510, "Ave.P/Ave.L": 1.058},
        "PG":      {"E(R)": 0.157, "Std(R)": 0.533, "DD": 0.312, "Sharpe": 0.295, "Sortino": 0.503, "MDD": 0.098, "Calmar": 0.148, "% +Ret": 0.506, "Ave.P/Ave.L": 1.029},
        "A2C":     {"E(R)": 0.159, "Std(R)": 0.455, "DD": 0.267, "Sharpe": 0.349, "Sortino": 0.592, "MDD": 0.081, "Calmar": 0.193, "% +Ret": 0.507, "Ave.P/Ave.L": 1.034},
    },
    "All": {
        "Long":    {"E(R)":-0.013, "Std(R)": 0.363, "DD": 0.230, "Sharpe":-0.036, "Sortino":-0.057, "MDD": 0.037, "Calmar":-0.009, "% +Ret": 0.519, "Ave.P/Ave.L": 0.919},
        "Sign(R)": {"E(R)": 0.086, "Std(R)": 0.296, "DD": 0.186, "Sharpe": 0.291, "Sortino": 0.461, "MDD": 0.016, "Calmar": 0.142, "% +Ret": 0.510, "Ave.P/Ave.L": 1.008},
        "MACD":    {"E(R)":-0.018, "Std(R)": 0.230, "DD": 0.143, "Sharpe":-0.080, "Sortino":-0.129, "MDD": 0.026, "Calmar":-0.029, "% +Ret": 0.493, "Ave.P/Ave.L": 1.013},
        "DQN":     {"E(R)": 0.318, "Std(R)": 0.252, "DD": 0.150, "Sharpe": 1.258, "Sortino": 2.111, "MDD": 0.008, "Calmar": 1.023, "% +Ret": 0.543, "Ave.P/Ave.L": 1.041},
        "PG":      {"E(R)": 0.168, "Std(R)": 0.279, "DD": 0.174, "Sharpe": 0.602, "Sortino": 0.968, "MDD": 0.011, "Calmar": 0.373, "% +Ret": 0.533, "Ave.P/Ave.L": 0.968},
        "A2C":     {"E(R)": 0.214, "Std(R)": 0.221, "DD": 0.134, "Sharpe": 0.969, "Sortino": 1.601, "MDD": 0.009, "Calmar": 0.672, "% +Ret": 0.538, "Ave.P/Ave.L": 1.014},
    }
}


# ============================================================
# 2) Asset-class ticker map from Appendix A
# ============================================================

PAPER_TICKERS_BY_ASSET_CLASS = {
    "Commodity": ["CC","DA","GI","JO","KC","KW","LB","NR","SB","ZA","ZC","ZF","ZG","ZH","ZI","ZK","ZL","ZN","ZO","ZP","ZR","ZT","ZU","ZW","ZZ"],
    "Equity Index": ["CA","EN","ER","ES","LX","MD","SC","SP","XU","XX","YM"],
    "Fixed Income": ["DT","FB","TY","UB","US"],
    "FX": ["AN","BN","CN","DX","FN","JN","MP","NK","SN"],
}
PAPER_TICKERS_BY_ASSET_CLASS["All"] = sorted(sum(PAPER_TICKERS_BY_ASSET_CLASS.values(), []))


# ============================================================
# 3) Metrics exactly matching the paper's list
# ============================================================

def compute_paper_metrics(daily_returns, periods_per_year=252):
    """
    daily_returns: 1D pandas Series of portfolio daily trade returns
    Returns a dict with the 9 metrics listed in the paper.
    """
    r = pd.Series(daily_returns).dropna().astype(float)
    if len(r) == 0:
        return {k: np.nan for k in PAPER_METRICS}

    ann_mean = r.mean() * periods_per_year
    ann_std = r.std(ddof=1) * np.sqrt(periods_per_year)

    neg_r = r[r < 0]
    if len(neg_r) > 1:
        downside_dev = neg_r.std(ddof=1) * np.sqrt(periods_per_year)
    else:
        downside_dev = np.nan

    sharpe = ann_mean / ann_std if pd.notna(ann_std) and ann_std != 0 else np.nan
    sortino = ann_mean / downside_dev if pd.notna(downside_dev) and downside_dev != 0 else np.nan

    cum = r.cumsum()
    running_max = cum.cummax()
    drawdown = running_max - cum
    mdd = drawdown.max() if len(drawdown) else np.nan

    calmar = ann_mean / mdd if pd.notna(mdd) and mdd != 0 else np.nan

    pct_pos = (r > 0).mean()

    pos_mean = r[r > 0].mean() if (r > 0).any() else np.nan
    neg_mean_abs = (-r[r < 0]).mean() if (r < 0).any() else np.nan
    ave_p_ave_l = pos_mean / neg_mean_abs if pd.notna(pos_mean) and pd.notna(neg_mean_abs) and neg_mean_abs != 0 else np.nan

    return {
        "E(R)": ann_mean,
        "Std(R)": ann_std,
        "DD": downside_dev,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MDD": mdd,
        "Calmar": calmar,
        "% +Ret": pct_pos,
        "Ave.P/Ave.L": ave_p_ave_l,
    }


# ============================================================
# 4) Portfolio-level volatility scaling for Table 2
#    The paper says Table 2 adds an additional portfolio-level
#    volatility scaling layer so different methods have the same target vol.
#    It does not spell out one exact implementation formula here,
#    so this is an implementation assumption.
# ============================================================

def apply_portfolio_volatility_targeting(
    portfolio_df,
    return_col="portfolio_pnl",
    target_vol=0.975,      # implementation choice, chosen to match Table 2 scale
    ewm_span=60,
    periods_per_year=252,
    out_col="portfolio_pnl_vol_scaled"
):
    """
    Adds a portfolio-level volatility scaling layer.

    Assumption:
    scale_t = target_vol / ann_vol_est_t
    where ann_vol_est_t comes from 60-day EWMA std of the portfolio daily returns.
    Then scaled_return_t = scale_{t-1} * return_t
    """
    out = portfolio_df.copy().sort_values("date").reset_index(drop=True)
    r = out[return_col].astype(float)

    daily_vol_est = r.ewm(span=ewm_span, adjust=False, min_periods=ewm_span).std()
    ann_vol_est = daily_vol_est * np.sqrt(periods_per_year)

    scale = target_vol / ann_vol_est.replace(0, np.nan)
    scale = scale.shift(1)  # no look-ahead

    out[out_col] = scale * r
    return out


# ============================================================
# 5) Existing portfolio aggregation, kept generic
# ============================================================

def build_portfolio_pnl(pnl_by_ticker, pnl_col="net_pnl"):
    frames = []
    for ticker, df in pnl_by_ticker.items():
        tmp = df[["date", pnl_col]].copy()
        tmp["ticker"] = ticker
        frames.append(tmp)

    panel = pd.concat(frames, ignore_index=True)

    port = (
        panel.groupby("date", as_index=False)[pnl_col]
        .mean()
        .rename(columns={pnl_col: "portfolio_pnl"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    return panel, port


# ============================================================
# 6) Compute one model's results in Table 2 / Table 3 style
# ============================================================

def compute_model_table2_table3_results(
    pnl_by_ticker,
    periods_per_year=252,
    table2_target_vol=0.975,
    table2_ewm_span=60,
):
    """
    pnl_by_ticker: dict[ticker] -> DataFrame with at least ['date', 'net_pnl']

    Returns:
        {
            "table3_portfolio": raw portfolio df,
            "table2_portfolio": portfolio df after vol targeting,
            "table3_metrics": {...},
            "table2_metrics": {...},
        }
    """
    _, port_raw = build_portfolio_pnl(pnl_by_ticker, pnl_col="net_pnl")

    table3_metrics = compute_paper_metrics(
        port_raw["portfolio_pnl"],
        periods_per_year=periods_per_year
    )

    port_t2 = apply_portfolio_volatility_targeting(
        port_raw,
        return_col="portfolio_pnl",
        target_vol=table2_target_vol,
        ewm_span=table2_ewm_span,
        periods_per_year=periods_per_year,
        out_col="portfolio_pnl_vol_scaled"
    )

    table2_metrics = compute_paper_metrics(
        port_t2["portfolio_pnl_vol_scaled"],
        periods_per_year=periods_per_year
    )

    return {
        "table3_portfolio": port_raw,
        "table2_portfolio": port_t2,
        "table3_metrics": table3_metrics,
        "table2_metrics": table2_metrics,
    }


# ============================================================
# 7) Compare your result vs paper
# ============================================================

def _safe_pct_diff(yours, paper):
    if pd.isna(yours) or pd.isna(paper):
        return np.nan
    if paper == 0:
        return np.nan
    return (yours - paper) / abs(paper) * 100.0


def compare_model_to_paper(
    your_metrics,
    paper_table,
    asset_class,
    model_name
):
    """
    your_metrics: dict metric -> value
    paper_table: PAPER_TABLE2 or PAPER_TABLE3
    """
    paper_metrics = paper_table[asset_class][model_name]

    rows = []
    for m in PAPER_METRICS:
        y = your_metrics.get(m, np.nan)
        p = paper_metrics.get(m, np.nan)
        rows.append({
            "asset_class": asset_class,
            "model": model_name,
            "metric": m,
            "your_value": y,
            "paper_value": p,
            "pct_diff_vs_paper": _safe_pct_diff(y, p),
            "abs_diff": y - p if pd.notna(y) and pd.notna(p) else np.nan,
        })

    return pd.DataFrame(rows)


# ============================================================
# 8) Batch comparison across many models
#    Ready for Long / Sign(R) / MACD / DQN / PG / A2C
# ============================================================

def compare_many_models_to_paper(
    results_by_model,
    asset_class,
):
    """
    results_by_model example:
    {
        "Long": {"table2_metrics": {...}, "table3_metrics": {...}},
        "Sign(R)": {...},
        "MACD": {...},
        "A2C": {...},
        ...
    }

    Returns:
        compare_t2_df, compare_t3_df
    """
    t2_frames = []
    t3_frames = []

    for model_name, res in results_by_model.items():
        if model_name in PAPER_TABLE2[asset_class]:
            t2_frames.append(
                compare_model_to_paper(
                    your_metrics=res["table2_metrics"],
                    paper_table=PAPER_TABLE2,
                    asset_class=asset_class,
                    model_name=model_name,
                )
            )

        if model_name in PAPER_TABLE3[asset_class]:
            t3_frames.append(
                compare_model_to_paper(
                    your_metrics=res["table3_metrics"],
                    paper_table=PAPER_TABLE3,
                    asset_class=asset_class,
                    model_name=model_name,
                )
            )

    compare_t2_df = pd.concat(t2_frames, ignore_index=True) if t2_frames else pd.DataFrame()
    compare_t3_df = pd.concat(t3_frames, ignore_index=True) if t3_frames else pd.DataFrame()

    return compare_t2_df, compare_t3_df


# ============================================================
# 9) Convenience helper:
#    build per-model per-asset-class results from raw pnl dicts
# ============================================================

def evaluate_models_for_asset_class(
    pnl_by_model_by_ticker,
    asset_class,
    periods_per_year=252,
    table2_target_vol=0.975,
    table2_ewm_span=60,
):
    """
    pnl_by_model_by_ticker example:
    {
        "Long":    {"ES": df, "TY": df, ...},
        "Sign(R)": {"ES": df, "TY": df, ...},
        "MACD":    {...},
        "A2C":     {...},
        "DQN":     {...},   # later
        "PG":      {...},   # later
    }

    Only keeps the tickers belonging to asset_class.
    """
    tickers = set(PAPER_TICKERS_BY_ASSET_CLASS[asset_class])
    results = {}

    for model_name, pnl_by_ticker in pnl_by_model_by_ticker.items():
        sub = {k: v for k, v in pnl_by_ticker.items() if k in tickers}
        if len(sub) == 0:
            continue

        results[model_name] = compute_model_table2_table3_results(
            sub,
            periods_per_year=periods_per_year,
            table2_target_vol=table2_target_vol,
            table2_ewm_span=table2_ewm_span,
        )

    return results


# ============================================================
# 10) Pretty summary table: metrics as columns
# ============================================================

def metrics_dict_to_frame(results_by_model, table_key="table2_metrics"):
    rows = []
    for model_name, res in results_by_model.items():
        row = {"model": model_name}
        row.update(res[table_key])
        rows.append(row)
    return pd.DataFrame(rows)


def paper_metrics_to_frame(paper_table, asset_class):
    rows = []
    for model_name, metrics in paper_table[asset_class].items():
        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)
