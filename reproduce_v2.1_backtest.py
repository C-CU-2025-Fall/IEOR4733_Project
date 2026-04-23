#!/usr/bin/env python3
"""Reproduce v2.1 DQN backtest: FI + FX, STRUCTURAL_38 preset, sigma=0.058."""
from config import ASSET_CLASSES, PAPER_TABLE3, METRIC_NAMES
from frontier_presets import STRUCTURAL_38_OVERRIDES, STRUCTURAL_38_EXCLUDED
import numpy as np, pandas as pd
from data_loader import load_clc_full
from drl.dqn.backtest.engine import _load_dqn_agent, _batched_action_ids
from drl_shared.state_space import (build_feature_matrix, compute_additive_returns,
    compute_ewma_sigma, get_feature_window, WARMUP)
from drl.dqn.spec import RETRAIN_ROUNDS
from baseline_run import compute_metrics, compute_contract_returns_from_positions

SIGMA = 0.058
trade_m = ['E(R)', 'std(R)', 'DD', 'Sharpe', 'Sortino', '% +ve', 'Ave P/L']
path_m = ['MDD', 'Calmar']

for asset in ['Fixed Income', 'Forex']:
    tickers = [t.upper() for t in ASSET_CLASSES[asset] if t.upper() not in STRUCTURAL_38_EXCLUDED]
    all_returns = {}
    for tk in tickers:
        source = STRUCTURAL_38_OVERRIDES.get(tk, 'RAD')
        df = load_clc_full(tk, source=source, start_date='2005-01-01', anchor_date='2010-01-01')
        if df is None: continue
        prices = df['Close'].to_numpy()
        returns = compute_additive_returns(prices)
        sigma = compute_ewma_sigma(returns)
        features = build_feature_matrix(prices, returns, sigma, model_version='v2.1')
        dates = df['Date'].values.astype('datetime64[D]')
        all_positions = np.zeros(len(prices))
        action_values = np.array([-1.0, 0.0, 1.0])
        for rnd_num in [1, 2]:
            try:
                agent, manifest, ckpt = _load_dqn_agent(rnd_num, tk, model_version='v2.1', device='cuda')
            except: continue
            states_list, valid_indices = [], []
            for i in range(WARMUP, len(prices)):
                w = get_feature_window(features, i)
                if w is not None: states_list.append(w); valid_indices.append(i)
            if not states_list: continue
            states = np.array(states_list)
            actions = _batched_action_ids(agent, states, 4096, False, tk)
            ri = RETRAIN_ROUNDS[rnd_num]
            start_ts, end_ts = np.datetime64(ri['test_start'][:10]), np.datetime64(ri['test_end'][:10])
            for j, idx in enumerate(valid_indices):
                if dates[idx] >= start_ts and dates[idx] <= end_ts:
                    all_positions[idx] = action_values[actions[j]]
        rd = {'tk': tk, 'rt': returns, 'sigma': sigma, 'prices': prices}
        Rt = compute_contract_returns_from_positions(rd, all_positions, SIGMA)
        test_mask = (dates >= np.datetime64('2011-01-01')) & (dates <= np.datetime64('2019-12-31'))
        all_returns[tk] = pd.Series(Rt[test_mask], index=dates[test_mask])

    df_aligned = pd.DataFrame(all_returns).dropna()
    R_portfolio = df_aligned.mean(axis=1)
    m = compute_metrics(R_portfolio.values, n_contracts=len(all_returns))
    metrics = dict(zip(METRIC_NAMES, m))
    paper = PAPER_TABLE3[asset]['Long']
    n_t = n_p = 0
    print(f"\nDQN v2.1 | STRUCTURAL_38 | sigma={SIGMA} | {asset} ({len(all_returns)} contracts)")
    print(f"  {'Metric':10} {'Ours':>10} {'Paper':>10} {'Error':>8}")
    for met in trade_m:
        o, p = metrics[met], paper[met]
        err = abs((o-p)/abs(p))*100 if p != 0 else 0
        if err <= 15: n_t += 1
        print(f"  {met:10} {o:+10.4f} {p:+10.4f} {err:6.1f}%  {'✅' if err<=15 else ''}")
    for met in path_m:
        o, p = metrics[met], paper[met]
        err = abs((o-p)/abs(p))*100 if p != 0 else 0
        if err <= 15: n_p += 1
        print(f"  {met:10} {o:+10.4f} {p:+10.4f} {err:6.1f}%  {'✅' if err<=15 else ''}")
    print(f"  Trade: {n_t}/7 | Path: {n_p}/2 | Total: {n_t+n_p}/9")
