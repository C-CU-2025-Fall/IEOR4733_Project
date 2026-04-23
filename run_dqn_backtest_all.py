#!/usr/bin/env python3
"""DQN Walk-Forward Backtest for ALL 4 asset classes."""
import sys, os, numpy as np, pandas as pd, torch, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ASSET_CLASSES, PAPER_TABLE3, METRIC_NAMES, EWMA_SPAN, BP
from baseline_run import compute_metrics
from data_loader import load_clc_full
from strategy_dqn import DQNAgent, build_all_features, get_feature_window, WARMUP

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'dqn_walkforward')

def load_model(ticker, round_num, device):
    path = os.path.join(MODEL_DIR, f"{ticker}_r{round_num}.pt")
    if not os.path.exists(path):
        return None
    agent = DQNAgent()
    agent.q_net.to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if 'q' in ckpt:
        agent.q_net.load_state_dict(ckpt['q'])
        agent.target.load_state_dict(ckpt['t'])
    else:
        agent.q_net.load_state_dict(ckpt.get('q_net', ckpt))
        agent.target.load_state_dict(ckpt.get('target_net', ckpt))
    agent.q_net.eval()
    return agent

def get_scaled_returns(ticker, round_num, sigma_tgt, period_start, period_end):
    df = load_clc_full(ticker, source='RAD', start_date='2005-01-01', anchor_date='2010-01-01')
    if df is None:
        return None, None
    prices = df['Close'].to_numpy()
    returns = np.zeros(len(prices)); returns[1:] = prices[1:] - prices[:-1]
    sigma = pd.Series(returns).ewm(span=EWMA_SPAN, adjust=False).std().values
    features = build_all_features(prices, returns, sigma)
    dates = pd.to_datetime(df['Date'])

    agent = load_model(ticker, round_num, DEVICE)
    if agent is None:
        return None, None

    scaled_returns = []
    valid_dates = []
    last_action = 0
    last_sig = sigma[WARMUP-1] if WARMUP >= 1 else sigma[0]

    period_start_ts = pd.Timestamp(period_start)
    period_end_ts = pd.Timestamp(period_end)

    for idx in range(WARMUP, len(prices)):
        date = dates.iloc[idx]
        if date < period_start_ts or date > period_end_ts:
            state = get_feature_window(features, idx)
            if state is not None:
                state_t = torch.from_numpy(state).unsqueeze(0).float().to(DEVICE)
                with torch.no_grad():
                    action = agent.q_net(state_t).argmax().item()
                last_action = int(action)
                last_sig = sigma[idx-1] if idx >= 1 else sigma[0]
            continue

        state = get_feature_window(features, idx)
        if state is None:
            continue

        state_t = torch.from_numpy(state).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            action = agent.q_net(state_t).argmax().item()

        sig_t_1 = sigma[idx-1]
        if sig_t_1 > 0:
            pos_current = action * (sigma_tgt / sig_t_1)
            pos_prev = last_action * (sigma_tgt / last_sig) if last_sig > 0 else 0
            gross = pos_current * returns[idx]
            tc = BP * prices[idx-1] * abs(pos_current - pos_prev)
            scaled_returns.append(gross - tc)
            valid_dates.append(date)

        last_sig = sig_t_1
        last_action = action

    return np.array(scaled_returns), valid_dates

if __name__ == '__main__':
    sigma_tgt = 0.058
    total_ok = 0
    total_metrics = 9

    for asset in ['Commodity', 'Equity Index', 'Fixed Income', 'Forex']:
        tickers = ASSET_CLASSES.get(asset, [])
        all_returns = {}

        t0 = time.time()
        print(f"\n{'='*70}")
        print(f"DQN Walk-Forward: {asset} ({len(tickers)} contracts)")
        print(f"Device: {DEVICE}, sigma_tgt={sigma_tgt}")
        print(f"{'='*70}")

        for tk in tickers:
            R1, d1 = get_scaled_returns(tk, 1, sigma_tgt, '2011-01-01', '2014-12-31')
            R2, d2 = get_scaled_returns(tk, 2, sigma_tgt, '2015-01-01', '2019-12-31')

            if R1 is None or R2 is None:
                print(f"  ❌ {tk}: failed")
                continue

            all_d = list(d1) + list(d2)
            all_R = np.concatenate([R1, R2])
            all_returns[tk] = pd.Series(all_R, index=all_d)
            print(f"  ✅ {tk}: {len(all_R)} days")

        t1 = time.time()
        print(f"\nInference: {t1-t0:.1f}s")

        if not all_returns:
            print("No contracts!")
            continue

        df = pd.DataFrame(all_returns).dropna()
        print(f"Aligned: {len(df)} trading days, {len(all_returns)} contracts")

        R_portfolio = df.mean(axis=1)
        m = compute_metrics(R_portfolio, n_contracts=len(all_returns))
        metrics_dqn = dict(zip(METRIC_NAMES, m))
        paper = PAPER_TABLE3[asset]['Long']

        n_ok = 0
        for met in ['E(R)', 'std(R)', 'DD', 'Sharpe', 'Sortino', 'MDD', 'Calmar', '% +ve', 'Ave P/L']:
            ours, p = metrics_dqn[met], paper[met]
            err = abs((ours - p) / abs(p)) * 100 if p != 0 else 0
            ok = "✅" if err <= 15 else "❌"
            if err <= 15:
                n_ok += 1
            print(f"  {met:10}: {ours:+9.4f} vs {p:+9.4f}  err={err:6.1f}%  {ok}")

        total_ok += n_ok
        print(f"\n  >>> {n_ok}/9 metrics <=15%")

    print(f"\n{'='*70}")
    print(f"TOTAL: {total_ok}/{total_metrics*4} metrics aligned (4 assets x 9 metrics)")
    print(f"{'='*70}")
