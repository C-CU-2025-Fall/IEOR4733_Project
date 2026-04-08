"""Compare σ_tgt = 0.064 (current) vs 10% annual vs 15% annual"""
import numpy as np, pandas as pd
from data_loader import load_clc_full, extract_test_period
from strategies import strategy_long_only, strategy_sign_r, strategy_macd
from vol_scaling import compute_ewma_vol
from config import ASSET_CLASSES, BP, TRADING_DAYS

T = TRADING_DAYS
W0 = 1.0
YEARS = 9

sigmas = {
    'σ=0.064 (~101%ann)': 0.064,
    'σ=10%ann': 0.10 / np.sqrt(T),
    'σ=15%ann': 0.15 / np.sqrt(T),
}

paper = {
    'Long':   [0.504, 0.928, 0.606, 0.543, 0.831, 0.127, 0.466, 0.541, 0.928],
    'Sign(R)': [0.168, 0.799, 0.526, 0.211, 0.319, 0.299, 0.075, 0.528, 0.928],
    'MACD':   [-0.068, 0.586, 0.385, -0.117, -0.178, 0.351, -0.041, 0.519, 0.904],
}
METRIC = ['E(R)','std(R)','DD','Sharpe','Sortino','MDD','Calmar','% +ve','Ave P/L']

for label, SIGMA in sigmas.items():
    print(f"\n{'='*95}")
    print(f"  {label} (daily={SIGMA:.6f}) | additive cumsum | Calmar=realized/MDD")
    print(f"{'='*95}")

    all_rets = {'Long': [], 'Sign(R)': [], 'MACD': []}
    for tk in ASSET_CLASSES['Equity Index']:
        df = load_clc_full(tk)
        if df is None: continue
        prices = df['Close'].values.astype(float)
        p0 = prices[0]; norm_p = prices / p0
        rt = np.diff(norm_p)
        sigma = compute_ewma_vol(rt, span=60)
        t0, t1, _ = extract_test_period(df)
        if t0 is None: continue
        start = max(t0, 252)
        pct = np.diff(prices) / prices[:-1]
        for pos, key in [(strategy_long_only(len(pct)),'Long'),
                         (strategy_sign_r(pct),'Sign(R)'),
                         (strategy_macd(prices),'MACD')]:
            n = len(rt); Rt = np.zeros(n)
            for t in range(1, n):
                if sigma[t-1] > 0 and (t < 2 or sigma[t-2] > 0):
                    sp = pos[t-1] * SIGMA / sigma[t-1]
                    spp = pos[t-2] * SIGMA / sigma[t-2] if t >= 2 else 0
                    Rt[t] = sp * rt[t] - BP * norm_p[t-1] * abs(sp - spp)
            dates = df['Date'].iloc[start:t1].values[:len(Rt[start:t1])]
            all_rets[key].append((dates, Rt[start:t1]))

    N = len(all_rets['Long'])

    for strat in ['Long', 'Sign(R)', 'MACD']:
        series = [pd.Series(r, index=d) for d, r in all_rets[strat]]
        df_all = pd.DataFrame(series).T.dropna()
        R_eq = df_all.mean(axis=1).values

        er = np.mean(R_eq) * T
        std = np.std(R_eq) * np.sqrt(T)
        dd = np.sqrt(np.mean(np.minimum(R_eq, 0)**2)) * np.sqrt(T)
        sharpe = er / std if std > 0 else 0
        sortino = er / dd if dd > 0 else 0
        pct_pos = np.sum(R_eq > 0) / len(R_eq)
        pos_r = R_eq[R_eq > 0]; neg_r = R_eq[R_eq < 0]
        avg_pl = np.mean(pos_r) / abs(np.mean(neg_r)) if len(pos_r) > 0 and len(neg_r) > 0 else 0

        # MDD on additive wealth
        cumret = np.cumsum(R_eq)
        wealth = N * W0 + cumret
        pk = np.maximum.accumulate(wealth)
        mdd = float(np.max((pk - wealth) / pk))

        # Calmar = realized annual return / MDD
        realized_ann = (wealth[-1] - wealth[0]) / wealth[0] / YEARS
        calmar = realized_ann / mdd if mdd > 0 else 0

        ours = [round(v,3) for v in [er, std, dd, sharpe, sortino, mdd, calmar, pct_pos, avg_pl]]
        p = paper[strat]

        n_ok = 0
        for i in range(9):
            if p[i] != 0:
                if abs((ours[i]-p[i])/abs(p[i]))*100 < 15:
                    n_ok += 1
            else:
                n_ok += 1  # both zero

        if strat == 'Long':
            print(f"\n  {strat} ({n_ok}/9 ✅):")
            print(f"  {'Metric':8s} | {'Ours':>8s} | {'Paper':>8s} | {'%Err':>6s} | OK?")
            print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*5}")
            for i, m in enumerate(METRIC):
                o = ours[i]; pp = p[i]
                e = abs((o - pp) / abs(pp)) * 100 if pp != 0 else 0
                ok = '✅' if e < 15 else ('⚠️' if e < 30 else '❌')
                print(f"  {m:8s} | {o:>+8.3f} | {pp:>+8.3f} | {e:>5.1f}% | {ok}")
        else:
            errs = [abs((ours[i]-p[i])/abs(p[i]))*100 if p[i]!=0 else 0 for i in range(9)]
            print(f"  {strat:8s}: {n_ok}/9 ✅  E(R)={ours[0]:+.3f} std={ours[1]:.3f} Sharpe={ours[3]:+.3f} MDD={ours[5]:.3f} Calmar={ours[6]:+.3f}")
