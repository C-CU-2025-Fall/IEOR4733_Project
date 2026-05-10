import json, csv, math, os
from statistics import median

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'gamma_tuning')
METRICS = ['E(R)','std(R)','DD','Sharpe','Sortino','MDD','Calmar','% +ve','Ave P/L']
GAMMAS = [0.5, 0.6, 0.7]
SEEDS = [42, 43, 44, 45, 46]
RNDS = [1, 2]

def load_all():
    rows = []
    for r in RNDS:
        for g in GAMMAS:
            for s in SEEDS:
                path = f'{BASE}/backtest_r{r}_{g}_{s}.json'
                with open(path) as f:
                    d = json.load(f)
                    d['round'] = r
                    rows.append(d)
    return rows

def q25(vals):
    n = len(vals)
    idx = (n - 1) * 0.25
    lo = int(idx)
    hi = lo + 1
    frac = idx - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac if hi < n else vals[lo]

def q75(vals):
    n = len(vals)
    idx = (n - 1) * 0.75
    lo = int(idx)
    hi = lo + 1
    frac = idx - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac if hi < n else vals[lo]

def quartile_stats(vals):
    sv = sorted(vals)
    return median(sv), q25(sv), q75(sv)

def write_summary(rows):
    with open(f'{BASE}/summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['gamma','metric','r1_median','r1_q1','r1_q3','r2_median','r2_q1','r2_q3'])
        for g in GAMMAS:
            for m in METRICS:
                r1v = [r[m] for r in rows if r['gamma'] == g and r['round'] == 1]
                r2v = [r[m] for r in rows if r['gamma'] == g and r['round'] == 2]
                r1m, r1q1, r1q3 = quartile_stats(r1v)
                r2m, r2q1, r2q3 = quartile_stats(r2v)
                w.writerow([g, m, r1m, r1q1, r1q3, r2m, r2q1, r2q3])

def write_per_seed(rows):
    with open(f'{BASE}/per_seed.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['gamma','seed','metric','r1_value','r2_value'])
        for g in GAMMAS:
            for s in SEEDS:
                for m in METRICS:
                    r1v = [r[m] for r in rows if r['gamma'] == g and r['seed'] == s and r['round'] == 1][0]
                    r2v = [r[m] for r in rows if r['gamma'] == g and r['seed'] == s and r['round'] == 2][0]
                    w.writerow([g, s, m, r1v, r2v])

def write_topk(rows):
    by_gamma = {}
    for g in GAMMAS:
        r2_all = [r['Sharpe'] for r in rows if r['gamma'] == g and r['round'] == 2]
        r1_all = [r['Sharpe'] for r in rows if r['gamma'] == g and r['round'] == 1]
        all_median_r2 = median(r2_all)
        all_median_r1 = median(r1_all)
        seed_sharpe = [(s, r['Sharpe']) for s in SEEDS
                       for r in rows if r['gamma'] == g and r['seed'] == s and r['round'] == 2]
        seed_sharpe.sort(key=lambda x: x[1], reverse=True)
        top3_seeds = [s for s, _ in seed_sharpe[:3]]
        top3_r2 = [sh for _, sh in seed_sharpe[:3]]
        top3_median_r2 = median(top3_r2)
        top3_median_r1 = median([r['Sharpe'] for s in top3_seeds
                                 for r in rows if r['gamma'] == g and r['seed'] == s and r['round'] == 1])
        by_gamma[g] = {
            'top_3_seeds': top3_seeds,
            'top_3_median_r1_sharpe': top3_median_r1,
            'top_3_median_r2_sharpe': top3_median_r2,
            'all_seed_median_r1_sharpe': all_median_r1,
            'all_seed_median_r2_sharpe': all_median_r2
        }
    best_g = max(GAMMAS, key=lambda g: by_gamma[g]['all_seed_median_r2_sharpe'])
    sharpe_vals = sorted([by_gamma[g]['all_seed_median_r2_sharpe'] for g in GAMMAS], reverse=True)
    decision = 'clear_winner' if sharpe_vals[0] > sharpe_vals[1] + 0.05 else 'unclear'
    out = {
        'best_gamma': best_g,
        'by_gamma': by_gamma,
        'decision': decision
    }
    with open(f'{BASE}/topk_models.json', 'w') as f:
        json.dump(out, f, indent=2)

def validate(rows):
    for r in rows:
        for m in METRICS:
            v = r[m]
            if not math.isfinite(v):
                raise ValueError(f'Non-finite {m} in {r}')
    for g in GAMMAS:
        for m in METRICS:
            r1v = [r[m] for r in rows if r['gamma'] == g and r['round'] == 1]
            r2v = [r[m] for r in rows if r['gamma'] == g and r['round'] == 2]
            for vals, label in [(r1v, 'r1'), (r2v, 'r2')]:
                q1 = q25(sorted(vals))
                q3 = q75(sorted(vals))
                if q3 <= q1:
                    raise ValueError(f'IQR <= 0 for gamma={g} metric={m} {label}: q1={q1} q3={q3}')

rows = load_all()
print(f'Loaded {len(rows)} rows')
validate(rows)
print('Validation passed')
write_summary(rows)
print('Wrote summary.csv')
write_per_seed(rows)
print('Wrote per_seed.csv')
write_topk(rows)
print('Wrote topk_models.json')
