"""v9 calibration audit. Read-only: touches no existing file, trains no model.

WHY CALIBRATION IS A SEPARATE QUESTION FROM EVERYTHING BEFORE IT
---------------------------------------------------------------
Every study in this project so far chased Resolution and failed. Calibration
is the other term. Murphy:

    BSS = Resolution - Reliability

and v9's Reliability is NOT small: 155.0 on the 2023 fold, 42.9 on 2024. If a
calibration map could drive Reliability to zero those are the BSS points on
the table.

THE INVARIANCE THAT DEFINES THE WHOLE AUDIT
-------------------------------------------
`metrics.decomp` bins by `pd.qcut(p, 50)` -- quantiles of the prediction. Any
STRICTLY MONOTONE map leaves every row in the same quantile bin, so the bins'
actual rates and weights are unchanged and

    Resolution is exactly invariant under monotone calibration.

Temperature, Platt and isotonic are all monotone. So none of them can add a
single point of Resolution, and the entire achievable gain is bounded above by
the current Reliability. That bound is what makes this cheap to settle: fit
the map in-sample to see how close to the bound one can get, then ask the only
question that matters, whether the map transfers forward a season.

    python research/calibration_analysis/calib_audit.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.deep_learning_state.metrics import decomp, paired  # noqa: E402

OUT = Path(__file__).resolve().parent
EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def load(fold):
    f = REPO / 'data' / 'folds' / f'fold_{fold}_va.parquet'
    d = pd.read_parquet(f, columns=['row_id', 'season', 'pitcher_id',
                                    'control_success', 'p_v9'])
    return (d.control_success.to_numpy(np.float64),
            d.p_v9.to_numpy(np.float64),
            d.pitcher_id.to_numpy())


# ---------------------------------------------------------------- fitters
def fit_temperature(y, p):
    """p' = sigmoid(a * logit(p)). One parameter, no intercept."""
    z = logit(p)
    lo, hi = 0.05, 5.0
    for _ in range(60):                      # golden-section on log-loss
        m1 = lo + (hi - lo) * 0.382
        m2 = lo + (hi - lo) * 0.618
        f = lambda a: _nll(y, sigmoid(a * z))
        if f(m1) < f(m2):
            hi = m2
        else:
            lo = m1
    a = (lo + hi) / 2
    return dict(kind='temperature', a=a), lambda q: sigmoid(a * logit(q))


def fit_platt(y, p):
    """p' = sigmoid(a * logit(p) + b). Newton on the 2-parameter log-loss."""
    z = logit(p)
    a, b = 1.0, 0.0
    X = np.column_stack([z, np.ones_like(z)])
    for _ in range(100):
        q = sigmoid(X @ np.array([a, b]))
        g = X.T @ (q - y)
        W = q * (1 - q)
        H = X.T @ (X * W[:, None]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        a, b = np.array([a, b]) - step
        if np.abs(step).max() < 1e-10:
            break
    return dict(kind='platt', a=float(a), b=float(b)), \
        lambda q: sigmoid(a * logit(q) + b)


def fit_isotonic(y, p):
    """Monotone step map, PAVA. The most flexible monotone calibrator, so its
    in-sample fit is the practical ceiling."""
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds='clip', y_min=0.001, y_max=0.999)
    ir.fit(p, y)
    return dict(kind='isotonic', n_knots=len(np.unique(ir.f_.x))), \
        lambda q: np.clip(ir.predict(q), 0.001, 0.999)


def _nll(y, q):
    q = np.clip(q, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


FITTERS = {'temperature': fit_temperature, 'platt': fit_platt,
           'isotonic': fit_isotonic}


# ---------------------------------------------------------------- reporting
def row(tag, y, p, base):
    res, rel, bss = decomp(y, p)
    b_res, b_rel, b_bss = decomp(y, base)
    m, se = paired(y, base, p)
    return dict(tag=tag, BSS=bss, Resolution=res, Reliability=rel,
                dBSS=bss - b_bss, dResolution=res - b_res,
                dReliability=rel - b_rel, paired=m, paired_se=se,
                sigma=abs(m) / se if se > 0 else 0.0, nll=_nll(y, p))


def show(r):
    print(f'  {r["tag"]:<34s} BSS {r["BSS"]:8.1f}  Res {r["Resolution"]:8.1f}  '
          f'Rel {r["Reliability"]:7.1f} | dBSS {r["dBSS"]:+7.1f}  '
          f'dRes {r["dResolution"]:+6.1f}  dRel {r["dReliability"]:+7.1f}  '
          f'{r["paired"]:+6.1f}±{r["paired_se"]:4.1f} {r["sigma"]:4.1f}s')


def reliability_table(y, p, nb=20):
    q = pd.qcut(p, nb, labels=False, duplicates='drop')
    g = pd.DataFrame({'p': p, 'y': y, 'q': q}).groupby('q').agg(
        n=('y', 'size'), pred=('p', 'mean'), actual=('y', 'mean'),
        lo=('p', 'min'), hi=('p', 'max'))
    g['gap'] = g.actual - g.pred
    return g


def main():
    print('=' * 104)
    print('  v9 CALIBRATION AUDIT   (읽기 전용, 기존 파일 무수정)')
    print('=' * 104)

    data = {S: load(S) for S in (2023, 2024)}
    rows = []

    # ---- 1. distribution -------------------------------------------------
    print('\n### 1. v9 prediction distribution')
    for S, (y, p, _) in data.items():
        qs = np.quantile(p, [0, .01, .05, .25, .5, .75, .95, .99, 1])
        print(f'  {S}  n {len(p):,}   mean {p.mean():.4f}  sd {p.std():.4f}   '
              f'y mean {y.mean():.4f}   bias {p.mean()-y.mean():+.4f}')
        print(f'        quantiles ' + ' '.join(f'{v:.3f}' for v in qs))
        rows.append(dict(section='dist', season=S, n=len(p), mean=p.mean(),
                         sd=p.std(), y_mean=y.mean(), bias=p.mean()-y.mean()))

    # ---- 2. reliability curve -------------------------------------------
    print('\n### 2. reliability curve (20 quantile bins)')
    for S, (y, p, _) in data.items():
        t = reliability_table(y, p)
        res, rel, bss = decomp(y, p)
        print(f'\n  --- {S}   BSS {bss:.1f}  Res {res:.1f}  Rel {rel:.1f} ---')
        print('   bin      n     pred   actual      gap')
        for i, r in t.iterrows():
            bar = '#' * min(30, int(abs(r.gap) * 400))
            print(f'   {int(i):3d} {int(r.n):7,}   {r.pred:.4f}   '
                  f'{r.actual:.4f}  {r.gap:+.4f}  {bar}')
        t.assign(season=S).to_csv(OUT / f'reliability_{S}.csv')

    # ---- 3. ceiling: fit and evaluate on the same season -----------------
    print('\n### 3. CEILING — 같은 시즌에서 적합 (배포 불가, 상한만)')
    print('    monotone 변환은 Resolution 을 바꾸지 못하므로 상한 = 현재 Reliability')
    for S, (y, p, _) in data.items():
        res, rel, bss = decomp(y, p)
        print(f'\n  --- {S}   이론 상한 dBSS = +{rel:.1f} (Reliability 를 0 으로) ---')
        for name, fit in FITTERS.items():
            _, f = fit(y, p)
            r = row(f'{name} (in-sample)', y, f(p), p)
            show(r); rows.append(dict(section='ceiling', season=S, **r))

    # ---- 4. temporal: fit on 2023, apply to 2024 ------------------------
    print('\n### 4. TEMPORAL — 2023 에서 적합 → 2024 에 적용 (배포 가능)')
    y23, p23, _ = data[2023]
    y24, p24, _ = data[2024]
    res, rel, bss = decomp(y24, p24)
    print(f'  2024 기준선  BSS {bss:.1f}  Res {res:.1f}  Rel {rel:.1f}   '
          f'(상한 +{rel:.1f})')
    for name, fit in FITTERS.items():
        par, f = fit(y23, p23)
        r = row(f'{name} (2023→2024)', y24, f(p24), p24)
        show(r); rows.append(dict(section='temporal', season=2024,
                                  params=str(par), **r))

    print('\n  역방향 확인 (2024 에서 적합 → 2023 에 적용)')
    for name, fit in FITTERS.items():
        par, f = fit(y24, p24)
        r = row(f'{name} (2024→2023)', y23, f(p23), p23)
        show(r); rows.append(dict(section='temporal_rev', season=2023,
                                  params=str(par), **r))

    out = OUT / 'calibration_results.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'\n  wrote {out}')
    print(f'  wrote {OUT}/reliability_2023.csv, reliability_2024.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
