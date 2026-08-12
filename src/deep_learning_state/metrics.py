"""Scoring, identical to what the resolution study used, so numbers compare.

Competition score:  100000 * max(0, 1 - Brier / (r(1-r))),  r = mean(y).

Murphy:  Brier = Uncertainty - Resolution + Reliability, with prediction bins
taken as 50 quantiles of p. Reported in BSS points (divided by r(1-r), x1e5)
so that BSS = Resolution - Reliability holds on the printed numbers.
"""
import numpy as np
import pandas as pd


def score(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    r = y.mean()
    return 100000.0 * max(0.0, 1.0 - np.mean((p - y) ** 2) / (r * (1 - r)))


def decomp(y, p, nb=50):
    y = np.asarray(y, float); p = np.asarray(p, float)
    r = y.mean(); unc = r * (1 - r)
    q = pd.qcut(p, nb, labels=False, duplicates='drop')
    g = pd.DataFrame({'p': p, 'y': y, 'q': q}).groupby('q').agg(
        n=('y', 'size'), pb=('p', 'mean'), yb=('y', 'mean'))
    w = g.n / len(y)
    return (float((w * (g.yb - r) ** 2).sum()) * 100000 / unc,
            float((w * (g.pb - g.yb) ** 2).sum()) * 100000 / unc,
            score(y, p))


def logloss(y, p):
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    o = np.argsort(p, kind='mergesort')
    ys = y[o]
    ranks = np.empty(len(p), float)
    ps = p[o]
    i = 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    n1 = ys.sum(); n0 = len(ys) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    return float((ranks[ys == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def paired(y, pa, pb, B=200, seed=0):
    """SE of score(pb) - score(pa) on shared bootstrap resamples."""
    y = np.asarray(y, float)
    pa = np.asarray(pa, float); pb = np.asarray(pb, float)
    rng = np.random.default_rng(seed)
    n = len(y); out = np.empty(B)
    for i in range(B):
        ix = rng.integers(0, n, n)
        out[i] = score(y[ix], pb[ix]) - score(y[ix], pa[ix])
    return float(out.mean()), float(out.std())


def calib_table(y, p, nb=10):
    q = pd.qcut(p, nb, labels=False, duplicates='drop')
    return pd.DataFrame({'y': y, 'p': p, 'q': q}).groupby('q').agg(
        n=('y', 'size'), pred=('p', 'mean'), actual=('y', 'mean')).assign(
        gap=lambda t: t.actual - t.pred)


def report(y, p, base=None, name='model'):
    res, rel, bss = decomp(y, p)
    d = dict(name=name, BSS=bss, Resolution=res, Reliability=rel,
             LogLoss=logloss(y, p), AUC=auc(y, p), pred_std=float(np.std(p)),
             pred_mean=float(np.mean(p)))
    if base is not None:
        bres, brel, bbss = decomp(y, base)
        m, se = paired(y, base, p)
        d.update(corr_with_v9=float(np.corrcoef(p, base)[0, 1]),
                 dBSS=bss - bbss, dResolution=res - bres,
                 dReliability=rel - brel, paired=m, paired_se=se,
                 sigma=(abs(m) / se if se > 0 else np.nan))
    return d


def verdict(d24, d23=None):
    """The adoption rule stated for this study, applied mechanically."""
    r24 = d24.get('dResolution', float('nan'))
    if not np.isfinite(r24) or r24 <= 0:
        v = 'FAIL'
    elif r24 < 10:
        v = 'WEAK'
    elif r24 < 20:
        v = 'INTERESTING'
    else:
        v = 'STRONG'
    if v == 'STRONG' and d23 is not None and d23.get('dResolution', -1) > 0:
        v = 'SUBMISSION CANDIDATE'
    return v
