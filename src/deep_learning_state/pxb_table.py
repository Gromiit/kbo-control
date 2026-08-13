"""pitcher x batter interaction tables. Frozen lookups, built from past seasons.

WHY THIS EXISTS
---------------
Round 16 swept 46 conditioning variables and pitcher x batter was the only one
positive in both forward tests (2023 +0.47 +/- 0.95, 2024 +2.29 +/- 1.12). Its
implementation in d2_forward.py left four things undone, and this module is
those four:

  hierarchy   unseen pairs got offset exactly 0. 48.8% of 2024 rows are pairs
              never seen before, so half the evaluation set was untouched.
  year decay  five seasons weighted equally, although a 2023 pair recurs in
              2024 at 33.3% and a 2019 pair at 12.1% -- 2.75x apart.
  confidence  the cell count n entered the shrinkage and nothing else. The
              model never learned that half its rows carry no information.
  honest K    K was chosen by best BSS on the evaluation season, so +2.29 is
              optimistic. Selection happens on an earlier fold here.

RULES
-----
Tables are fitted on seasons STRICTLY BEFORE the target season and then
frozen; a row reads its own (pitcher, batter) and nothing else. test.csv is
never opened. Keys are pitcher_id / batter_id / batter_hand, all present in
the 48 official test columns.

TWO TARGETS, BECAUSE p_v9 IS NOT AVAILABLE EVERYWHERE
-----------------------------------------------------
  resid  y - p_v9, the quantity round 16 used and the one aimed at v9's blind
         spot. The OOF only exists for 2022-2024, so a fold-2024 table can use
         2022+2023 and covers 44.8% of rows.
  rate   y - (that pitcher's own weighted mean y). No p_v9 needed, so all of
         2019-2023 is usable and coverage rises to 51.2%. A different quantity:
         the pair effect relative to the pitcher's marginal, part of which v9
         already holds.

Neither dominates. resid is aimed correctly but data-starved; rate has more
data but overlaps v9. Both are built and reported.
"""
import numpy as np
import pandas as pd

from . import paths

L1 = ['pitcher_id', 'batter_id']
L2 = ['pitcher_id', 'batter_hand']
L3 = ['pitcher_id']

NEED = ['row_id', 'season', 'pitcher_id', 'batter_id', 'batter_hand',
        'control_success']

# p_v9 OOF exists for these seasons only; a `resid` table cannot reach further
RESID_SEASONS = (2022, 2023, 2024)


def _oof(season):
    p = paths.WORK / 'research' / f'oof_{season}.parquet'
    if not p.exists():
        return None
    return pd.read_parquet(p, columns=['row_id', 'p_v9'])


def load_source(fold, max_season, target):
    """Rows usable for table construction: seasons <= max_season, from the
    fold's own training parquet so the featurisation matches."""
    f = paths.DATA / 'folds' / f'fold_{fold}_tr.parquet'
    if not f.exists():
        raise SystemExit(f'need {f}')
    d = pd.read_parquet(f, columns=NEED)
    d = d[d.season <= max_season]
    if target == 'resid':
        d = d[d.season.isin(RESID_SEASONS)]
        if not len(d):
            raise SystemExit(
                f'target=resid needs p_v9, which exists only for '
                f'{RESID_SEASONS}; none are <= {max_season}')
        oof = pd.concat([_oof(s) for s in sorted(d.season.unique())
                         if _oof(s) is not None], ignore_index=True)
        n0 = len(d)
        d = d.merge(oof, on='row_id', how='inner')
        if len(d) != n0:
            raise SystemExit(f'p_v9 join lost rows: {n0:,} -> {len(d):,}')
    if len(d) and int(d.season.max()) > max_season:
        raise SystemExit('season gate failed')
    return d


def _residual(d, target):
    y = d.control_success.to_numpy(np.float64)
    if target == 'resid':
        r = y - d.p_v9.to_numpy(np.float64)
    else:
        # deviation from the pitcher's own mean, so the cell carries the
        # interaction rather than the pitcher level v9 already knows
        r = y - d.groupby('pitcher_id')['control_success'].transform('mean').to_numpy()
    return r - r.mean()          # season-level mean does not transfer


def _agg(d, r, w, keys):
    t = pd.DataFrame({'w': w, 'wr': w * r})
    for k in keys:
        t[k] = d[k].to_numpy()
    g = t.groupby(keys, sort=False).agg(n_eff=('w', 'sum'), sw=('wr', 'sum'),
                                        n_raw=('w', 'size'))
    g['m'] = g.sw / g.n_eff.replace(0, np.nan)
    return g[['n_eff', 'n_raw', 'm']].reset_index()


def build(fold, max_season, target='resid', tau=None):
    """Three nested tables. `tau` None means no decay (round 16's behaviour)."""
    d = load_source(fold, max_season, target)
    r = _residual(d, target)
    age = (max_season - d.season.to_numpy()).astype(np.float64)
    w = np.ones(len(d)) if tau is None else np.exp(-age / float(tau))
    tabs = {'L1': _agg(d, r, w, L1), 'L2': _agg(d, r, w, L2),
            'L3': _agg(d, r, w, L3)}
    meta = dict(fold=fold, max_season=max_season, target=target, tau=tau,
                rows=len(d), seasons=sorted(int(s) for s in d.season.unique()))
    return tabs, meta


def apply(frame, tabs, K1=100.0, K2=300.0, K3=300.0):
    """Per-row features. Hierarchical: L1 where thick, else L2, else L3.

    alpha_k = n / (n + K) is the usual empirical-Bayes weight, and the levels
    compose so a pair with no history falls back rather than being handed a
    zero it cannot be distinguished from a genuine zero effect.
    """
    f = frame[['pitcher_id', 'batter_id', 'batter_hand']].copy()
    out = {}
    for lvl, keys, K in (('L1', L1, K1), ('L2', L2, K2), ('L3', L3, K3)):
        t = tabs[lvl].rename(columns={'m': f'm_{lvl}', 'n_eff': f'ne_{lvl}',
                                      'n_raw': f'nr_{lvl}'})
        f = f.merge(t, on=keys, how='left')
        ne = f[f'ne_{lvl}'].fillna(0.0).to_numpy(np.float64)
        m = f[f'm_{lvl}'].fillna(0.0).to_numpy(np.float64)
        out[f'a_{lvl}'] = ne / (ne + K)
        out[f'm_{lvl}'] = m
        out[f'ne_{lvl}'] = ne
        out[f'nr_{lvl}'] = f[f'nr_{lvl}'].fillna(0.0).to_numpy(np.float64)

    a1, a2 = out['a_L1'], out['a_L2']
    lower = a2 * out['m_L2'] + (1 - a2) * out['a_L3'] * out['m_L3']
    offset = a1 * out['m_L1'] + (1 - a1) * lower

    return pd.DataFrame({
        'pxb_offset': offset,
        'pxb_offset_l1only': a1 * out['m_L1'],   # ablation: no fallback
        'pxb_n_raw': np.log1p(out['nr_L1']),
        'pxb_n_eff': np.log1p(out['ne_L1']),
        'pxb_alpha1': a1,
        'pxb_seen': (out['nr_L1'] > 0).astype(np.float64),
        'pxb_n_l2': np.log1p(out['ne_L2']),
    }, index=frame.index)


PXB_FEATS = ['pxb_offset', 'pxb_n_raw', 'pxb_n_eff', 'pxb_alpha1',
             'pxb_seen', 'pxb_n_l2']
