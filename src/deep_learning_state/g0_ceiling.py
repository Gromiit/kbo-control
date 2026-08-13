"""G0: does the historical pitcher profile carry information at all?

THE MEASUREMENT
---------------
Same-season ceiling on fold 2024, GroupKFold by pitcher. The downstream model
is fitted on 2024 rows themselves, so this is NOT a temporal result and NOT
deployable -- it is the most generous reading available. Phase C-2 measured
the attenuation from ceiling to temporal at roughly 5x (blend gain +10.2 ->
+1.9), so a ceiling that cannot clear the bar means the temporal number is
zero and no encoder is worth building.

The profile itself stays legal throughout: it is built from Trackman seasons
<= 2023 only (`pitcher_profile.py --max-season 2023`). Only the downstream
fit is same-season, and that is exactly what makes this a ceiling rather than
a result.

WHAT THE GATE SHOULD BE MEASURED AGAINST
----------------------------------------
Two deltas are reported and they answer different questions.

  B vs p_v9   the shipped baseline. Inflated here, because model A is already
              fitted on the evaluation season and beats v9 for that reason
              alone. Not a clean read on the profile.

  B vs A      identical folds, identical params, identical seed, the only
              difference being the profile columns. THIS is the profile's
              contribution and the number the gate belongs on.

Reporting only the first would credit the profile with the same-season
fitting advantage, which is the same class of error as anchoring a recovery
ratio at zero instead of at the floor.

Grouping by pitcher matters: rows of one pitcher must not sit on both sides
of a split, or the profile -- a per-pitcher constant -- becomes a lookup of
that pitcher's own held-out rows.

    python -m src.deep_learning_state.g0_ceiling \
        --profile data/profiles/pitcher_profile_le2023_plus.parquet
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .dataset import feature_cols
from .metrics import decomp, report

BANNED = ('control_success', 'p_v9', 'init', 'row_id', 'season', 'pitcher_id')


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _lgb():
    try:
        import lightgbm as lgb
    except (ImportError, OSError) as e:
        raise SystemExit(f'LightGBM unavailable: {e}\n'
                         '  brew install libomp && pip install lightgbm==4.7.0')
    return lgb


def _load(fold, profile_path):
    v9 = list(feature_cols())
    cols = list(dict.fromkeys(
        ['row_id', 'season', 'pitcher_id', 'init', 'p_v9', 'control_success'] + v9))
    fp = paths.DATA / 'folds' / f'fold_{fold}_va.parquet'
    if not fp.exists():
        raise SystemExit(f'need {fp} (Mac-only step)')
    d = pd.read_parquet(fp, columns=cols)
    n0 = len(d)

    pp = Path(profile_path)
    if not pp.exists():
        raise SystemExit(
            f'no profile at {pp}\n'
            '  python -m src.deep_learning_state.pitcher_profile '
            '--max-season 2023 --variant plus')
    prof = pd.read_parquet(pp)
    pf = [c for c in prof.columns if c != 'pitcher_id']
    bad = [c for c in pf if c in BANNED or 'success' in c or 'control' in c]
    if bad:
        raise SystemExit(f'profile contains forbidden columns: {bad}')

    d = d.merge(prof, on='pitcher_id', how='left')
    if len(d) != n0:
        raise SystemExit(f'{n0:,} rows -> {len(d):,} after profile join')
    # pitchers with no prior-season Trackman get zeros and the flag off, not a
    # learned "unknown" vector -- round 17 saw unknown-entity handling inflate
    # temporal overfit.
    d['tm_has_history'] = d['tm_has_history'].fillna(0.0)
    d[pf] = d[pf].fillna(0.0)
    return d, v9, pf


def _fit_oof(lgb, d, feats, groups, a, tag):
    """GroupKFold OOF predictions. init enters as init_score, as v9 does."""
    from sklearn.model_selection import GroupKFold
    X = d[feats].to_numpy(np.float32)
    y = d.control_success.to_numpy(np.float64)
    init = d.init.to_numpy(np.float64)
    oof = np.zeros(len(d), dtype=np.float64)
    imp = np.zeros(len(feats), dtype=np.float64)
    params = dict(objective='binary', metric='binary_logloss',
                  learning_rate=a.learning_rate, num_leaves=a.num_leaves,
                  min_data_in_leaf=a.min_data_in_leaf, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                  verbosity=-1, seed=a.seed, num_threads=0)
    for k, (i_tr, i_va) in enumerate(
            GroupKFold(n_splits=a.folds).split(d, groups=groups), 1):
        if set(groups[i_tr]) & set(groups[i_va]):
            raise SystemExit('pitcher appears on both sides of a fold')
        dtr = lgb.Dataset(X[i_tr], label=y[i_tr], init_score=init[i_tr])
        dva = lgb.Dataset(X[i_va], label=y[i_va], init_score=init[i_va],
                          reference=dtr)
        bst = lgb.train(params, dtr, num_boost_round=a.rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(a.early_stopping,
                                                      verbose=False)])
        oof[i_va] = init[i_va] + bst.predict(X[i_va],
                                             num_iteration=bst.best_iteration,
                                             raw_score=True)
        imp += bst.feature_importance('gain')
        print(f'    {tag} fold {k}/{a.folds}: train {len(i_tr):,} '
              f'valid {len(i_va):,}  best_iter {bst.best_iteration}')
    return _sigmoid(oof), imp


def _line(name, y, p, base):
    r = report(y, p, base, name=name)
    sig = abs(r['paired']) / r['paired_se'] if r['paired_se'] > 0 else 0.0
    print(f'  {name:<26s} BSS {r["BSS"]:8.1f}  Res {r["Resolution"]:8.1f}  '
          f'Rel {r["Reliability"]:7.1f}  dRes {r["dResolution"]:+8.1f}  '
          f'dBSS {r["dBSS"]:+8.1f}  paired {r["paired"]:+7.1f}±{r["paired_se"]:4.1f} '
          f'{sig:4.1f}s')
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(prog='g0_ceiling')
    ap.add_argument('--fold', type=int, default=2024)
    ap.add_argument('--profile',
                    default='data/profiles/pitcher_profile_le2023_plus.parquet')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--num-leaves', type=int, default=63)
    ap.add_argument('--learning-rate', type=float, default=0.05)
    ap.add_argument('--min-data-in-leaf', type=int, default=200)
    ap.add_argument('--rounds', type=int, default=2000)
    ap.add_argument('--early-stopping', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='')
    a = ap.parse_args(argv)

    lgb = _lgb()
    d, v9, pf = _load(a.fold, a.profile)
    y = d.control_success.to_numpy(np.float64)
    p9 = d.p_v9.to_numpy(np.float64)
    groups = d.pitcher_id.to_numpy()
    hh = d.tm_has_history.to_numpy() > 0.5

    print('=' * 100)
    print(f'  G0 CEILING   fold {a.fold}   profile {Path(a.profile).name}')
    print(f'  rows {len(d):,}   pitchers {d.pitcher_id.nunique()}   '
          f'GroupKFold {a.folds}')
    print(f'  v9 feats {len(v9)}   profile feats {len(pf)}')
    print(f'  has_history=1  {hh.mean()*100:.1f}%   =0  {(~hh).mean()*100:.1f}%')
    print('  ** same-season fit: NOT temporal, NOT deployable -- ceiling only **')
    print('=' * 100)

    print('  A: v9 features only')
    pA, impA = _fit_oof(lgb, d, v9, groups, a, 'A')
    print('  B: v9 + profile')
    pB, impB = _fit_oof(lgb, d, v9 + pf, groups, a, 'B')

    br, brel, bbss = decomp(y, p9)
    print(f'\n=== 전체 {len(d):,} 행 ===')
    print(f'  {"v9 (baseline)":<26s} BSS {bbss:8.1f}  Res {br:8.1f}  Rel {brel:7.1f}')
    rA = _line('A  v9 only', y, pA, p9)
    rB = _line('B  v9 + profile', y, pB, p9)

    # the gate number: identical folds/params/seed, profile the only difference
    dres_BA = rB['Resolution'] - rA['Resolution']
    dbss_BA = rB['BSS'] - rA['BSS']
    from .metrics import paired as paired_boot
    m, se = paired_boot(y, pA, pB)
    print(f'\n  >>> B - A (프로필 순기여)   dRes {dres_BA:+8.1f}   '
          f'dBSS {dbss_BA:+8.1f}   paired {m:+.1f} ± {se:.1f}  '
          f'{abs(m)/se if se>0 else 0:.1f}s')

    print(f'\n=== has_history=1 부분집합  {hh.sum():,} 행 ===')
    bh = decomp(y[hh], p9[hh])
    print(f'  {"v9 (baseline)":<26s} BSS {bh[2]:8.1f}  Res {bh[0]:8.1f}  Rel {bh[1]:7.1f}')
    rAh = _line('A  v9 only', y[hh], pA[hh], p9[hh])
    rBh = _line('B  v9 + profile', y[hh], pB[hh], p9[hh])
    print(f'  >>> B - A  dRes {rBh["Resolution"]-rAh["Resolution"]:+8.1f}   '
          f'dBSS {rBh["BSS"]-rAh["BSS"]:+8.1f}')

    print('\n=== feature importance (gain) — B 모델 상위 20 ===')
    order = np.argsort(-impB)
    names = v9 + pf
    tot = impB.sum()
    prof_gain = impB[[names.index(c) for c in pf]].sum()
    for i in order[:20]:
        mark = ' *PROFILE*' if names[i] in pf else ''
        print(f'  {impB[i]/tot*100:6.2f}%  {names[i]}{mark}')
    print(f'\n  프로필 {len(pf)}개 feature 의 gain 점유율: {prof_gain/tot*100:.2f}%')

    # ---- gate --------------------------------------------------------------
    print('\n' + '=' * 100)
    gate = (dres_BA >= 20.0) and (dbss_BA > 0.0)
    print(f'  GATE G0-b:  dRes(B-A) >= +20 AND dBSS(B-A) > 0')
    print(f'              dRes {dres_BA:+.1f}   dBSS {dbss_BA:+.1f}   '
          f'-> {"PASS" if gate else "FAIL"}')
    if not gate:
        print('  FAIL -> E1/E2 encoder 구현하지 않음. 주변분포가 담을 수 있는 최대치가')
        print('          same-season ceiling 에서도 기준 미달이므로 temporal 은 0 에 가깝습니다.')
    print('=' * 100)

    out = Path(a.out or (paths.EXP / f'g0_ceiling_S{a.fold}.csv'))
    rows = []
    for tag, r, sub in (('A_full', rA, 'all'), ('B_full', rB, 'all'),
                        ('A_hasneq0', rAh, 'has_history'),
                        ('B_hasneq0', rBh, 'has_history')):
        rows.append(dict(tag=tag, subset=sub, profile=Path(a.profile).name,
                         n=len(d) if sub == 'all' else int(hh.sum()),
                         **{k: r[k] for k in ('BSS', 'Resolution', 'Reliability',
                                              'dBSS', 'dResolution', 'paired',
                                              'paired_se')}))
    rows.append(dict(tag='B_minus_A', subset='all', profile=Path(a.profile).name,
                     n=len(d), dResolution=dres_BA, dBSS=dbss_BA,
                     paired=m, paired_se=se))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'  wrote {out}')
    return 0 if gate else 0      # measurement tool: gate is reported, not fatal


if __name__ == '__main__':
    sys.exit(main())
