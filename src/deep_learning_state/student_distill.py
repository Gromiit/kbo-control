"""Phase C-2: how much of the teacher's edge survives into a single-row student.

THE QUESTION
------------
Phase A measured that the pitcher's within-season sequence is worth
dResolution +102 over v9 on 2024 (3 seeds, +-8). That model cannot ship: it
reads the previous 32 pitches of the same season, and the 2025 test set is
five shuffled unlabelled rows. So the only route left is distillation --
teach a model that sees ONE ROW to imitate the teacher.

This measures whether that is even possible. If single-row features cannot
express the teacher's offset, Phase D is pointless and the honest conclusion
is "the within-season state is large but undeployable".

WHAT THE STUDENT IS ALLOWED TO SEE
----------------------------------
Exactly v9's own numeric feature list, imported from the round-17 builder
rather than copied, so it cannot drift. Every one of those is a function of
the row plus lookup tables frozen on earlier seasons -- which is the same
thing script.py computes for test.csv, so anything learned here is
deployable. No sequence channel, no p_v9, no label. Asserted, not assumed.

`init` (the level model's log-odds) is NOT a feature. It enters additively,
the way v9 gives it to LightGBM as init_score and the way the GRU used it:

    teacher_offset = logit(teacher_mean) - init      <- the target
    p_student      = sigmoid(init + student_offset)

So the student is asked for precisely the thing the sequence added, not for
the base rate it could copy from the level model.

TWO MEASUREMENTS, AND WHY BOTH
------------------------------
CEILING  GroupKFold by pitcher within the validation season. Same-season, so
         it makes NO temporal claim and is NOT deployable -- it answers only
         "is this information present in single-row features at all?" It is
         the cheap gate: if the ceiling is near zero, nothing downstream can
         work and no further runs are needed. Grouping by pitcher stops rows
         of one pitcher sitting on both sides of a split and inflating it.

TEMPORAL The real test, when --train-teacher is supplied: fit on an earlier
         season's teacher outputs, predict the later season. Train season
         must be strictly earlier, which is checked.

    python -m src.deep_learning_state.student_distill \
        --valid-teacher experiments/deep_learning_state/teacher_predictions_S2024.parquet
    # add --train-teacher .../teacher_predictions_S2023.parquet for TEMPORAL
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .dataset import feature_cols
from .metrics import decomp, report

BANNED = ('control_success', 'p_v9', 'init', 'row_id', 'season', 'teacher_mean')


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _lgb():
    try:
        import lightgbm as lgb
    except (ImportError, OSError) as e:
        raise SystemExit(
            f'LightGBM unavailable: {e}\n\n'
            'On macOS the wheel needs the OpenMP runtime:\n'
            '  brew install libomp && pip install lightgbm==4.7.0\n'
            '(4.7.0 is what the submission bundle pins.)')
    return lgb


def _load(teacher_parquet, feats):
    """Join teacher predictions to the v9 features of their own fold.

    The teacher parquet carries no features by design -- it is a prediction
    artefact. Features come from the fold parquet the predictions were made
    on, matched on row_id, so a mismatch surfaces here instead of silently
    training on misaligned rows.
    """
    tp = Path(teacher_parquet)
    if not tp.exists():
        raise SystemExit(
            f'no teacher predictions at {tp}\n'
            'Produce them first:\n'
            '  python -m src.deep_learning_state.teacher_predict --fold <S> '
            '--sequence-length 32')
    t = pd.read_parquet(tp)
    need = {'row_id', 'season', 'control_success', 'p_v9', 'teacher_mean'}
    missing = sorted(need - set(t.columns))
    if missing:
        raise SystemExit(f'{tp.name}: missing columns {missing}')
    seasons = sorted(int(s) for s in t.season.unique())
    if len(seasons) != 1:
        raise SystemExit(f'{teacher_parquet}: expected one season, got {seasons}')
    S = seasons[0]
    fold_p = paths.DATA / 'folds' / f'fold_{S}_va.parquet'
    if not fold_p.exists():
        raise SystemExit(
            f'need {fold_p} for the features.\n'
            'This step runs on the Mac -- the fold parquets are not uploaded.')
    cols = list(dict.fromkeys(['row_id', 'season', 'init', 'pitcher_id'] + feats))
    f = pd.read_parquet(fold_p, columns=cols)
    n0 = len(t)
    d = t.merge(f, on='row_id', how='inner', suffixes=('', '_fold'))
    if len(d) != n0:
        raise SystemExit(f'{teacher_parquet}: {n0:,} teacher rows -> {len(d):,} '
                         f'after joining features; row_id mismatch')
    if d.row_id.duplicated().any():
        raise SystemExit(f'{teacher_parquet}: duplicate row_id after join')
    if 'season_fold' in d and not (d.season == d.season_fold).all():
        raise SystemExit(f'{teacher_parquet}: season disagrees with the fold parquet')
    d['teacher_offset'] = _logit(d.teacher_mean) - d.init.to_numpy(np.float64)
    return d, S


def _fit_predict(lgb, tr, va, feats, a):
    """One LightGBM regression on the teacher offset. No label is ever seen."""
    dtr = lgb.Dataset(tr[feats].to_numpy(np.float32),
                      label=tr.teacher_offset.to_numpy(np.float64))
    dva = lgb.Dataset(va[feats].to_numpy(np.float32),
                      label=va.teacher_offset.to_numpy(np.float64), reference=dtr)
    params = dict(objective='regression', metric='rmse',
                  learning_rate=a.learning_rate, num_leaves=a.num_leaves,
                  min_data_in_leaf=a.min_data_in_leaf, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                  verbosity=-1, seed=a.seed, num_threads=0)
    booster = lgb.train(params, dtr, num_boost_round=a.rounds,
                        valid_sets=[dva],
                        callbacks=[lgb.early_stopping(a.early_stopping, verbose=False)])
    return booster.predict(va[feats].to_numpy(np.float32),
                           num_iteration=booster.best_iteration), booster


def _score(tag, d, s_off):
    """Everything the phase is judged on, for one set of student offsets."""
    y = d.control_success.to_numpy(np.float64)
    p9 = d.p_v9.to_numpy(np.float64)
    t_off = d.teacher_offset.to_numpy(np.float64)
    init = d.init.to_numpy(np.float64)
    p_stu = _sigmoid(init + s_off)
    p_tea = d.teacher_mean.to_numpy(np.float64)

    resid = s_off - t_off
    sst = float(((t_off - t_off.mean()) ** 2).sum())
    out = dict(
        tag=tag, n=len(d),
        corr_teacher=float(np.corrcoef(s_off, t_off)[0, 1]),
        logit_rmse=float(np.sqrt((resid ** 2).mean())),
        r2_offset=float(1.0 - (resid ** 2).sum() / sst) if sst > 0 else float('nan'),
        teacher_offset_sd=float(t_off.std()),
        student_offset_sd=float(s_off.std()),
    )
    st = report(y, p_stu, p9, name=tag)
    te = report(y, p_tea, p9, name='teacher')
    # A student that learns nothing predicts offset == 0, i.e. the level model
    # alone -- whose dResolution is about -320, NOT 0. Anchoring the ratio at
    # zero therefore scores a model that learned nothing at ~100%, which is how
    # this was wrong the first time. The floor is the real origin.
    fl = report(y, _sigmoid(init), p9, name='floor')
    out.update(BSS=st['BSS'], Resolution=st['Resolution'],
               Reliability=st['Reliability'],
               dBSS=st['dBSS'], dResolution=st['dResolution'],
               paired=st['paired'], paired_se=st['paired_se'],
               teacher_dResolution=te['dResolution'], teacher_dBSS=te['dBSS'],
               floor_dResolution=fl['dResolution'])
    span = te['dResolution'] - fl['dResolution']
    out['span'] = span
    # The ratio is only meaningful when the teacher actually beats the floor by
    # something worth splitting. The real fold-2024 teacher has span ~+421
    # (dRes +102 over a floor of -319); anything under 10 points is noise being
    # divided by noise, so say so instead of printing a number.
    out['recovery'] = ((out['dResolution'] - fl['dResolution']) / span
                       if span > 10.0 else float('nan'))
    return out, p_stu


def _print(r, base):
    print(f'\n=== {r["tag"]}  (n={r["n"]:,}) ===')
    print(f'  teacher 재현       corr {r["corr_teacher"]:.4f}   '
          f'logit RMSE {r["logit_rmse"]:.4f}   R2(offset) {r["r2_offset"]:+.4f}')
    print(f'  offset 표준편차     teacher {r["teacher_offset_sd"]:.4f}  '
          f'-> student {r["student_offset_sd"]:.4f}')
    print(f'  v9 (baseline)      BSS {base[2]:9.1f}   Res {base[0]:9.1f}   '
          f'Rel {base[1]:8.1f}')
    print(f'  student            BSS {r["BSS"]:9.1f}   Res {r["Resolution"]:9.1f}   '
          f'Rel {r["Reliability"]:8.1f}')
    print(f'  v9 대비            dRes {r["dResolution"]:+9.1f}   '
          f'dBSS {r["dBSS"]:+9.1f}   paired {r["paired"]:+.1f} ± {r["paired_se"]:.1f}')
    print(f'  복원율 = (student - floor) / (teacher - floor)')
    print(f'    floor (offset=0, level model only)  dRes {r["floor_dResolution"]:+9.1f}')
    print(f'    teacher                             dRes {r["teacher_dResolution"]:+9.1f}')
    pct = ('  n/a (teacher-floor span %.1f < 10, 비율이 무의미)' % r['span']
           if not np.isfinite(r['recovery']) else '%6.1f%%' % (r['recovery'] * 100))
    print(f'    student                             dRes {r["dResolution"]:+9.1f}'
          f'   -> 복원율 {pct}')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='student_distill')
    ap.add_argument('--valid-teacher',
                    default=str(paths.EXP / 'teacher_predictions_S2024.parquet'))
    ap.add_argument('--train-teacher', default='',
                    help='earlier season teacher parquet; enables TEMPORAL mode')
    ap.add_argument('--out', default='')
    ap.add_argument('--folds', type=int, default=5, help='CEILING GroupKFold')
    ap.add_argument('--skip-ceiling', action='store_true')
    ap.add_argument('--num-leaves', type=int, default=63)
    ap.add_argument('--learning-rate', type=float, default=0.05)
    ap.add_argument('--min-data-in-leaf', type=int, default=200)
    ap.add_argument('--rounds', type=int, default=2000)
    ap.add_argument('--early-stopping', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args(argv)

    feats = list(feature_cols())
    bad = [c for c in feats if c in BANNED or c.startswith('teacher_')]
    if bad:
        raise SystemExit(f'feature list contains forbidden columns: {bad}')

    lgb = _lgb()
    va, S_va = _load(a.valid_teacher, feats)
    bres, brel, bbss = decomp(va.control_success.to_numpy(np.float64),
                              va.p_v9.to_numpy(np.float64))

    print('=' * 74)
    print(f'  Phase C-2 student distillation   valid season {S_va}')
    print(f'  features   {len(feats)} (v9 numeric_feats, imported)  '
          f'-- sequence/label/p_v9 제외 확인됨')
    print(f'  target     logit(teacher_mean) - init')
    print(f'  rows       valid {len(va):,}')
    print('=' * 74)

    results, preds = [], {}

    # ---- CEILING: same-season, grouped by pitcher. Not deployable. ---------
    if not a.skip_ceiling:
        from sklearn.model_selection import GroupKFold
        oof = np.zeros(len(va), dtype=np.float64)
        gkf = GroupKFold(n_splits=a.folds)
        groups = va.pitcher_id.to_numpy()
        for k, (i_tr, i_va) in enumerate(gkf.split(va, groups=groups), 1):
            p, b = _fit_predict(lgb, va.iloc[i_tr], va.iloc[i_va], feats, a)
            oof[i_va] = p
            print(f'  ceiling fold {k}/{a.folds}: train {len(i_tr):,} '
                  f'valid {len(i_va):,}  best_iter {b.best_iteration}')
        r, p_stu = _score(f'CEILING (same-season {S_va}, GroupKFold/pitcher)', va, oof)
        _print(r, (bres, brel, bbss))
        print('  ** 배포 불가 · temporal 주장 아님 — "정보가 존재하는가"만 측정 **')
        results.append(r)
        preds['p_student_ceiling'] = p_stu
        preds['student_offset_ceiling'] = oof

    # ---- TEMPORAL: the real test -------------------------------------------
    if a.train_teacher:
        tr, S_tr = _load(a.train_teacher, feats)
        if S_tr >= S_va:
            raise SystemExit(f'train season {S_tr} must be < valid season {S_va}')
        ov = len(set(tr.row_id) & set(va.row_id))
        if ov:
            raise SystemExit(f'{ov} row_id shared between train and valid')
        print(f'\n  TEMPORAL: train season {S_tr} ({len(tr):,} rows) '
              f'-> valid season {S_va} ({len(va):,} rows), row_id 겹침 0')
        p, b = _fit_predict(lgb, tr, va, feats, a)
        print(f'  best_iter {b.best_iteration}')
        r, p_stu = _score(f'TEMPORAL ({S_tr} -> {S_va})', va, p)
        _print(r, (bres, brel, bbss))
        results.append(r)
        preds['p_student_temporal'] = p_stu
        preds['student_offset_temporal'] = p
    else:
        print('\n  TEMPORAL 생략 — --train-teacher 미지정 '
              '(이전 시즌 teacher parquet 이 필요합니다)')

    # ---- write -------------------------------------------------------------
    out_p = Path(a.out or (paths.EXP / f'student_predictions_S{S_va}.parquet'))
    df = va[['row_id', 'season', 'control_success', 'p_v9', 'teacher_mean',
             'init', 'teacher_offset']].copy()
    for k, v in preds.items():
        df[k] = np.asarray(v, dtype=np.float32)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_p, index=False)
    print(f'\n  wrote {out_p}   {len(df):,} rows x {len(df.columns)} cols')

    res_p = out_p.with_name(f'student_results_S{S_va}.csv')
    pd.DataFrame(results).to_csv(res_p, index=False)
    print(f'  wrote {res_p}')

    print('\n' + '-' * 74)
    for r in results:
        pct = ('n/a' if not np.isfinite(r['recovery'])
               else '%6.1f%%' % (r['recovery'] * 100))
        print(f'  {r["tag"]:<48s} dRes {r["dResolution"]:+8.1f}  '
              f'R2(offset) {r["r2_offset"]:+.4f}  복원율 {pct}')
    print('  판정 기준: TEMPORAL 복원율이 낮으면 (15차 Trackman student 는 '
          '+1.6/+5.3 수준)\n            Phase D 는 의미가 없고 '
          '"시즌 내 state 는 크지만 배포 불가" 로 종결하는 것이 맞습니다.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
