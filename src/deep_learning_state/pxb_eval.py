"""M1 / M2 evaluation of the pitcher x batter interaction.

M1  round 16's method, reproduced: p = clip(p_v9 + offset). Deterministic, so
    uncertainty comes from a paired bootstrap rather than seeds.
M2  the offset and its confidence columns as FEATURES into LightGBM, A (v9
    only) vs B (v9 + pxb), identical folds and params. B - A is the profile's
    contribution; measuring B against p_v9 would credit it with the model's
    own fitting advantage, which is the error G0 was designed around.

SELECTION DISCIPLINE
--------------------
Hyperparameters (tau, K1, K2, K3) are chosen on the 2023 fold and then FROZEN.
The 2024 fold is opened once, at the end, with those values. Round 16 chose K
by best BSS on the evaluation season, which is why its +2.29 is optimistic;
this is the correction.

For a training row of season s, the table must come from seasons < s, so M2
builds one table per training season rather than reusing the evaluation
table. Otherwise a 2023 training row would be scored with a table that
contains 2023.

    python -m src.deep_learning_state.pxb_eval --stage select
    python -m src.deep_learning_state.pxb_eval --stage final --tau ... --k1 ...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths, pxb_table
from .dataset import feature_cols
from .metrics import decomp, paired, report

TAUS = [None, 3.0, 1.5, 0.8]
K1S = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
K23 = [(300.0, 300.0)]          # L2/L3 shrinkage; L1 is what the sweep is about


def _valid(fold):
    f = paths.DATA / 'folds' / f'fold_{fold}_va.parquet'
    cols = list(dict.fromkeys(
        ['row_id', 'season', 'pitcher_id', 'batter_id', 'batter_hand', 'init',
         'p_v9', 'control_success'] + list(feature_cols())))
    d = pd.read_parquet(f, columns=cols)
    if 'p_v9' not in d:
        raise SystemExit(f'fold_{fold}_va has no p_v9')
    return d


def _m1(va, tabs, K1, K2, K3, variant='l1only'):
    """`variant` is the ablation axis the design calls for.

    l1only  the pair cell alone, unseen pairs get 0. This is round 16's method
            and the only one that isolates the pitcher x batter signal.
    hier    fall back to P x Bhand then P. Diagnosed as harmful: with K1=1000
            the L1 weight averages 0.006, so 99% of the offset is the
            pitcher-level term, and d2_forward priced that at -40.9 / -58.2.
            Kept as a measured ablation rather than dropped silently.
    """
    off = pxb_table.apply(va, tabs, K1, K2, K3)
    col = 'pxb_offset_l1only' if variant == 'l1only' else 'pxb_offset'
    y = va.control_success.to_numpy(np.float64)
    p0 = va.p_v9.to_numpy(np.float64)
    p1 = np.clip(p0 + off[col].to_numpy(np.float64), 0.001, 0.999)
    res, rel, bss = decomp(y, p1)
    b_res, b_rel, b_bss = decomp(y, p0)
    m, se = paired(y, p0, p1)
    return dict(BSS=bss, Resolution=res, Reliability=rel,
                dBSS=bss - b_bss, dResolution=res - b_res,
                paired=m, paired_se=se,
                sigma=abs(m) / se if se > 0 else 0.0), p1, off


def _lgb():
    try:
        import lightgbm as lgb
    except (ImportError, OSError) as e:
        raise SystemExit(f'LightGBM unavailable: {e}')
    return lgb


def _m2(fold, va, target, tau, K1, K2, K3, seeds, train_seasons, a):
    """Train on earlier seasons (each with its own past-only table), predict
    the evaluation season."""
    lgb = _lgb()
    v9 = list(feature_cols())
    pf = pxb_table.PXB_FEATS

    tr_parts = []
    src = paths.DATA / 'folds' / f'fold_{fold}_tr.parquet'
    cols = list(dict.fromkeys(
        ['row_id', 'season', 'pitcher_id', 'batter_id', 'batter_hand', 'init',
         'control_success'] + v9))
    full = pd.read_parquet(src, columns=cols)
    for s in train_seasons:
        rows = full[full.season == s]
        if not len(rows):
            continue
        t_s, meta_s = pxb_table.build(fold, s - 1, target, tau)
        feats = pxb_table.apply(rows, t_s, K1, K2, K3)
        tr_parts.append(pd.concat([rows.reset_index(drop=True),
                                   feats.reset_index(drop=True)], axis=1))
        print(f'    train season {s}: {len(rows):,} rows, table from '
              f'{meta_s["seasons"]}')
    if not tr_parts:
        raise SystemExit('no training rows')
    tr = pd.concat(tr_parts, ignore_index=True)

    tabs, _ = pxb_table.build(fold, fold - 1, target, tau)
    va_feats = pxb_table.apply(va, tabs, K1, K2, K3)
    vaf = pd.concat([va.reset_index(drop=True),
                     va_feats.reset_index(drop=True)], axis=1)

    y = vaf.control_success.to_numpy(np.float64)
    p9 = vaf.p_v9.to_numpy(np.float64)
    out = {}
    for tag, feats in (('A', v9), ('B', v9 + pf)):
        preds = []
        for sd in seeds:
            params = dict(objective='binary', metric='binary_logloss',
                          learning_rate=a.learning_rate, num_leaves=a.num_leaves,
                          min_data_in_leaf=a.min_data_in_leaf,
                          feature_fraction=0.7, bagging_fraction=0.8,
                          bagging_freq=1, lambda_l2=10.0, verbosity=-1,
                          seed=sd, num_threads=0)
            dtr = lgb.Dataset(tr[feats].to_numpy(np.float32),
                              label=tr.control_success.to_numpy(np.float64),
                              init_score=tr.init.to_numpy(np.float64))
            dva = lgb.Dataset(vaf[feats].to_numpy(np.float32), label=y,
                              init_score=vaf.init.to_numpy(np.float64),
                              reference=dtr)
            bst = lgb.train(params, dtr, num_boost_round=a.rounds,
                            valid_sets=[dva],
                            callbacks=[lgb.early_stopping(a.early_stopping,
                                                          verbose=False)])
            z = vaf.init.to_numpy(np.float64) + bst.predict(
                vaf[feats].to_numpy(np.float32),
                num_iteration=bst.best_iteration, raw_score=True)
            preds.append(1.0 / (1.0 + np.exp(-z)))
            if tag == 'B' and sd == seeds[0]:
                imp = bst.feature_importance('gain')
                tot = imp.sum()
                out['pxb_gain_share'] = float(
                    sum(imp[feats.index(c)] for c in pf) / tot * 100)
        out[tag] = preds
    return out, y, p9, vaf


def _summary(tag, y, preds, p9):
    rs = [report(y, p, p9, name=tag) for p in preds]
    g = lambda k: np.array([r[k] for r in rs], float)
    return dict(BSS=g('BSS').mean(), Resolution=g('Resolution').mean(),
                dResolution=g('dResolution').mean(),
                dBSS=g('dBSS').mean(),
                dRes_sd=g('dResolution').std(ddof=1) if len(rs) > 1 else 0.0,
                dBSS_sd=g('dBSS').std(ddof=1) if len(rs) > 1 else 0.0)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='pxb_eval')
    ap.add_argument('--stage', choices=['select', 'final'], required=True)
    ap.add_argument('--target', choices=['resid', 'rate'], default='resid')
    ap.add_argument('--tau', default='')
    ap.add_argument('--k1', type=float, default=100.0)
    ap.add_argument('--k2', type=float, default=300.0)
    ap.add_argument('--k3', type=float, default=300.0)
    ap.add_argument('--seeds', default='42,43,44')
    ap.add_argument('--num-leaves', type=int, default=63)
    ap.add_argument('--learning-rate', type=float, default=0.05)
    ap.add_argument('--min-data-in-leaf', type=int, default=200)
    ap.add_argument('--rounds', type=int, default=2000)
    ap.add_argument('--early-stopping', type=int, default=100)
    ap.add_argument('--variant', choices=['l1only','hier'], default='l1only')
    ap.add_argument('--skip-m2', action='store_true')
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(',')]

    if a.stage == 'select':
        fold = 2023
        print('=' * 96)
        print(f'  STAGE 1 SELECT   fold {fold}  target={a.target}   '
              '(2024 는 열지 않음)')
        print('=' * 96)
        va = _valid(fold)
        b_res, b_rel, b_bss = decomp(va.control_success.to_numpy(np.float64),
                                     va.p_v9.to_numpy(np.float64))
        print(f'  v9 base   BSS {b_bss:.1f}  Res {b_res:.1f}  Rel {b_rel:.1f}'
              f'   rows {len(va):,}')
        rows = []
        print(f'\n  {"variant":>8} {"tau":>6} {"K1":>7} | {"M1 dRes":>9} '
              f'{"M1 dBSS":>9} {"paired":>16} {"sigma":>6}')
        for tau in TAUS:
            tabs, meta = pxb_table.build(fold, fold - 1, a.target, tau)
            n_season = len(meta['seasons'])
            for variant in ('l1only', 'hier'):
                for K1 in K1S:
                    for K2, K3 in K23:
                        r, _, _ = _m1(va, tabs, K1, K2, K3, variant)
                        rows.append(dict(variant=variant, tau=tau, K1=K1,
                                         K2=K2, K3=K3, n_season=n_season, **r))
                        print(f'  {variant:>8} {str(tau):>6} {K1:>7.0f} | '
                              f'{r["dResolution"]:>+9.1f} {r["dBSS"]:>+9.1f} '
                              f'{r["paired"]:>+9.1f}±{r["paired_se"]:<5.1f} '
                              f'{r["sigma"]:>6.2f}')
            if n_season < 2 and tau is TAUS[0]:
                print(f'  (소스 시즌 {meta["seasons"]} 1개 -> tau 는 무효, '
                      f'행이 반복됩니다)')
        df = pd.DataFrame(rows)
        best = df.sort_values('dBSS', ascending=False).iloc[0]
        print(f'\n  >>> 선택: variant={best.variant}  tau={best.tau}  K1={best.K1:.0f}  '
              f'K2={best.K2:.0f}  K3={best.K3:.0f}   '
              f'(2023 dBSS {best.dBSS:+.1f}, dRes {best.dResolution:+.1f})')
        if best.dBSS <= 0:
            print('  *** 2023 에서 최적 설정도 dBSS <= 0 -> 설계상 여기서 중단. '
                  '2024 를 열지 않습니다. ***')
        out = paths.EXP / f'pxb_select_S{fold}_{a.target}.csv'
        df.to_csv(out, index=False)
        (paths.EXP / f'pxb_best_{a.target}.json').write_text(json.dumps(
            dict(variant=str(best.variant),
                 tau=best.tau if pd.notna(best.tau) else None,
                 K1=float(best.K1), K2=float(best.K2), K3=float(best.K3),
                 sel_dBSS=float(best.dBSS), sel_dRes=float(best.dResolution)),
            default=str))
        print(f'  wrote {out}')
        return 0

    # ---------------- final ----------------
    fold = 2024
    tau = None if a.tau in ('', 'None', 'none') else float(a.tau)
    print('=' * 96)
    print(f'  STAGE 2 FINAL   fold {fold}  target={a.target}   '
          f'tau={tau} K1={a.k1:.0f} K2={a.k2:.0f} K3={a.k3:.0f}  (2023 에서 고정)')
    print('=' * 96)
    va = _valid(fold)
    y = va.control_success.to_numpy(np.float64)
    p9 = va.p_v9.to_numpy(np.float64)
    b_res, b_rel, b_bss = decomp(y, p9)
    print(f'  v9 base   BSS {b_bss:.1f}  Res {b_res:.1f}  Rel {b_rel:.1f}'
          f'   rows {len(va):,}')

    tabs, meta = pxb_table.build(fold, fold - 1, a.target, tau)
    print(f'  table: seasons {meta["seasons"]}  rows {meta["rows"]:,}  '
          f'L1 cells {len(tabs["L1"]):,}')
    r1, p1, off = _m1(va, tabs, a.k1, a.k2, a.k3, a.variant)
    seen = off.pxb_seen.to_numpy() > 0.5
    print(f'  pxb_seen=1 {seen.mean()*100:.1f}%   =0 {(~seen).mean()*100:.1f}%')
    print(f'\n  M1 (offset)   dRes {r1["dResolution"]:+8.1f}  '
          f'dBSS {r1["dBSS"]:+8.1f}  paired {r1["paired"]:+.1f}±{r1["paired_se"]:.1f} '
          f'{r1["sigma"]:.2f}s')
    for lab, msk in (('  seen=1', seen), ('  seen=0', ~seen)):
        rr, _, _ = None, None, None
        rs, rl, bs = decomp(y[msk], p1[msk])
        b2 = decomp(y[msk], p9[msk])
        print(f'  {lab}      dRes {rs-b2[0]:+8.1f}  dBSS {bs-b2[2]:+8.1f}  '
              f'n={msk.sum():,}')

    rows = [dict(model='M1', **r1)]
    if not a.skip_m2:
        print(f'\n  M2 (feature into LightGBM), seeds {seeds}')
        res, y2, p92, vaf = _m2(fold, va, a.target, tau, a.k1, a.k2, a.k3,
                                seeds, [fold - 1], a)
        sA = _summary('A', y2, res['A'], p92)
        sB = _summary('B', y2, res['B'], p92)
        print(f'    A  v9 only        dRes {sA["dResolution"]:+8.1f} ± {sA["dRes_sd"]:4.1f}   '
              f'dBSS {sA["dBSS"]:+8.1f} ± {sA["dBSS_sd"]:4.1f}')
        print(f'    B  v9 + pxb       dRes {sB["dResolution"]:+8.1f} ± {sB["dRes_sd"]:4.1f}   '
              f'dBSS {sB["dBSS"]:+8.1f} ± {sB["dBSS_sd"]:4.1f}')
        dres = sB['dResolution'] - sA['dResolution']
        dbss = sB['dBSS'] - sA['dBSS']
        ms, ses = [], []
        for pa, pb in zip(res['A'], res['B']):
            m, se = paired(y2, pa, pb)
            ms.append(m); ses.append(se)
        m, se = float(np.mean(ms)), float(np.mean(ses))
        sig = abs(m) / se if se > 0 else 0.0
        print(f'    >>> B - A         dRes {dres:+8.1f}   dBSS {dbss:+8.1f}   '
              f'paired {m:+.1f}±{se:.1f}  {sig:.2f}s')
        print(f'    pxb gain 점유율   {res.get("pxb_gain_share", float("nan")):.2f}%')
        rows.append(dict(model='M2_A', **sA))
        rows.append(dict(model='M2_B', **sB))
        rows.append(dict(model='M2_B_minus_A', dResolution=dres, dBSS=dbss,
                         paired=m, paired_se=se, sigma=sig))
        gate_res, gate_bss, gate_sig = dres, dbss, sig
    else:
        gate_res, gate_bss, gate_sig = (r1['dResolution'], r1['dBSS'],
                                        r1['sigma'])

    print('\n' + '=' * 96)
    ok = (gate_bss > 0) and (gate_res >= 10.0) and (gate_sig >= 2.0)
    print(f'  GATE:  dBSS > 0  AND  dRes >= +10  AND  paired >= 2 sigma')
    print(f'         dBSS {gate_bss:+.1f}   dRes {gate_res:+.1f}   '
          f'{gate_sig:.2f} sigma   ->  {"PASS" if ok else "FAIL"}')
    if not ok:
        print('  FAIL -> 추가 실험 금지. v9 가 최종 제출본으로 유지됩니다.')
    print('=' * 96)
    out = paths.EXP / f'pxb_final_S{fold}_{a.target}.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'  wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
