"""E0: XGBoost out-of-fold predictions on the existing walk-forward folds.

PURPOSE IS DIVERSITY, NOT ACCURACY
----------------------------------
v9 is already a 26-model ensemble -- A7 and A9 are each 4 variants x 3 seeds of
LightGBM, plus 2 CatBoost -- and on 2024 its three components already correlate
0.81 to 0.96 in logit space. A fourth model only helps if its errors differ, so
this script exists to produce predictions whose correlation with the incumbents
can be measured, not to build a better single model.

TWO ARMS
--------
  base    base_margin = init, exactly what v9's A families get. The fair
          comparison, and the one most likely to correlate highly.
  noinit  no base_margin. Its level will be off and its standalone BSS lower,
          but its errors are free to differ. Diversity is the thing being
          measured, so an arm that is deliberately not level-matched belongs
          in the design.

DISCIPLINE
----------
Training rows come from fold_S_tr (seasons < S) and predictions are made on
fold_S_va (season S) -- the same walk-forward split v9's own OOF used, so the
resulting columns are directly comparable to p_A7 / p_A9 / p_Bcat.

The competition test.csv is never opened. "test prediction" in this experiment
means the held-out fold, which is what fold_S_va is; predicting the real
test.csv would need v9's frozen tables, has no labels to score against, and is
explicitly out of scope.

    python experiments/ensemble/xgb/build_xgb_oof.py --folds 2023,2024
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from src.deep_learning_state.dataset import feature_cols  # noqa: E402
from src.deep_learning_state.metrics import decomp        # noqa: E402

OUT = Path(__file__).resolve().parent / 'oof'

PARAMS = dict(objective='binary:logistic', eval_metric='logloss',
              tree_method='hist', max_depth=6, eta=0.05, subsample=0.8,
              colsample_bytree=0.7, reg_lambda=10.0, min_child_weight=200,
              nthread=0, verbosity=0)


def _xgb():
    try:
        import xgboost as xgb
    except (ImportError, OSError) as e:
        raise SystemExit(f'xgboost unavailable: {e}\n'
                         '  brew install libomp && pip install xgboost')
    return xgb


def load_fold(fold, feats):
    root = REPO / 'data' / 'folds'
    cols = list(dict.fromkeys(
        ['row_id', 'season', 'init', 'control_success'] + feats))
    tr = pd.read_parquet(root / f'fold_{fold}_tr.parquet', columns=cols)
    va_cols = cols + ['p_v9']
    va = pd.read_parquet(root / f'fold_{fold}_va.parquet', columns=va_cols)
    if int(tr.season.max()) >= fold:
        raise SystemExit(f'train season {tr.season.max()} >= fold {fold}')
    if set(va.season.unique()) != {fold}:
        raise SystemExit(f'valid seasons {sorted(va.season.unique())} != {fold}')
    if len(set(tr.row_id) & set(va.row_id)):
        raise SystemExit('train/valid row_id overlap')
    return tr, va


def run(fold, arm, seed, feats, tr, va, rounds, early):
    xgb = _xgb()
    Xtr = tr[feats].to_numpy(np.float32)
    Xva = va[feats].to_numpy(np.float32)
    ytr = tr.control_success.to_numpy(np.float64)
    yva = va.control_success.to_numpy(np.float64)

    dtr = xgb.DMatrix(Xtr, label=ytr)
    dva = xgb.DMatrix(Xva, label=yva)
    if arm == 'base':
        # the A families receive init as init_score; base_margin is the same
        # thing for xgboost, so this arm is the like-for-like comparison
        dtr.set_base_margin(tr.init.to_numpy(np.float64))
        dva.set_base_margin(va.init.to_numpy(np.float64))

    p = dict(PARAMS, seed=seed)
    t0 = time.time()
    bst = xgb.train(p, dtr, num_boost_round=rounds,
                    evals=[(dva, 'va')], early_stopping_rounds=early,
                    verbose_eval=False)
    pred = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
    res, rel, bss = decomp(yva, np.clip(pred, 1e-6, 1 - 1e-6))
    print(f'    fold {fold} {arm:6s} seed {seed}: best_iter {bst.best_iteration:4d}  '
          f'BSS {bss:8.1f}  Res {res:8.1f}  Rel {rel:7.1f}  [{time.time()-t0:.0f}s]')
    return pred, dict(fold=fold, arm=arm, seed=seed, BSS=bss, Resolution=res,
                      Reliability=rel, best_iter=int(bst.best_iteration),
                      n_train=len(tr), n_valid=len(va))


def main(argv=None):
    ap = argparse.ArgumentParser(prog='build_xgb_oof')
    ap.add_argument('--folds', default='2023,2024')
    ap.add_argument('--arms', default='base,noinit')
    ap.add_argument('--seeds', default='42,43,44')
    ap.add_argument('--rounds', type=int, default=2000)
    ap.add_argument('--early-stopping', type=int, default=100)
    a = ap.parse_args(argv)

    feats = list(feature_cols())
    folds = [int(x) for x in a.folds.split(',')]
    arms = a.arms.split(',')
    seeds = [int(x) for x in a.seeds.split(',')]
    OUT.mkdir(parents=True, exist_ok=True)

    print('=' * 92)
    print(f'  E0 XGBoost OOF   folds {folds}  arms {arms}  seeds {seeds}')
    print(f'  features {len(feats)} (v9 numeric_feats, imported)')
    print('  test.csv 미사용 — fold_S_va 가 held-out 예측 대상')
    print('=' * 92)

    rows = []
    for fold in folds:
        tr, va = load_fold(fold, feats)
        print(f'\n  fold {fold}: train {len(tr):,} (seasons '
              f'{sorted(int(s) for s in tr.season.unique())}) -> '
              f'valid {len(va):,} (season {fold})')
        for arm in arms:
            for seed in seeds:
                pred, meta = run(fold, arm, seed, feats, tr, va,
                                 a.rounds, a.early_stopping)
                pd.DataFrame({'row_id': va.row_id.to_numpy(),
                              'p': pred.astype(np.float32)}).to_parquet(
                    OUT / f'xgb_{fold}_s{seed}_{arm}.parquet', index=False)
                rows.append(meta)

    man = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'),
               params=PARAMS, folds=folds, arms=arms, seeds=seeds,
               n_features=len(feats), runs=rows)
    (OUT / 'manifest.json').write_text(json.dumps(man, indent=1, default=str))
    pd.DataFrame(rows).to_csv(OUT / 'xgb_runs.csv', index=False)
    print(f'\n  wrote {len(rows)} OOF parquets + manifest.json to {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
