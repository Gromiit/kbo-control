"""Leakage audit. Runs on the Mac, against the shards that will be uploaded.

Six checks, each of which has actually caught something in this project or in
this refactor:

  1. season isolation      no training row is from season >= S
  2. no row overlap        train and valid share no row_id
  3. window causality      every sequence step is a real earlier pitch of the
                           SAME pitcher in the SAME season -- verified by
                           rebuilding the window from the parquet and comparing
  4. padding correctness   rows with a short history have zeros, and `length`
                           matches the number of non-padded steps
  5. scaler provenance     the scaler stored in the manifest is the TRAIN one;
                           refitting on train reproduces it exactly
  6. v9 alignment          valid p_v9 matches the stored OOF prediction row for
                           row, so dResolution is measured against the right
                           baseline

Exit code is non-zero if any check fails, so scripts/mac_prepare.sh stops
before anything gets uploaded.
"""
import json
import sys

import numpy as np
import pandas as pd

from . import paths

OK, BAD = '  [ok]  ', '  [FAIL]'


class Audit:
    def __init__(self):
        self.fail = 0

    def check(self, name, cond, detail=''):
        print(f'{OK if cond else BAD} {name}' + (f'   {detail}' if detail else ''))
        if not cond:
            self.fail += 1
        return cond


def audit_season(a, S, L):
    man = json.loads((paths.SEQ / f'manifest_S{S}_L{L}.json').read_text())
    folds = paths.DATA / 'folds'
    tr = pd.read_parquet(folds / f'fold_{S}_tr.parquet')
    va = pd.read_parquet(folds / f'fold_{S}_va.parquet')

    print(f'\n--- season {S}, L={L} ---')
    a.check('1. 학습 시즌이 전부 S 미만',
            int(tr.season.max()) < S,
            f'train {tr.season.min()}-{tr.season.max()}, valid {S}')
    a.check('2. train/valid row_id 겹침 없음',
            len(set(tr.row_id) & set(va.row_id)) == 0,
            f'{len(tr):,} vs {len(va):,}')

    # ---- 3/4: rebuild one split's windows and compare against the shards
    from .make_sequences import _steps, _windows
    for split, d in (('train', tr), ('valid', va)):
        info = man['splits'][split]
        n_shard = info['rows']
        d = d.sort_values(['season', 'pitcher_id', 'row_id']).reset_index(drop=True)
        d = d.head(n_shard).reset_index(drop=True)
        grp = (d.season.to_numpy(np.int64) * 1_000_000
               + d.pitcher_id.to_numpy(np.int64))
        seq_ref, len_ref = _windows(_steps(d), grp, L)

        sd = paths.SEQ / split / f'S{S}_L{L}'
        ns = len(sorted(sd.glob('shard_*_y.npy')))
        cat = lambda f: np.concatenate(
            [np.load(sd / f'shard_{i:04d}_{f}.npy') for i in range(ns)])
        seq, ln = cat('seq'), cat('length')
        rid = cat('row_id').astype(str)

        a.check(f'3. {split}: shard 순서가 parquet 순서와 일치',
                bool((rid == d.row_id.to_numpy().astype(str)).all()))
        a.check(f'3. {split}: shard seq 폭 == manifest L',
                seq.shape[1] == L, f'{seq.shape[1]} vs {L}')
        a.check(f'3. {split}: window 가 재계산과 비트 일치',
                bool(np.array_equal(seq, seq_ref)) and bool(np.array_equal(ln, len_ref)))

        # the causal property itself, checked directly on a sample
        rng = np.random.default_rng(0)
        pick = rng.choice(len(d), size=min(2000, len(d)), replace=False)
        pos_in_grp = np.arange(len(d)) - np.searchsorted(grp, grp, side='left')
        a.check(f'4. {split}: length == min(앞선 같은 투수·시즌 투구 수, L)',
                bool((ln[pick] == np.minimum(pos_in_grp[pick], L)).all()))
        pad_ok = True
        for i in pick[:500]:
            k = int(ln[i])
            if k < L and seq[i, :L - k].any():
                pad_ok = False
                break
        a.check(f'4. {split}: 패딩 구간이 전부 0', pad_ok)

        # the outcome channel of the last step must equal the label of the
        # immediately preceding row of the same pitcher -- the sharpest test
        # that no step is the row's own future
        same = pos_in_grp[pick] > 0
        idx = pick[same]
        prev_y = d.control_success.to_numpy(np.float32)[idx - 1]
        a.check(f'4. {split}: 마지막 step = 직전 투구의 결과 (자기 자신 아님)',
                bool(np.allclose(seq[idx, -1, 0].astype(np.float32), prev_y)))

    # ---- 5: scaler came from train, and from the SAME train slice.
    # The reference must be refitted on exactly the rows build_fold saw --
    # sorted, and truncated to the shard row count when --max-rows was used.
    from .dataset import feature_cols, fit_scaler
    tr_slice = (tr.sort_values(['season', 'pitcher_id', 'row_id'])
                  .head(man['splits']['train']['rows']))
    ref = fit_scaler(tr_slice[feature_cols()].to_numpy(np.float32))
    got = man['scaler']
    a.check('5. manifest scaler == train 에서 재적합한 scaler',
            np.allclose(ref[0], np.array(got['med']), equal_nan=True)
            and np.allclose(ref[1], np.array(got['iqr']))
            and (np.array(got['miss'], bool) == ref[2]).all())

    # ---- 6: v9 baseline alignment
    if 'p_v9' in va.columns:
        sd = paths.SEQ / 'valid' / f'S{S}_L{L}'
        ns = len(sorted(sd.glob('shard_*_y.npy')))
        got = np.concatenate(
            [np.load(sd / f'shard_{i:04d}_p_v9.npy') for i in range(ns)])
        ref = (va.sort_values(['season', 'pitcher_id', 'row_id'])
                 .head(len(got)).p_v9.to_numpy(np.float32))
        a.check('6. valid p_v9 가 v9 OOF 와 행 단위 일치',
                bool(np.allclose(got, ref, atol=1e-6)),
                f'n={len(got):,}')


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog='audit')
    ap.add_argument('--seasons', default='2023,2024')
    ap.add_argument('--sequence-length', type=int, default=16)
    x = ap.parse_args(argv)
    a = Audit()
    for S in [int(s) for s in x.seasons.split(',')]:
        audit_season(a, S, x.sequence_length)
    print('\n' + ('leakage audit PASSED' if not a.fail
                  else f'leakage audit FAILED ({a.fail} checks)'))
    return 1 if a.fail else 0


if __name__ == '__main__':
    sys.exit(main())
