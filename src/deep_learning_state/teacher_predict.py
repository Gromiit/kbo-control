"""Phase C-1: teacher predictions from the fold-2024 GRU seeds. INFERENCE ONLY.

This module never trains. It builds no optimiser, takes no gradient step, and
runs entirely under torch.no_grad() with the model in eval(). The checkpoints
are read-only inputs.

WHAT IT PRODUCES AND WHY
------------------------
Phase A answered "is there within-season pitcher state that v9 does not have?"
with yes -- fold 2024, 3-seed mean dResolution +102. That model cannot be
submitted: it reads the pitcher's previous 32 pitches of the same season, and
the 2025 test set is five shuffled unlabelled rows, so the window cannot be
built at inference (competition rule 5, and script.py's own contract: "No
statistic is ever pooled across rows of the evaluation set").

So it becomes a teacher. This writes its predictions to disk so Phase C-2 can
ask the only question that matters next: how much of the teacher's edge over
v9 can a SINGLE-ROW student recover? Everything downstream reads this parquet,
so it is produced once and never regenerated on the fly.

SEED AVERAGING
--------------
`teacher_mean` averages the three seeds in LOGIT space, then squashes. That is
the convention v9 itself uses -- script.py's a_family() says "Seeds average in
raw-score space, groups in probability space" -- so a teacher built the same
way stays comparable to the baseline it is measured against. The
probability-space average is computed too and printed side by side, so the
choice is visible rather than buried.

VERIFYING THE CHECKPOINTS LOADED CORRECTLY
------------------------------------------
Per-seed dResolution printed here must match the row for that seed in
results.csv. Same weights, same rows, same metric -- a mismatch means the
checkpoint did not load as intended, not that the teacher changed.

    python -m src.deep_learning_state.teacher_predict --fold 2024 \
        --sequence-length 32 --seeds 42,43,44
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from . import config as C
from . import device as D
from . import models, paths
from .dataset import ShardDataset, make_loader
from .metrics import decomp, report


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _load_model(ck_path, n_static, n_channels, fold, L, dev):
    """Rebuild the exact architecture the checkpoint was written with.

    The config travels inside the checkpoint, so nothing here guesses at
    hidden_size or num_layers. Everything that could silently produce a
    different model -- a checkpoint from another fold, another window length,
    another architecture -- is refused rather than loaded.
    """
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    cfg = C.Config(**ck['config'])
    if cfg.fold != fold:
        raise SystemExit(f'{ck_path.name}: fold {cfg.fold}, expected {fold}')
    if cfg.sequence_length != L:
        raise SystemExit(f'{ck_path.name}: L={cfg.sequence_length}, expected {L}')
    if cfg.model != 'gru':
        raise SystemExit(f'{ck_path.name}: model {cfg.model!r}, expected gru')
    model = models.build(cfg, n_static, n_channels).to(dev)
    model.load_state_dict(ck['model'])        # strict: shape drift raises here
    model.eval()
    return model, ck, cfg


@torch.no_grad()
def _predict_logits(model, loader, dev, n):
    """Returns init + f(x, seq), i.e. the LOGIT. Probabilities are derived
    later so the seeds can be averaged in either space."""
    out = np.empty(n, dtype=np.float64)
    i = 0
    for x, seq, ln, init, _y in loader:
        x, seq = x.to(dev), seq.to(dev)
        ln, init = ln.to(dev), init.to(dev)
        z = init + model(x, seq, ln)
        b = len(z)
        out[i:i + b] = z.float().cpu().numpy()
        i += b
    if i != n:
        raise SystemExit(f'predicted {i} rows, dataset has {n}')
    return out


def _cross_check_parquet(fold, row_id):
    """Mac only. The shards carry no season column -- valid is season `fold` by
    construction and audit check 1 proves it -- but when the fold parquet is at
    hand, confirm the row order too rather than trusting it."""
    p = paths.DATA / 'folds' / f'fold_{fold}_va.parquet'
    if not p.exists():
        return None
    import pandas as pd
    d = (pd.read_parquet(p, columns=['row_id', 'season', 'pitcher_id'])
           .sort_values(['season', 'pitcher_id', 'row_id']).reset_index(drop=True))
    ref = d.row_id.to_numpy().astype(str)[:len(row_id)]
    ok = bool((ref == row_id).all())
    seasons = sorted(set(d.season.to_numpy()[:len(row_id)].tolist()))
    return ok, seasons


def main(argv=None):
    ap = argparse.ArgumentParser(prog='teacher_predict')
    ap.add_argument('--fold', type=int, default=2024)
    ap.add_argument('--sequence-length', type=int, default=32)
    ap.add_argument('--seeds', default='42,43,44')
    ap.add_argument('--name', default='gru_full',
                    help='checkpoint stem: {name}_fold{fold}_s{seed}_best.pt')
    ap.add_argument('--checkpoint-dir', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--device', default=None)
    a = ap.parse_args(argv)

    seeds = [int(s) for s in a.seeds.split(',')]
    dev = D.pick(a.device)
    ck_dir = Path(a.checkpoint_dir or paths.CKPT)
    out_p = Path(a.out or (paths.EXP / f'teacher_predictions_S{a.fold}.parquet'))

    va = ShardDataset(a.fold, a.sequence_length, 'valid')
    n = len(va)
    man_rows = int(__import__('json').loads(
        (paths.SEQ / f'manifest_S{a.fold}_L{a.sequence_length}.json').read_text()
    )['splits']['valid']['rows'])

    print('=' * 70)
    print(f'  Phase C-1 teacher predictions   fold {a.fold}  L={a.sequence_length}')
    print(f'  device      {dev}')
    print(f'  shards      {va.dir}')
    print(f'  rows        {n:,}   (manifest {man_rows:,})')
    print(f'  checkpoints {ck_dir}')
    print('=' * 70)
    if n != man_rows:
        raise SystemExit(f'row count {n:,} != manifest {man_rows:,}')

    y = va.column('y').astype(np.float64)
    p9 = va.column('p_v9')
    if p9 is None:
        raise SystemExit(f'valid split has no p_v9; cannot compare to the '
                         f'v9 baseline for fold {a.fold}')
    p9 = p9.astype(np.float64)
    row_id = va.column('row_id').astype(str)
    for nm, arr in (('y', y), ('p_v9', p9), ('row_id', row_id)):
        if len(arr) != n:
            raise SystemExit(f'{nm} has {len(arr)} rows, expected {n}')

    loader = make_loader(va, a.batch_size, False,
                         D.safe_workers(a.num_workers, dev), D.pin_memory(dev))

    # ---- per-seed inference ------------------------------------------------
    logits, meta = {}, {}
    for s in seeds:
        p = ck_dir / f'{a.name}_fold{a.fold}_s{s}_best.pt'
        if not p.exists():
            raise SystemExit(
                f'missing checkpoint {p}\n'
                f'Colab writes these to $KBO_CKPT (Drive). Point --checkpoint-dir '
                f'at them or set KBO_CKPT.')
        model, ck, cfg = _load_model(p, va.n_static, va.n_channels,
                                     a.fold, a.sequence_length, dev)
        logits[s] = _predict_logits(model, loader, dev, n)
        meta[s] = dict(epoch=ck.get('epoch'), best_epoch=ck.get('best_epoch'),
                       selection=ck.get('selection'), git=ck.get('git'),
                       params=models.count_params(model))
        print(f'  seed {s}: loaded {p.name}  best_epoch={meta[s]["best_epoch"]}  '
              f'{meta[s]["params"]:,} params  git={meta[s]["git"]}')
        del model
        D.empty_cache(dev)

    Z = np.stack([logits[s] for s in seeds])          # n_seeds x n
    p_seed = _sigmoid(Z)
    p_logit_mean = _sigmoid(Z.mean(0))                # v9's convention
    p_prob_mean = p_seed.mean(0)

    # ---- verification ------------------------------------------------------
    print('\n--- row count ---')
    print(f'  dataset {n:,} == manifest {man_rows:,}  [ok]')
    xc = _cross_check_parquet(a.fold, row_id)
    if xc is None:
        print('  fold parquet 없음 (Colab) — season은 fold 구성상 '
              f'{a.fold} 로 확정, audit check 1 이 보증')
    else:
        ok, seasons = xc
        print(f'  parquet row_id 순서 일치: {ok}   valid seasons: {seasons}')
        if not ok:
            raise SystemExit('shard row order does not match the fold parquet')

    print('\n--- seed prediction correlation (probability space) ---')
    print('        ' + '  '.join(f's{s:<7d}' for s in seeds))
    for i, si in enumerate(seeds):
        cells = '  '.join(f'{np.corrcoef(p_seed[i], p_seed[j])[0,1]:<8.5f}'
                          for j in range(len(seeds)))
        print(f'  s{si:<4d} {cells}')
    print(f'  seed 간 pred std (행별 평균): {p_seed.std(0).mean():.5f}')

    print('\n--- metrics vs v9 baseline ---')
    bres, brel, bbss = decomp(y, p9)
    print(f'  {"v9 (baseline)":<22s} BSS {bbss:9.1f}  Res {bres:9.1f}  '
          f'Rel {brel:8.1f}')
    for i, s in enumerate(seeds):
        d = report(y, p_seed[i], p9, name=f'seed{s}')
        print(f'  {"teacher_s"+str(s):<22s} BSS {d["BSS"]:9.1f}  '
              f'Res {d["Resolution"]:9.1f}  Rel {d["Reliability"]:8.1f}  '
              f'dRes {d["dResolution"]:+8.1f}  (results.csv 와 일치해야 함)')
    for label, p in (('teacher_mean (logit)', p_logit_mean),
                     ('teacher_mean (prob)', p_prob_mean)):
        d = report(y, p, p9, name=label)
        print(f'  {label:<22s} BSS {d["BSS"]:9.1f}  Res {d["Resolution"]:9.1f}  '
              f'Rel {d["Reliability"]:8.1f}  dRes {d["dResolution"]:+8.1f}  '
              f'dBSS {d["dBSS"]:+8.1f}  paired {d["paired"]:+.1f}±{d["paired_se"]:.1f}')

    # ---- write -------------------------------------------------------------
    import pandas as pd
    df = pd.DataFrame({
        'row_id': row_id,
        'season': np.full(n, a.fold, dtype=np.int32),
        'control_success': y.astype(np.float32),
        'p_v9': p9.astype(np.float32),
        **{f'teacher_s{s}': p_seed[i].astype(np.float32)
           for i, s in enumerate(seeds)},
        'teacher_mean': p_logit_mean.astype(np.float32),
    })
    if len(df) != n:
        raise SystemExit(f'frame has {len(df)} rows, expected {n}')
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_p, index=False)
    print(f'\n  wrote {out_p}   {len(df):,} rows x {len(df.columns)} cols '
          f'({out_p.stat().st_size/1e6:.1f} MB)')
    print(f'  columns: {list(df.columns)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
