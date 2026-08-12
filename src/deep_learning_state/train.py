"""Entry point.

    python -m src.deep_learning_state.train --config configs/gru_smoke.yaml
    python -m src.deep_learning_state.train --config configs/gru_full.yaml --device cuda

Device is auto-detected (CUDA > MPS > CPU) unless --device says otherwise.
Sequential by design: one model at a time, cache cleared between runs.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from . import device as D
from . import envinfo, models, paths
from .dataset import ShardDataset, make_loader
from .metrics import report

# ---------------------------------------------------------------------------
# WHICH EPOCH IS "BEST"
#
# Phase A asks one question: does the pitcher's within-season sequence carry
# Resolution that v9's single-row features do not already have? So Resolution
# is what selects the epoch.
#
# Selecting on BSS -- which is what this used to do -- answers a different
# question. BSS = Resolution - Reliability, so a BSS-best epoch can simply be
# the better-calibrated one. Calibration is exactly what rounds 12-14 already
# established buys nothing here: post-processing did not move Resolution, and
# a model picked for Reliability would look good on the fold and carry no new
# information into a student. Ties (Resolution is a qcut-binned statistic, so
# exact ties are possible when bins collapse) fall back to BSS.
#
# Nothing else changes: every metric is still computed and logged, the v9
# baseline is still v9's own OOF, and the train/valid split is untouched.
SELECTION = 'Resolution (tie-break: BSS)'


def _sel_key(row):
    return (row['Resolution'], row['BSS'])


def _evaluate(model, loader, dev, n):
    model.eval()
    out = np.empty(n, dtype=np.float64)
    i = 0
    with torch.no_grad():
        for x, seq, ln, init, y in loader:
            x, seq = x.to(dev), seq.to(dev)
            ln, init = ln.to(dev), init.to(dev)
            p = torch.sigmoid(init + model(x, seq, ln))
            b = len(p)
            out[i:i + b] = p.float().cpu().numpy()
            i += b
    return out[:i]


def _save(path, model, opt, sched, scaler, epoch, cfg, best, best_epoch=0):
    """`best` is the (Resolution, BSS) key of the selected epoch, not a scalar.
    Both halves are also stored under their own names so a checkpoint says what
    it was selected on without anyone having to read this file."""
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                    sched=sched.state_dict() if sched else None,
                    scaler=scaler.state_dict() if scaler else None,
                    epoch=epoch, config=cfg.to_dict(),
                    best=list(best), best_epoch=best_epoch,
                    selection=SELECTION,
                    best_resolution=best[0], best_bss=best[1],
                    git=envinfo.git_commit()), path)


def main(argv=None):
    cfg, args = C.resolve(argv, prog='src.deep_learning_state.train')
    dev = D.pick(args.device)
    desc = D.describe(dev)
    paths.ensure_dirs()

    # seed is in the stem: train_colab.sh sweeps seeds sequentially with the
    # same cfg.name, and without it seed 44 overwrites seed 42's best.pt.
    tag = f'{cfg.name}_fold{cfg.fold}_s{cfg.seed}'
    ck_dir = Path(cfg.checkpoint_dir or paths.CKPT)
    out_dir = Path(cfg.output_dir or paths.EXP)
    ck_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    tr = ShardDataset(cfg.fold, cfg.sequence_length, 'train', cfg.max_rows)
    va = ShardDataset(cfg.fold, cfg.sequence_length, 'valid', cfg.max_rows)
    print(envinfo.header(cfg, dev, desc, len(tr), len(va)), flush=True)
    if args.dry_run:
        return 0

    nw = D.safe_workers(cfg.num_workers, dev)
    if nw != cfg.num_workers:
        print(f'  num_workers {cfg.num_workers} -> {nw} (macOS/MPS guard)')
    pin = D.pin_memory(dev)
    tl = make_loader(tr, cfg.batch_size, True, nw, pin)
    vl = make_loader(va, cfg.batch_size, False, nw, pin)

    model = models.build(cfg, tr.n_static, tr.n_channels).to(dev)
    print(f'  model      {cfg.model}  {models.count_params(model):,} params',
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                            weight_decay=cfg.weight_decay)
    steps = max(1, cfg.epochs * ((len(tr) + cfg.batch_size - 1) // cfg.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.learning_rate, total_steps=steps, pct_start=0.25)
    amp = cfg.mixed_precision and D.supports_amp(dev)
    if cfg.mixed_precision and not amp:
        print(f'  mixed_precision ignored on {dev.type} (unsupported)')
    scaler = torch.amp.GradScaler('cuda') if amp else None

    start_epoch, best, best_epoch = 1, (-1e18, -1e18), 0
    if cfg.resume:
        ck = torch.load(cfg.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        if ck.get('scaler') and scaler:
            scaler.load_state_dict(ck['scaler'])
        start_epoch = ck['epoch'] + 1
        # a checkpoint written before the criterion changed stores a bare BSS
        # float; it is not comparable, so start the search over rather than
        # letting a BSS-selected epoch win a Resolution race it never entered.
        b = ck.get('best')
        if isinstance(b, (list, tuple)) and len(b) == 2:
            best, best_epoch = tuple(b), ck.get('best_epoch', ck['epoch'])
        elif b is not None:
            print('  checkpoint predates the Resolution criterion; '
                  'best-epoch search restarts', flush=True)
        # The scheduler state is deliberately NOT loaded. OneCycleLR stores its
        # own total_steps, so restoring it into a run with a different --epochs
        # overruns the schedule on the first step. Fast-forwarding the fresh
        # scheduler instead keeps the shape tied to the CURRENT config.
        done = (start_epoch - 1) * ((len(tr) + cfg.batch_size - 1) // cfg.batch_size)
        if done >= steps:
            raise SystemExit(
                f'checkpoint is at epoch {ck["epoch"]} but --epochs is '
                f'{cfg.epochs}; nothing left to run. Raise --epochs to resume.')
        for _ in range(done):
            sched.step()
        print(f'  resumed from {cfg.resume} at epoch {start_epoch} '
              f'(lr schedule fast-forwarded {done}/{steps} steps)', flush=True)

    y = va.column('y').astype(np.float64)
    p9c = va.column('p_v9')
    p9 = p9c.astype(np.float64) if p9c is not None else None
    bce = nn.BCEWithLogitsLoss()
    rows, t_start = [], time.time()

    for ep in range(start_epoch, cfg.epochs + 1):
        model.train()
        tot, seen, t0 = 0.0, 0, time.time()
        for x, seq, ln, init, yb in tl:
            x, seq = x.to(dev, non_blocking=pin), seq.to(dev, non_blocking=pin)
            ln, init, yb = ln.to(dev), init.to(dev), yb.to(dev)
            with D.autocast_ctx(dev, amp):
                loss = bce(init + model(x, seq, ln), yb)
            opt.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
            sched.step()
            tot += loss.detach().item() * len(yb)
            seen += len(yb)
        train_loss = tot / max(seen, 1)

        row = dict(epoch=ep, train_loss=train_loss,
                   train_seconds=round(time.time() - t0, 1))
        if ep % cfg.eval_every == 0 or ep == cfg.epochs:
            p = _evaluate(model, vl, dev, len(va))
            row.update(report(y[:len(p)], p, p9[:len(p)] if p9 is not None else None,
                              name=cfg.name))
            print('  ep%-3d loss %.5f  BSS %8.1f  Res %8.1f  Rel %7.1f  '
                  'dRes %+8.1f  [%.0fs]'
                  % (ep, train_loss, row['BSS'], row['Resolution'],
                     row['Reliability'], row.get('dResolution', float('nan')),
                     row['train_seconds']), flush=True)
            if _sel_key(row) > best:
                best, best_epoch = _sel_key(row), ep
                row['selected'] = True
                _save(ck_dir / f'{tag}_best.pt', model, opt,
                      sched, scaler, ep, cfg, best, best_epoch)
        else:
            print('  ep%-3d loss %.5f  [%.0fs]' % (ep, train_loss,
                                                   row['train_seconds']),
                  flush=True)
        rows.append(row)
        if ep % cfg.save_every == 0:
            _save(ck_dir / f'{tag}_last.pt', model, opt,
                  sched, scaler, ep, cfg, best, best_epoch)
        # `patience` was declared in the config but never read, so a 20-epoch
        # run always burned all 20. Counted in EVALUATED epochs, not raw ones,
        # and on the selection criterion so it cannot disagree with it.
        if cfg.patience and best_epoch and ep - best_epoch >= cfg.patience:
            print(f'  early stop: no Resolution improvement for {cfg.patience} '
                  f'evals (best epoch {best_epoch}, Res {best[0]:.1f})',
                  flush=True)
            break

    # ---- results ---------------------------------------------------------
    # The reported row is the SELECTED epoch, same criterion as _best.pt, so
    # results.csv and the checkpoint can never describe different epochs.
    import pandas as pd
    last = max((r for r in rows if 'BSS' in r), key=_sel_key)
    print(f'\n  selected   epoch {last["epoch"]} by {SELECTION}'
          f'   Res {last["Resolution"]:.1f}  BSS {last["BSS"]:.1f}', flush=True)
    rec = dict(run=f'{cfg.name}-{envinfo.stamp()}', git_commit=envinfo.git_commit(),
               config=Path(args.config).name, fold=cfg.fold, seed=cfg.seed,
               model=cfg.model, sequence_length=cfg.sequence_length,
               hidden_size=cfg.hidden_size, epochs=cfg.epochs,
               batch_size=cfg.batch_size, device=str(dev), gpu=desc['gpu'],
               dataset=envinfo.dataset_version(cfg.fold, cfg.sequence_length),
               train_rows=len(tr), valid_rows=len(va),
               total_seconds=round(time.time() - t_start, 1),
               selection=SELECTION,
               **{k: last.get(k) for k in
                  ('epoch', 'BSS', 'Resolution', 'Reliability', 'LogLoss',
                   'AUC', 'pred_std', 'corr_with_v9', 'dBSS', 'dResolution',
                   'dReliability', 'paired', 'paired_se')})
    res = out_dir / 'results.csv'
    new = pd.DataFrame([rec])
    if not res.exists():
        new.to_csv(res, index=False)
    else:
        old = pd.read_csv(res)
        if list(old.columns) == list(new.columns):
            new.to_csv(res, mode='a', header=False, index=False)
        else:
            # a column was added (`selection`). A blind append would put the
            # values under the wrong headers for every future row, so migrate
            # the log once; older rows get NaN where they have no value.
            pd.concat([old, new], ignore_index=True).to_csv(res, index=False)
    (out_dir / f'{rec["run"]}_trace.json').write_text(
        json.dumps(dict(config=cfg.to_dict(), env=desc, selection=SELECTION,
                        best_epoch=last['epoch'], trace=rows), indent=1,
                   default=float))
    print(f'\n  results -> {res}')
    print(f'  checkpoints -> {ck_dir}')
    D.empty_cache(dev)
    return 0


if __name__ == '__main__':
    sys.exit(main())
