"""Mac step 1: walk-forward featurised folds. Reads train.csv, writes parquet.

This is a thin, path-aware wrapper around the round-17 builder
(work/research/deep_learning/data.py). The leakage rule it enforces is the one
v9's own backtest used, in exp5.build_split:

    for validation season S, every lookup table -- carry, split, role,
    low-rank -- and the level model's logistic regression are fitted on
    seasons < S ONLY, and season S is then pushed through those frozen
    tables.

That is the same path a 2025 test row takes at inference. No row of season S
influences anything season S is scored on.

If the round-17 cache is already present it is reused as-is rather than
rebuilt, so this is cheap to re-run.
"""
import sys
import time

from . import paths

# raw situation columns the sequence channels need but the round-17 cache drops
EXTRA_COLS = ['li']


def main(seasons=(2022, 2023, 2024), force=False):
    paths.require_mac_roots()
    paths.ensure_dirs()
    out = paths.DATA / 'folds'
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.LEGACY_DL))
    paths.add_work_to_syspath()
    import data as legacy                       # round-17 builder, unmodified

    t0 = time.time()
    legacy.build_cache(force=force)             # writes into LEGACY_DL

    # The round-17 cache keeps only what an MLP needed. The sequence channels
    # want a couple of raw situation columns as well, so they are joined on
    # here rather than by editing the legacy builder.
    import pandas as pd
    extra = ['row_id'] + EXTRA_COLS
    raw = pd.read_csv(paths.RAW / 'train.csv', usecols=extra,
                      encoding='utf-8-sig')

    for S in seasons:
        for kind in ('tr', 'va'):
            src = paths.LEGACY_DL / f'dl_{S}_{kind}.parquet'
            dst = out / f'fold_{S}_{kind}.parquet'
            if force or not dst.exists():
                d = pd.read_parquet(src)
                miss = [c for c in EXTRA_COLS if c not in d.columns]
                if miss:
                    d = d.merge(raw[['row_id'] + miss], on='row_id', how='left')
                    assert d[miss].notna().all().all(), (S, kind, miss)
                d.to_parquet(dst, index=False)
            print(f'  fold_{S}_{kind}.parquet  {dst.stat().st_size/1e6:.0f} MB')
    print(f'folds ready in {out}  [{time.time()-t0:.0f}s]')
    return out


if __name__ == '__main__':
    main(force='--force' in sys.argv)
