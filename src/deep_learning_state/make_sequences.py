"""Mac step 2: turn the pitch log into per-row history windows, sharded.

WHAT A SEQUENCE IS HERE
-----------------------
For a row belonging to pitcher p in season S, the sequence is the L pitches p
threw earlier in the SAME season, most recent last, strictly before this pitch.
Rows earlier than L into the season are left-padded and carry a length.

train.csv is a pitcher-ordered time log -- asof_pitcher_n increases by one per
pitch (verified 792/792 in round 9) -- so row order within (season, pitcher) is
chronological and the window needs no timestamp.

WHY THIS IS LEAKAGE-SAFE, AND WHERE ITS LIMIT IS
------------------------------------------------
Each step carries that past pitch's OUTCOME. Past outcomes of the same season
are exactly what the organisers' own asof_* counters encode, so using them
breaks no rule and imports no future: everything in the window happened before
the row being predicted, and nothing crosses a season boundary.

The limit is deployment, and it is worth stating plainly rather than
discovering it after a full training run: the 2025 test set is 5 isolated
shuffled rows with no labels, and competition rule 5 forbids aggregating across
test rows. A window of the pitcher's previous 16 pitches CANNOT be built at
inference time. So a sequence model here is a measuring instrument -- how much
within-season pitcher state exists, which round 16 priced at +45 Resolution
oracle and could not recover from single-row features -- and a possible teacher
for a single-row student, not itself a submission.

CHANNELS (per past pitch)
    0 outcome (control_success)      the state signal
    1 balls / 3
    2 strikes / 2
    3 outs / 2
    4 inning / 9   (clipped at 9)
    5 batter hand
    6 runners / 3
    7 leverage index, /2 and clipped
"""
import sys
import time
import numpy as np
import pandas as pd

from . import paths

CHANNELS = ['outcome', 'balls', 'strikes', 'outs', 'inning', 'bhand',
            'runners', 'li']
N_CH = len(CHANNELS)


def _steps(df):
    """Per-pitch channel matrix, float16, all in [0, 1]-ish."""
    z = np.zeros((len(df), N_CH), dtype=np.float16)
    z[:, 0] = df.control_success.to_numpy(np.float32)
    z[:, 1] = df.balls_before.to_numpy(np.float32) / 3.0
    z[:, 2] = df.strikes_before.to_numpy(np.float32) / 2.0
    z[:, 3] = df.outs_before.to_numpy(np.float32) / 2.0
    z[:, 4] = np.clip(df.inning.to_numpy(np.float32), 1, 9) / 9.0
    z[:, 5] = df.batter_hand.to_numpy(np.float32)
    z[:, 6] = df.num_runners_on.to_numpy(np.float32) / 3.0
    z[:, 7] = np.clip(df.li.to_numpy(np.float32), 0, 4) / 4.0
    return np.nan_to_num(z)


def _windows(steps, group, L):
    """Left-padded window of the L rows before each row, within its group.

    Vectorised: pad the front of each group by L, then gather with an index
    matrix. Memory is one int32 index array of n x L, released immediately.
    """
    n = len(steps)
    order = np.lexsort((np.arange(n), group))
    g = group[order]
    starts = np.searchsorted(g, g, side='left')
    pos = np.arange(n) - starts                       # index within the group
    offs = np.arange(L, 0, -1)                        # L, L-1, ..., 1  (back)
    take = (np.arange(n)[:, None] - offs[None, :])
    valid = (pos[:, None] - offs[None, :]) >= 0
    take = np.where(valid, take, 0)
    seq = steps[order][take]
    seq[~valid] = 0
    length = np.minimum(pos, L).astype(np.int16)
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)
    return seq[inv], length[inv]


def build_fold(S, L, shard_rows, max_rows=0, force=False):
    from .dataset import feature_cols, fit_scaler, apply_scaler

    FEATURE_COLS = feature_cols()

    folds = paths.DATA / 'folds'
    outs = {}
    scaler = None
    for kind, split in (('tr', 'train'), ('va', 'valid')):
        d = pd.read_parquet(folds / f'fold_{S}_{kind}.parquet')
        d = d.sort_values(['season', 'pitcher_id', 'row_id']).reset_index(drop=True)
        if max_rows:
            d = d.head(max_rows).reset_index(drop=True)

        X = d[FEATURE_COLS].to_numpy(np.float32)
        if scaler is None:                    # fitted on TRAIN only, then frozen
            scaler = fit_scaler(X)
        Xs = apply_scaler(scaler, X)

        grp = (d.season.to_numpy(np.int64) * 1_000_000
               + d.pitcher_id.to_numpy(np.int64))
        seq, length = _windows(_steps(d), grp, L)

        # L is part of the directory name, not just the manifest. Without it an
        # L=32 build silently overwrites the L=16 shards in place while
        # manifest_S{S}_L16.json survives and keeps claiming L=16 -- and the
        # smoke config (L=16) and the full config (L=32) cannot coexist at all.
        dest = paths.SEQ / split / f'S{S}_L{L}'
        dest.mkdir(parents=True, exist_ok=True)
        # always clear: build_fold regenerates every shard, so anything left
        # from a larger previous build would be read as extra rows.
        for f in dest.glob('shard_*.npy'):
            f.unlink()
        n = len(d)
        meta = []
        for i, s in enumerate(range(0, n, shard_rows)):
            e = min(s + shard_rows, n)
            payload = dict(
                x=Xs[s:e].astype(np.float16), seq=seq[s:e], length=length[s:e],
                init=d['init'].to_numpy(np.float32)[s:e],
                y=d.control_success.to_numpy(np.float32)[s:e],
                row_id=d.row_id.to_numpy()[s:e].astype('S16'))
            if 'p_v9' in d.columns:
                payload['p_v9'] = d.p_v9.to_numpy(np.float32)[s:e]
            # One .npy per field, NOT a .npz. np.load(mmap_mode='r') genuinely
            # memory-maps a .npy; on a .npz it silently re-inflates the whole
            # array on every access, which turns a 100k-row epoch into hours.
            for k, v in payload.items():
                np.save(dest / f'shard_{i:04d}_{k}.npy', v)
            meta.append(dict(shard=i, rows=e - s))
        outs[split] = dict(dir=str(dest), rows=n, shards=len(meta),
                           n_static=Xs.shape[1], n_channels=N_CH, L=L)
        print(f'  {split} S{S}: {n:,} rows -> {len(meta)} shards, '
              f'static {Xs.shape[1]}, L={L}')
        del d, X, Xs, seq

    manifest = dict(season=S, sequence_length=L, channels=CHANNELS,
                    feature_cols=FEATURE_COLS,
                    scaler=dict(med=scaler[0].tolist(), iqr=scaler[1].tolist(),
                                miss=scaler[2].tolist()),
                    splits=outs, built=time.strftime('%Y-%m-%d %H:%M:%S'))
    import json
    p = paths.SEQ / f'manifest_S{S}_L{L}.json'
    p.write_text(json.dumps(manifest, indent=1))
    print(f'  manifest -> {p.name}')
    return manifest


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog='make_sequences')
    ap.add_argument('--seasons', default='2023,2024')
    ap.add_argument('--sequence-length', type=int, default=16)
    ap.add_argument('--shard-rows', type=int, default=100_000)
    ap.add_argument('--max-rows', type=int, default=0,
                    help='cap rows per split; 100000 for a smoke build')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(argv)
    paths.ensure_dirs()
    if not (paths.DATA / 'folds').exists():
        raise SystemExit('run prepare_data first (python -m '
                         'src.deep_learning_state.prepare_data)')
    for S in [int(s) for s in a.seasons.split(',')]:
        print(f'season {S}:')
        build_fold(S, a.sequence_length, a.shard_rows, a.max_rows, a.force)


if __name__ == '__main__':
    main()
