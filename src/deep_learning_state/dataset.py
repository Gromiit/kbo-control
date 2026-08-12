"""Shard-backed dataset: disk -> CPU -> batch -> GPU. Never the whole set on GPU.

Shards are memory-mapped through np.load(mmap_mode='r'), so resident memory is
one shard's worth of pages plus the batch, whatever the fold size. That is what
makes a 1.2M-row fold trainable on a 16 GB laptop and on a Colab runtime with
an unknown amount of free RAM.

Batches are assembled on the CPU as float16/int and cast on the device, so the
host-to-device transfer is half the bytes of a float32 batch.
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from . import paths

_FC = None


def feature_cols():
    """The round-17 feature list is the source of truth, imported not copied.

    Mac-only: it needs work/features.py. Training on Colab reads the column
    names out of the shard manifest instead and never calls this.
    """
    global _FC
    if _FC is None:
        import sys
        sys.path.insert(0, str(paths.LEGACY_DL))
        paths.add_work_to_syspath()
        from data import numeric_feats
        _FC = list(numeric_feats())
    return _FC


def fit_scaler(X):
    """Median-impute + robust-scale + missing indicators. Fitted on train only."""
    with np.errstate(all='ignore'):
        med = np.nanmedian(X, axis=0)
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
    # a column that is all-NaN (happens on a truncated smoke slice) would send
    # every value to NaN downstream; it carries no information, so pin it to 0
    med = np.nan_to_num(med, nan=0.0)
    iqr = np.where(np.isfinite(q3 - q1) & ((q3 - q1) > 1e-9), q3 - q1, 1.0)
    miss = np.isnan(X).mean(0) > 1e-6
    return med, iqr, miss


def apply_scaler(sc, X):
    med, iqr, miss = sc
    m = np.isnan(X)
    Z = np.clip((np.where(m, med, X) - med) / iqr, -8, 8).astype(np.float32)
    if np.any(miss):
        Z = np.hstack([Z, m[:, miss].astype(np.float32)])
    return Z


def load_manifest(season, L):
    p = paths.SEQ / f'manifest_S{season}_L{L}.json'
    if not p.exists():
        raise SystemExit(
            f'no shards for season {season}, L={L}.\n'
            f'Build them on the Mac:\n'
            f'  python -m src.deep_learning_state.make_sequences '
            f'--seasons {season} --sequence-length {L}')
    return json.loads(p.read_text())


FIELDS = ('x', 'seq', 'length', 'init', 'y')


class ShardDataset(Dataset):
    """Memory-mapped shards with a BATCHED fetch path.

    Two things matter for speed here and both were learned the hard way:

      * fields are separate .npy files, so mmap_mode='r' really maps them and
        the OS pages in only what a batch touches;
      * `__getitems__` (torch >= 2.0) receives the whole index list, so a batch
        costs one vectorised gather instead of 512 Python round trips.
    """

    def __init__(self, season, L, split, max_rows=0):
        man = load_manifest(season, L)
        info = man['splits'][split]
        self.dir = paths.SEQ / split / f'S{season}'
        self.n_shards = len(sorted(self.dir.glob('shard_*_y.npy')))
        if not self.n_shards:
            raise SystemExit(f'no shards in {self.dir}')
        self.maps = []
        counts = []
        for i in range(self.n_shards):
            m = {f: np.load(self.dir / f'shard_{i:04d}_{f}.npy', mmap_mode='r')
                 for f in FIELDS}
            self.maps.append(m)
            counts.append(len(m['y']))
        self.offsets = np.cumsum([0] + counts)
        self.n = int(self.offsets[-1])
        if max_rows:
            self.n = min(self.n, max_rows)
        self.n_static = info['n_static']
        self.n_channels = info['n_channels']
        self.L = info['L']

    def __len__(self):
        return self.n

    def _gather(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        k = np.searchsorted(self.offsets, idx, side='right') - 1
        out = {f: [] for f in FIELDS}
        order = np.argsort(k, kind='stable')
        for s in np.unique(k):
            sel = order[k[order] == s]
            j = idx[sel] - int(self.offsets[s])
            js = np.sort(j)                      # mmap likes ascending reads
            back = np.argsort(np.argsort(j))
            for f in FIELDS:
                out[f].append(np.asarray(self.maps[s][f][js])[back])
        return {f: np.concatenate(v) for f, v in out.items()}

    def __getitems__(self, idxs):
        g = self._gather(idxs)
        return (torch.from_numpy(g['x'].astype(np.float32, copy=False)),
                torch.from_numpy(g['seq'].astype(np.float32, copy=False)),
                torch.from_numpy(g['length'].astype(np.int64, copy=False)),
                torch.from_numpy(g['init'].astype(np.float32, copy=False)),
                torch.from_numpy(g['y'].astype(np.float32, copy=False)))

    def __getitem__(self, i):
        return tuple(t[0] for t in self.__getitems__([i]))

    def column(self, name):
        """Whole-column read for scoring -- one pass, never held twice."""
        parts = []
        for i in range(self.n_shards):
            p = self.dir / f'shard_{i:04d}_{name}.npy'
            if not p.exists():
                return None
            parts.append(np.load(p))
        return np.concatenate(parts)[:self.n]


def _identity(b):
    """__getitems__ already returns collated tensors."""
    return b


def make_loader(ds, batch_size, shuffle, num_workers, pin):
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=pin, drop_last=False, collate_fn=_identity,
        persistent_workers=bool(num_workers))
