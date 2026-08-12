"""Verify an extracted shard tree. Runs anywhere -- Mac or Colab.

WHY THIS IS NOT audit.py
------------------------
audit.py is the leakage audit and it is a MAC step: it rebuilds every window
from data/folds/*.parquet and compares bit for bit, refits the scaler, and
matches p_v9 against v9's OOF file. None of those inputs exist on Colab (the
fold parquets are 1.5 GB and are deliberately not uploaded), so audit.py
cannot run there and this module does not try to reimplement it.

What this checks instead is that the shard tree that arrived is the shard tree
that was built: the inventory is complete and the invariants that are visible
in the shards ALONE still hold after the tar round trip.

    inventory      every field present, one file per shard, counts agree with
                   the manifest, p_v9 in valid only
    shapes         seq width == manifest L, channels and static width agree,
                   summed rows == manifest rows
    windows        left-padding: the leading L-length steps are exactly zero,
                   the trailing `length` steps are not. Every real step carries
                   inning/9 >= 1/9, so "all-zero step" and "padding" are the
                   same thing and `length` is checkable without the parquet.
    values         y in {0,1}, init finite, p_v9 in (0,1)
    AppleDouble    macOS tar ships `._*` sidecars; prove the shard glob is
                   blind to them rather than assuming it

The split of responsibility is deliberate. Season isolation, row_id overlap,
window-vs-parquet equality, scaler provenance and p_v9-vs-OOF alignment are
properties of how the shards were BUILT and were established on the Mac before
upload. This checks they survived TRANSPORT. Use --sha256 to close the gap
completely: write the sums next to the build, verify them after extraction.

    python -m src.deep_learning_state.check_shards --seasons 2023,2024 \
        --sequence-length 32
"""
import json
import sys

import numpy as np

from . import paths

OK, BAD = '  [ok]  ', '  [FAIL]'
FIELDS = ('seq', 'x', 'row_id', 'y', 'length', 'init')


class Check:
    def __init__(self):
        self.fail = 0

    def __call__(self, name, cond, detail=''):
        cond = bool(cond)
        print(f'{OK if cond else BAD} {name}' + (f'   {detail}' if detail else ''),
              flush=True)
        if not cond:
            self.fail += 1
        return cond


def _shard_files(d, field):
    """The same glob ShardDataset uses. `._shard_0000_y.npy` does not match it
    because the pattern is anchored at the start of the basename."""
    return sorted(d.glob(f'shard_*_{field}.npy'))


def check_split(c, man, S, L, split):
    d = paths.SEQ / split / f'S{S}_L{L}'
    info = man['splits'][split]
    tag = f'{split} S{S}_L{L}'

    if not c(f'{tag}: 디렉터리 존재', d.is_dir(), str(d)):
        return

    n = len(_shard_files(d, 'y'))
    c(f'{tag}: shard 수 == manifest', n == info['shards'],
      f'{n} vs {info["shards"]}')

    counts = {f: len(_shard_files(d, f)) for f in FIELDS}
    c(f'{tag}: 모든 field가 shard 수와 일치',
      all(v == n for v in counts.values()),
      '  '.join(f'{k}={v}' for k, v in counts.items()))

    n_p9 = len(_shard_files(d, 'p_v9'))
    want = n if split == 'valid' else 0
    c(f'{tag}: p_v9 는 valid 에만', n_p9 == want, f'{n_p9} (기대 {want})')

    # AppleDouble sidecars: present is fine, matched is not
    dot = sorted(p for p in d.iterdir() if p.name.startswith('._'))
    c(f'{tag}: ._AppleDouble 이 shard glob 에 안 잡힘',
      not any(p.name.startswith('._') for p in _shard_files(d, 'y')),
      f'{len(dot)}개 발견, glob 결과 {n}개' if dot else '사이드카 없음')

    if n == 0:
        return

    rows = 0
    shape_ok = value_ok = True
    for i in range(n):
        y = np.load(d / f'shard_{i:04d}_y.npy', mmap_mode='r')
        seq = np.load(d / f'shard_{i:04d}_seq.npy', mmap_mode='r')
        x = np.load(d / f'shard_{i:04d}_x.npy', mmap_mode='r')
        ln = np.load(d / f'shard_{i:04d}_length.npy', mmap_mode='r')
        rows += len(y)
        if (seq.shape[1:] != (L, info['n_channels'])
                or x.shape[1] != info['n_static']
                or len(seq) != len(y) or len(x) != len(y) or len(ln) != len(y)):
            shape_ok = False
        if not np.isin(np.asarray(y), (0.0, 1.0)).all():
            value_ok = False
        if not np.isfinite(np.load(d / f'shard_{i:04d}_init.npy')).all():
            value_ok = False

    c(f'{tag}: 총 행 수 == manifest', rows == info['rows'],
      f'{rows:,} vs {info["rows"]:,}')
    c(f'{tag}: shape (seq L·채널, static, 행 정렬)', shape_ok,
      f'seq(*, {L}, {info["n_channels"]})  x(*, {info["n_static"]})')
    c(f'{tag}: y in {{0,1}}, init 유한', value_ok)

    if split == 'valid' and n_p9 == n:
        p9 = np.concatenate([np.load(d / f'shard_{i:04d}_p_v9.npy')
                             for i in range(n)])
        c(f'{tag}: p_v9 in (0,1)',
          bool(np.isfinite(p9).all() and (p9 > 0).all() and (p9 < 1).all()),
          f'[{p9.min():.4f}, {p9.max():.4f}]  n={len(p9):,}')

    # ---- window invariants, from the shards alone --------------------------
    # every real step has inning/9 >= 1/9, so a step is padding iff it is all
    # zero. That makes `length` and the left-padding checkable here.
    # EXHAUSTIVE, not sampled: a spot check on 2% of rows would miss exactly
    # the single-shard corruption this is here to find. Blocked so peak memory
    # is one block, not one shard.
    BLOCK = 20_000
    bad_pad = bad_len = checked = 0
    cols = np.arange(L)
    for i in range(n):
        seq = np.load(d / f'shard_{i:04d}_seq.npy', mmap_mode='r')
        ln = np.load(d / f'shard_{i:04d}_length.npy', mmap_mode='r')
        for s0 in range(0, len(ln), BLOCK):
            s1 = min(s0 + BLOCK, len(ln))
            blk = np.asarray(seq[s0:s1]).astype(np.float32)
            lv = np.asarray(ln[s0:s1]).astype(np.int64)
            nz = (blk != 0).any(-1)                 # B, L  real-step mask
            want = cols[None, :] >= (L - lv)[:, None]
            bad_pad += int(nz[~want].sum())         # padding that is not zero
            bad_len += int((~nz[want]).sum())       # real step that is zero
            checked += s1 - s0
    c(f'{tag}: 패딩 구간(앞 L-length)이 전부 0', bad_pad == 0,
      f'위반 {bad_pad}' if bad_pad else f'left-padding 확인 ({checked:,}행 전수)')
    c(f'{tag}: 뒤 length 스텝이 전부 실제 투구', bad_len == 0,
      f'위반 {bad_len}' if bad_len else f'length == 실제 스텝 수 ({checked:,}행 전수)')


def check_season(c, S, L):
    print(f'\n--- season {S}, L={L} ---')
    p = paths.SEQ / f'manifest_S{S}_L{L}.json'
    if not c(f'manifest_S{S}_L{L}.json 존재', p.exists(), str(p)):
        return
    man = json.loads(p.read_text())
    c(f'manifest L == {L}', man['sequence_length'] == L)
    for split in ('train', 'valid'):
        check_split(c, man, S, L, split)


def sha256(c, path, seasons, L):
    """Write the sums if `path` is new, verify against them if it exists.

    This is what actually proves the tree on Colab is the tree the leakage
    audit passed on the Mac; everything else here proves it is self-consistent.
    """
    import hashlib
    files = []
    for S in seasons:
        m = paths.SEQ / f'manifest_S{S}_L{L}.json'
        if m.exists():
            files.append(m)
        for split in ('train', 'valid'):
            d = paths.SEQ / split / f'S{S}_L{L}'
            if d.is_dir():
                files += sorted(q for q in d.glob('shard_*.npy')
                                if not q.name.startswith('._'))
    sums = {}
    for q in files:
        h = hashlib.sha256()
        with open(q, 'rb') as fh:
            for blk in iter(lambda: fh.read(1 << 22), b''):
                h.update(blk)
        sums[str(q.relative_to(paths.SEQ))] = h.hexdigest()

    if not path.exists():
        path.write_text(''.join(f'{v}  {k}\n' for k, v in sorted(sums.items())))
        print(f'\n  sha256: {len(sums)} 개 파일 -> {path} (업로드해서 '
              f'추출 후 --sha256 로 대조하십시오)')
        return
    ref = {}
    for line in path.read_text().splitlines():
        if line.strip():
            v, k = line.split(None, 1)
            ref[k.strip()] = v
    # the sums file may cover more seasons than this run asked for; comparing
    # against all of it would report every unscanned season as missing.
    scope = tuple(f'S{S}_L{L}' for S in seasons)
    ref = {k: v for k, v in ref.items() if any(s in k for s in scope)}
    if not ref:
        c('sha256: 요청한 시즌이 체크섬 파일에 있음', False,
          f'{path.name} 에 {", ".join(scope)} 항목 없음')
        return
    miss = sorted(set(ref) - set(sums))
    extra = sorted(set(sums) - set(ref))
    diff = sorted(k for k in set(ref) & set(sums) if ref[k] != sums[k])
    c('sha256: 파일 목록 일치', not miss and not extra,
      f'누락 {len(miss)}, 추가 {len(extra)}'
      + (f' {(miss + extra)[:3]}' if miss or extra else ''))
    c('sha256: 내용 일치', not diff,
      f'{len(ref)}개 대조, 불일치 {len(diff)}' + (f' {diff[:3]}' if diff else ''))


def main(argv=None):
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser(prog='check_shards')
    ap.add_argument('--seasons', default='2023,2024')
    ap.add_argument('--sequence-length', type=int, default=32)
    ap.add_argument('--sha256', default='',
                    help='체크섬 파일 경로. 없으면 생성, 있으면 대조')
    a = ap.parse_args(argv)
    seasons = [int(s) for s in a.seasons.split(',')]

    print(f'shard root  {paths.SEQ}')
    c = Check()
    for S in seasons:
        check_season(c, S, a.sequence_length)
    if a.sha256:
        sha256(c, Path(a.sha256), seasons, a.sequence_length)

    print('\n' + ('shard check PASSED' if not c.fail
                  else f'shard check FAILED ({c.fail} checks)'))
    print('참고: 시즌 격리 / row_id 중복 / window-parquet 비트 일치 / scaler 출처 /'
          '\n      p_v9-OOF 정렬은 fold parquet 이 필요한 Mac 단계입니다 '
          '(audit.py, 업로드 전 실행).')
    return 1 if c.fail else 0


if __name__ == '__main__':
    sys.exit(main())
