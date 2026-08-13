"""G0: per-pitcher Trackman profile tables. Pure aggregation, no learning.

WHAT THIS IS
------------
A frozen lookup table, one row per pitcher, summarising how that pitcher threw
in seasons STRICTLY BEFORE the target season. At inference a row reads its own
pitcher's entry and nothing else -- the same shape as v9's carry/split/role/
lowrank tables, which the submission audit already showed satisfies the
row-independence rule.

RULES THIS RESPECTS (organiser text, data_description.md)
---------------------------------------------------------
  s3  "참가자는 이 파일을 이용해 과거 투구 특성, 구종 특성, 투수 단위 요약값 등
       추가 피처를 만들 수 있습니다"          -> this is exactly that
  s5  each evaluation row predicted independently -> per-pitcher lookup, no
      test row is ever read here
  s6  2025 Trackman forbidden, current pitch's Trackman forbidden
      -> --max-season is a hard gate, asserted, and the current pitch never
         enters because the table is keyed by pitcher only

NO TARGET, BY CONSTRUCTION
--------------------------
trackman_history.csv has no `control_success` column at all, so no
target-derived quantity can leak in even by accident. Asserted anyway: the
loader refuses to read any column outside the declared whitelist.

That absence is also this approach's ceiling. The Phase A GRU teacher's
strongest channel was the outcome of each past pitch; here there is none. The
profile can say how a pitcher throws, never how well.

    python -m src.deep_learning_state.pitcher_profile --max-season 2023 \
        --variant basic
    python -m src.deep_learning_state.pitcher_profile --max-season 2023 \
        --variant plus
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths

# the eight physics axes; round 15 measured their oracle value as
# movement +349.6 > velocity +261.0 > spin +91.6 > release +88.7
PHYS = ['rel_speed', 'spin_rate', 'induced_vert_break', 'horz_break',
        'extension', 'rel_height', 'rel_side', 'zone_speed']
MIX = ['fastball', 'breaking', 'offspeed', 'other']
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

# nothing outside this list is read off disk. `control_success` is not a
# column of trackman_history.csv, and this makes that structural.
USECOLS = ['pitcher_trackman_id', 'season', 'pitch_type_group'] + PHYS

PMAP = Path('work/research/trackman/pmap.parquet')   # round-15 artefact, reused


def _load_trackman(raw_csv, max_season, chunksize=400_000):
    """Stream the 337 MB log, dropping seasons > max_season as we go.

    Chunked so peak memory stays near one chunk rather than 1.79M x 30. The
    season filter is applied inside the loop, so a later season is never even
    materialised.
    """
    keep = []
    seen_seasons = set()
    for ch in pd.read_csv(raw_csv, usecols=USECOLS, chunksize=chunksize,
                          encoding='utf-8-sig'):
        seen_seasons |= set(ch.season.unique().tolist())
        keep.append(ch[ch.season <= max_season])
    d = pd.concat(keep, ignore_index=True)
    if len(d) and int(d.season.max()) > max_season:
        raise SystemExit(f'season gate failed: max {d.season.max()} > {max_season}')
    return d, sorted(int(s) for s in seen_seasons)


def _basic(g):
    """mean + sd per axis. This is round 15's LEVEL + SPREAD, learned-free."""
    a = g[PHYS].agg(['mean', 'std'])
    a.columns = [f'tm_{c}_{s}' for c, s in a.columns]
    return a


def _plus(d):
    """basic, plus the shape of each axis and the pitcher's arsenal.

    Takes the frame and groups here -- passing a groupby in would make
    `value_counts` and `size` below awkward, and grouping twice is cheap
    next to the CSV read.

    Quantiles are the point: mean and sd are two moments, and any set encoder
    reading only the marginals can extract little more than the quantiles
    already carry. If this cannot express the information, what is left for a
    sequence encoder is joint or sequential structure, which is a much
    narrower hypothesis than "add an encoder".
    """
    grp = d.groupby('pitcher_trackman_id')
    parts = [_basic(grp)]

    q = grp[PHYS].quantile(QUANTILES).unstack()
    q.columns = [f'tm_{c}_q{int(p*100):02d}' for c, p in q.columns]
    parts.append(q)

    iqr = pd.DataFrame(index=q.index)
    for c in PHYS:
        iqr[f'tm_{c}_iqr'] = q[f'tm_{c}_q75'] - q[f'tm_{c}_q25']
        iqr[f'tm_{c}_p90m10'] = q[f'tm_{c}_q90'] - q[f'tm_{c}_q10']
    parts.append(iqr)

    # (grp already built above)
    mix = (grp['pitch_type_group'].value_counts(normalize=True)
              .unstack(fill_value=0.0))
    for m in MIX:
        if m not in mix.columns:
            mix[m] = 0.0
    mix = mix[MIX]
    mix.columns = [f'tm_mix_{m}' for m in MIX]
    parts.append(mix)

    vol = pd.DataFrame({'tm_log_n': np.log1p(grp.size())})
    parts.append(vol)
    return pd.concat(parts, axis=1)


def build(raw_csv, max_season, variant):
    if not PMAP.exists():
        raise SystemExit(f'need {PMAP} (round-15 pitcher id map). Do not rebuild it.')
    pmap = pd.read_parquet(PMAP)[['pitcher_id', 'pitcher_trackman_id', 'share']]

    print(f'  reading {raw_csv}  (season <= {max_season}, {len(USECOLS)} cols)')
    d, seen = _load_trackman(raw_csv, max_season)
    print(f'  file seasons {seen}  ->  kept {len(d):,} pitches, '
          f'seasons {sorted(int(s) for s in d.season.unique())}')

    g = d.groupby('pitcher_trackman_id')
    prof = _basic(g) if variant == 'basic' else _plus(d)
    prof.index.name = 'pitcher_trackman_id'
    prof = prof.reset_index()

    prof = pmap.merge(prof, on='pitcher_trackman_id', how='inner')
    prof = prof.drop(columns=['pitcher_trackman_id', 'share'])
    prof['tm_has_history'] = 1.0

    feat = [c for c in prof.columns if c != 'pitcher_id']
    bad = [c for c in feat if 'success' in c or 'control' in c or 'target' in c]
    if bad:
        raise SystemExit(f'target-derived column produced: {bad}')

    paths.ensure_dirs()
    out_dir = paths.DATA / 'profiles'
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f'pitcher_profile_le{max_season}_{variant}.parquet'
    prof.to_parquet(out, index=False)

    print(f'  pitchers {len(prof):,}   features {len(feat)}')
    print(f'  wrote {out}  ({out.stat().st_size/1e6:.2f} MB)')
    return out, prof


def main(argv=None):
    ap = argparse.ArgumentParser(prog='pitcher_profile')
    ap.add_argument('--raw', default=str(Path.home() /
                    'Desktop/open/data/trackman_history.csv'))
    ap.add_argument('--max-season', type=int, required=True,
                    help='inclusive upper bound; for fold S use S-1')
    ap.add_argument('--variant', choices=['basic', 'plus'], required=True)
    a = ap.parse_args(argv)
    print('=' * 70)
    print(f'  G0 pitcher profile   variant={a.variant}  season <= {a.max_season}')
    print('=' * 70)
    build(a.raw, a.max_season, a.variant)
    return 0


if __name__ == '__main__':
    sys.exit(main())
