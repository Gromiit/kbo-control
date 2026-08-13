"""[Mac] Slim the fold parquets down to what FT-Transformer V0 needs, then tar.

Training happens on Colab GPU, so this trims 392 MB of fold/OOF files to only
the columns the run touches and packs them for upload. test.csv is never
opened here or downstream.

WHAT GOES IN
------------
  101 numeric  v9's numeric_feats, imported from the round-17 builder
  4 categorical pitcher_team_id, batter_team_id, count_state, base_state_c
                -- the same four CatBoost uses in b_cat_features
  init         level model log-odds, the offset base for models A and B
  p_v9         only where the OOF exists, which is 2022-2024
  control_success, row_id, season

`count_state` is not stored in the fold parquet; v9 derives it as
balls*3 + strikes inside b_build_features, and it is derived the same way here
rather than invented.

MODEL C's CONSTRAINT, MADE EXPLICIT
-----------------------------------
Model C offsets from logit(p_v9), so its training rows must have p_v9. The OOF
covers 2022-2024 only, so C trains on 492,997 rows against A and B's
1,221,585 -- 40%. That asymmetry is not a bug to fix, it is a property of the
data, and the report has to carry it when comparing C against A and B.

    python experiments/deep_learning/ft_transformer/prep_data.py
"""
import json
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from src.deep_learning_state.dataset import feature_cols  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / 'data'
# The four CatBoost treats as categorical. Three of them -- pitcher_team_id,
# batter_team_id, base_state_c -- are ALSO inside v9's 101 numeric features, so
# model B does not gain new columns from them; what it gains is embedding
# treatment instead of a raw numeric. They are emitted under cat_* names so the
# numeric and categorical blocks can both hold them without colliding.
CAT_SRC = ['pitcher_team_id', 'batter_team_id', 'count_state', 'base_state_c']
CATS = [f'cat_{c}' for c in CAT_SRC]
KEEP = ['row_id', 'season', 'control_success', 'init']


def add_count_state(d):
    """v9's own definition (b_build_features): balls*3 + strikes."""
    b = pd.to_numeric(d.balls_before, errors='coerce').fillna(0).clip(0, 3)
    s = pd.to_numeric(d.strikes_before, errors='coerce').fillna(0).clip(0, 2)
    return (b * 3 + s).astype(np.int16)


def oof_p_v9(season):
    p = REPO / 'work' / 'research' / f'oof_{season}.parquet'
    if not p.exists():
        return None
    o = pd.read_parquet(p, columns=['row_id', 'p_A7', 'p_A9', 'p_Bcat'])
    # data.py overrides p_A9 from v9_a8_{S}.npy ONLY when that file exists and
    # otherwise keeps the p_A9 already stored in the OOF. Overwriting it with
    # NaN instead would silently drop 2022 -- which has no v9_a8 file -- and
    # cut model C's training rows from 492,997 to 245,525.
    a9 = REPO / 'work' / 'strat' / f'v9_a8_{season}.npy'
    if a9.exists():
        o['p_A9'] = np.load(a9)
    o['p_v9'] = 0.25 * o.p_A7 + 0.30 * o.p_A9 + 0.45 * o.p_Bcat
    return o[['row_id', 'p_v9', 'p_A7', 'p_A9', 'p_Bcat']]


def slim(fold, kind, feats):
    src = REPO / 'data' / 'folds' / f'fold_{fold}_{kind}.parquet'
    need = list(dict.fromkeys(
        KEEP + feats + ['pitcher_team_id', 'batter_team_id', 'base_state_c',
                        'balls_before', 'strikes_before']))
    d = pd.read_parquet(src, columns=[c for c in need if c])
    d['count_state'] = add_count_state(d)
    for c in CAT_SRC:
        d[f'cat_{c}'] = pd.to_numeric(d[c], errors='coerce').fillna(-1).astype(np.int16)

    # p_v9 where the OOF has it; model C is limited to those rows
    parts = []
    for s in sorted(d.season.unique()):
        o = oof_p_v9(int(s))
        if o is not None:
            parts.append(o)
    if parts:
        o = pd.concat(parts, ignore_index=True)
        n0 = len(d)
        d = d.merge(o, on='row_id', how='left')
        if len(d) != n0:
            raise SystemExit(f'p_v9 join changed row count: {n0} -> {len(d)}')
    else:
        for c in ('p_v9', 'p_A7', 'p_A9', 'p_Bcat'):
            d[c] = np.nan

    cols = list(dict.fromkeys(
        KEEP + feats + CATS + ['p_v9', 'p_A7', 'p_A9', 'p_Bcat']))
    d = d[[c for c in cols if c in d.columns]]
    for c in feats:
        d[c] = d[c].astype(np.float32)
    d['init'] = d.init.astype(np.float32)
    d['control_success'] = d.control_success.astype(np.int8)

    out = OUT / f'ftt_{fold}_{kind}.parquet'
    d.to_parquet(out, index=False, compression='zstd')
    have = int(d.p_v9.notna().sum()) if 'p_v9' in d else 0
    print(f'  {out.name:22s} {len(d):>9,} 행  {len(d.columns):>3d} 컬럼  '
          f'{out.stat().st_size/1e6:6.1f} MB   p_v9 있는 행 {have:,} '
          f'({have/len(d)*100:.0f}%)')
    return out


def schema(made, feats):
    """What the Colab side needs to know without re-deriving anything."""
    import hashlib
    out = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'),
               fold=2024, numeric_features=feats, n_numeric=len(feats),
               categorical_features=CATS, categorical_source=CAT_SRC,
               meta_columns=KEEP + ['p_v9', 'p_A7', 'p_A9', 'p_Bcat'],
               note=('3 of the 4 categoricals are also inside the numeric '
                     'block; model B gains embedding treatment, not new '
                     'columns. count_state is derived as balls*3+strikes, '
                     "v9's own definition."),
               files={})
    for f in made:
        d = pd.read_parquet(f, columns=['season', 'control_success', 'p_v9'])
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        out['files'][f.name] = dict(
            rows=len(d), bytes=f.stat().st_size, sha256=h,
            seasons=sorted(int(x) for x in d.season.unique()),
            y_mean=float(d.control_success.mean()),
            p_v9_rows=int(d.p_v9.notna().sum()))
    return out


FORBIDDEN = ('test.csv', 'work/', '.pkl', '.cbm', '.zip', 'submit')


def check_payload(names):
    """The tarball travels to Drive, so verify what is in it rather than
    trusting that the right things were added."""
    bad = [n for n in names if any(f in n for f in FORBIDDEN)]
    if bad:
        raise SystemExit(f'tarball 에 금지 항목: {bad}')
    print(f'  payload 검사: {len(names)}개, 금지 항목 0개 '
          f'(test.csv / work / .pkl / .cbm / .zip / submit)')


def main():
    feats = list(feature_cols())
    OUT.mkdir(parents=True, exist_ok=True)
    print('=' * 88)
    print(f'  FT-Transformer V0 데이터 준비   numeric {len(feats)} + cat {len(CATS)}')
    print('  test.csv 미사용 · work/ 미포함 · 모델 파일 미포함')
    print('=' * 88)
    made = [slim(2024, k, feats) for k in ('tr', 'va')]

    sc = schema(made, feats)
    sp = OUT / 'schema.json'
    sp.write_text(json.dumps(sc, indent=1))
    print(f'\n  schema -> {sp.name}  '
          f'(numeric {sc["n_numeric"]}, cat {len(CATS)}, 파일 {len(sc["files"])})')

    members = [f'data/{p.name}' for p in made] + ['data/schema.json']
    check_payload(members)

    tar = HERE / 'ftt_data.tgz'
    if tar.exists():
        tar.unlink()
    with tarfile.open(tar, 'w:gz') as t:
        for p in made + [sp]:
            t.add(p, arcname=f'data/{p.name}')
    with tarfile.open(tar) as t:
        got = [m.name for m in t.getmembers() if m.isfile()]
    check_payload(got)
    print(f'  tar -> {tar.name}  ({tar.stat().st_size/1e6:.1f} MB)  멤버 {got}')
    print(f'\n  다음: {tar.name} 을 Drive 의 MyDrive/kbo/ 에 업로드')
    print('        Colab 에서는 README_COLAB.md 참조')
    return 0


if __name__ == '__main__':
    sys.exit(main())
