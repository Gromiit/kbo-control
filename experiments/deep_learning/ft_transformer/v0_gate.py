"""V0 gate: does FT-Transformer leave the round-17 frontier?

Runs anywhere the OOF parquets and work/research/oof_2024.parquet are present.

THE FRONTIER
------------
Round 17's six MLP variants on the 2024 fold fall on

    BSS = 2052 * corr(v9) - 1051        R^2 = 0.998

accuracy and decorrelation trading one-for-one. The gate wants BSS >= 750 at
corr < 0.85; the line predicts 693.2 there, so clearing it means sitting about
57 BSS ABOVE the line. Distance from the line is therefore reported alongside
the raw numbers -- a model can look mediocre and still be interesting if it is
off the line, and can look decent while being exactly on it.

GATE (as specified)
  1. BSS < 750            -> stop
  2. corr(v9) > 0.95      -> stop
  3. blend improvement < +30 -> stop

Blend improvement is measured two ways and both are printed: the in-sample
logit-stacking ceiling on 2024 (generous, not deployable) and a fixed-weight
sweep. The ceiling is the fair test of "is there room"; the sweep says what a
pre-chosen weight would actually have delivered.

    python v0_gate.py --seed 42
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from src.deep_learning_state.metrics import decomp, paired  # noqa: E402

SLOPE, INTERCEPT = 2052.0, -1051.0     # round-17 frontier, R^2 0.998
BASE3_CEILING = 19.4                   # 3-component stacking ceiling, 2024
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sg = lambda z: 1.0 / (1.0 + np.exp(-z))


def incumbents(data_dir):
    """v9's components on the 2024 fold.

    Read from the prepared data parquet first, which already carries p_A7 /
    p_A9 / p_Bcat / p_v9. That is what lets the gate run on Colab, where work/
    is deliberately absent -- the experiment must not reach into it. The
    work/research copy is used only as a fallback when running on the Mac
    without the prepared archive.
    """
    local = Path(data_dir) / 'ftt_2024_va.parquet'
    if local.exists():
        o = pd.read_parquet(local, columns=['row_id', 'control_success',
                                            'p_A7', 'p_A9', 'p_Bcat'])
        src = f'{local.name} (work/ 미사용)'
    else:
        f = REPO / 'work' / 'research' / 'oof_2024.parquet'
        if not f.exists():
            raise SystemExit(
                f'v9 구성요소를 찾을 수 없습니다.\n'
                f'  기대 위치 1: {local}   (prep_data.py 산출물)\n'
                f'  기대 위치 2: {f}')
        o = pd.read_parquet(f, columns=['row_id', 'control_success',
                                        'p_A7', 'p_A9', 'p_Bcat'])
        a9 = REPO / 'work' / 'strat' / 'v9_a8_2024.npy'
        if a9.exists():
            o['p_A9'] = np.load(a9)
        src = 'work/research/oof_2024.parquet'
    if o[['p_A7', 'p_A9', 'p_Bcat']].isna().any().any():
        raise SystemExit('v9 구성요소에 결측이 있습니다')
    o['p_v9'] = np.clip(0.25 * o.p_A7 + 0.30 * o.p_A9 + 0.45 * o.p_Bcat,
                        1e-3, 1 - 1e-3)
    return o, src


def fit_stack(y, cols):
    X = np.column_stack([np.ones(len(y))] + [lg(c) for c in cols])
    b = np.zeros(X.shape[1])
    for _ in range(80):
        q = sg(X @ b)
        g = X.T @ (q - y)
        Wt = np.clip(q * (1 - q), 1e-12, None)
        H = X.T @ (X * Wt[:, None]) + 1e-8 * np.eye(X.shape[1])
        st = np.linalg.solve(H, g)
        b -= st
        if np.abs(st).max() < 1e-11:
            break
    return b, sg(X @ b)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='v0_gate')
    ap.add_argument('--oof-dir', default=str(HERE / 'oof'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--models', default='A,B,C')
    ap.add_argument('--data-dir', default=str(HERE / 'data'))
    a = ap.parse_args(argv)

    inc, src = incumbents(a.data_dir)
    y = inc.control_success.to_numpy(np.float64)
    comp = {c: inc[c].to_numpy(np.float64) for c in ('p_A7', 'p_A9', 'p_Bcat')}
    v9 = inc.p_v9.to_numpy(np.float64)
    b_res, b_rel, b_bss = decomp(y, v9)
    b3, p3 = fit_stack(y, [comp[c] for c in ('p_A7', 'p_A9', 'p_Bcat')])
    c3 = decomp(y, p3)[2] - b_bss

    print('=' * 100)
    print(f'  FT-Transformer V0 gate   fold 2024  seed {a.seed}  n {len(inc):,}')
    print(f'  v9 구성요소 출처  {src}')
    print(f'  v9 blend   BSS {b_bss:.1f}  Res {b_res:.1f}  Rel {b_rel:.1f}')
    print(f'  참조 단독  A7 {decomp(y, comp["p_A7"])[2]:.1f}   '
          f'A9 {decomp(y, comp["p_A9"])[2]:.1f}   '
          f'BCAT {decomp(y, comp["p_Bcat"])[2]:.1f}')
    print(f'  3요소 stacking 상한 (in-sample) {c3:+.1f}')
    print(f'  frontier  BSS = {SLOPE:.0f}*corr {INTERCEPT:+.0f}   '
          f'(corr 0.85 -> BSS {SLOPE*0.85+INTERCEPT:.1f}, 게이트 750)')
    print('=' * 100)

    rows = []
    for m in a.models.split(','):
        f = Path(a.oof_dir) / f'ftt_{m}_s{a.seed}.parquet'
        if not f.exists():
            print(f'\n  model {m}: {f.name} 없음 — 건너뜀')
            continue
        d = pd.read_parquet(f)
        d = inc[['row_id']].merge(d, on='row_id', how='left')
        if d.p.isna().any():
            raise SystemExit(f'{f.name}: row_id 정렬 실패')
        p = np.clip(d.p.to_numpy(np.float64), 1e-6, 1 - 1e-6)

        res, rel, bss = decomp(y, p)
        cors = {k: float(np.corrcoef(lg(p), lg(v))[0, 1]) for k, v in comp.items()}
        cor_v9 = float(np.corrcoef(lg(p), lg(v9))[0, 1])
        bias = float(p.mean() - y.mean())
        pred_line = SLOPE * cor_v9 + INTERCEPT
        off_line = bss - pred_line

        b4, p4 = fit_stack(y, [comp[c] for c in ('p_A7', 'p_A9', 'p_Bcat')] + [p])
        c4 = decomp(y, p4)[2] - b_bss
        gain = c4 - c3
        mm, se = paired(y, p3, p4)

        print(f'\n  ===== model {m} =====')
        print(f'    BSS {bss:8.1f}   Resolution {res:8.1f}   Reliability {rel:7.1f}')
        print(f'    calibration bias (p - y)  {bias:+.5f}')
        print(f'    corr(v9) {cor_v9:.4f}   |  A7 {cors["p_A7"]:.4f}  '
              f'A9 {cors["p_A9"]:.4f}  BCAT {cors["p_Bcat"]:.4f}')
        print(f'    frontier 예측 BSS {pred_line:7.1f}  ->  이탈 {off_line:+7.1f}'
              f'   (게이트 진입에 +57 필요)')
        print(f'    4요소 stacking 상한 {c4:+.1f}   3요소 대비 {gain:+.1f}   '
              f'paired {mm:+.1f}±{se:.1f}')
        print(f'    고정가중 blend  ', end='')
        best_w, best_d = 0.0, 0.0
        for w in (0.05, 0.10, 0.15, 0.20, 0.30):
            pb = sg((1 - w) * lg(v9) + w * lg(p))
            dd = decomp(y, pb)[2] - b_bss
            if dd > best_d:
                best_w, best_d = w, dd
            print(f'w={w:.2f} {dd:+7.1f}  ', end='')
        print()

        g1 = bss >= 750
        g2 = cor_v9 <= 0.95
        g3 = gain >= 30.0
        print(f'    GATE  1) BSS>=750 {"PASS" if g1 else "FAIL"}   '
              f'2) corr<=0.95 {"PASS" if g2 else "FAIL"}   '
              f'3) blend>=+30 {"PASS" if g3 else "FAIL"}   '
              f'-> {"PASS" if (g1 and g2 and g3) else "FAIL"}')
        rows.append(dict(model=m, seed=a.seed, BSS=bss, Resolution=res,
                         Reliability=rel, calib_bias=bias, corr_v9=cor_v9,
                         **{f'corr_{k}': v for k, v in cors.items()},
                         frontier_pred=pred_line, off_frontier=off_line,
                         ceil_4=c4, ceil_3=c3, ceil_gain=gain,
                         paired=mm, paired_se=se,
                         best_fixed_w=best_w, best_fixed_dbss=best_d,
                         gate_bss=g1, gate_corr=g2, gate_blend=g3,
                         gate_pass=bool(g1 and g2 and g3)))

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(HERE / f'v0_results_s{a.seed}.csv', index=False)
        print('\n' + '=' * 100)
        ok = bool(df.gate_pass.any())
        print(f'  V0 종합: {"PASS" if ok else "FAIL"}')
        if not ok:
            print('  FAIL -> seed 3개 확장 없음. 추가 실험 없음. v9 유지.')
        print('=' * 100)
        print(f'  wrote {HERE}/v0_results_s{a.seed}.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
