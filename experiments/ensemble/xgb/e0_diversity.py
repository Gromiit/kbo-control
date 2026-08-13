"""E0 gate: is XGBoost different enough from v9's existing components?

WHAT IS BEING DECIDED
---------------------
Not whether XGBoost is good. v9 already averages 26 models, and on 2024 its
three components correlate 0.81-0.96 in logit space, so a fourth only earns a
place by making different mistakes. Two questions:

  1. logit correlation with p_A7 / p_A9 / p_Bcat
  2. whether adding it lifts the 4-model stacking ceiling above the 3-model one

The stacking number is an IN-SAMPLE ceiling on the evaluation season. It is not
a deployable result and makes no temporal claim -- it answers "is there room",
and E2 would be the one to ask "does it transfer". Reported as such.

2023 IS EXCLUDED FROM THE VERDICT
---------------------------------
p_Bcat on the 2023 fold has Resolution 523.1 but Reliability 1777.6, so its BSS
clips to 0 and v9's blend (403.1) scores below A9 alone (843.8). The 2023
correlation structure reflects that breakage rather than model diversity, so it
is computed and recorded for the audit trail but excluded from the gate.

    python experiments/ensemble/xgb/e0_diversity.py
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from src.deep_learning_state.metrics import decomp, paired  # noqa: E402

HERE = Path(__file__).resolve().parent
OOF = HERE / 'oof'
BASE3_CEILING = 19.4          # measured: 3-model logit stacking, 2024 in-sample
GATE_CORR_STRONG = 0.82
GATE_CORR_FAIL = 0.90
GATE_BSS_MIN = 750.0
GATE_CEILING_GAIN = 10.0

lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sg = lambda z: 1.0 / (1.0 + np.exp(-z))


def load_incumbent(fold):
    o = pd.read_parquet(REPO / 'work' / 'research' / f'oof_{fold}.parquet')
    a9 = REPO / 'work' / 'strat' / f'v9_a8_{fold}.npy'
    if a9.exists():
        o['p_A9'] = np.load(a9)
    return o[['row_id', 'control_success', 'p_A7', 'p_A9', 'p_Bcat']]


def load_xgb(fold, arm, seeds):
    """Seed average in raw-score space, the convention v9's own script uses."""
    zs, rid = [], None
    for s in seeds:
        d = pd.read_parquet(OOF / f'xgb_{fold}_s{s}_{arm}.parquet')
        if rid is None:
            rid = d.row_id.to_numpy()
        elif not (rid == d.row_id.to_numpy()).all():
            raise SystemExit(f'row_id mismatch across seeds ({fold}/{arm})')
        zs.append(lg(d.p.to_numpy(np.float64)))
    return rid, sg(np.mean(zs, axis=0)), [sg(z) for z in zs]


def fit_stack(y, cols):
    X = np.column_stack([np.ones(len(y))] + [lg(c) for c in cols])
    b = np.zeros(X.shape[1])
    for _ in range(80):
        q = sg(X @ b)
        g = X.T @ (q - y)
        W = np.clip(q * (1 - q), 1e-12, None)
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        b -= step
        if np.abs(step).max() < 1e-11:
            break
    return b, sg(X @ b)


def analyse(fold, seeds, verdict_fold):
    inc = load_incumbent(fold)
    y = inc.control_success.to_numpy(np.float64)
    comp = {c: inc[c].to_numpy(np.float64) for c in ('p_A7', 'p_A9', 'p_Bcat')}
    v9 = np.clip(0.25 * comp['p_A7'] + 0.30 * comp['p_A9']
                 + 0.45 * comp['p_Bcat'], 1e-3, 1 - 1e-3)

    tag = '' if verdict_fold else '   [판정 제외 — BCAT calibration 붕괴]'
    print(f'\n{"="*92}\n  fold {fold}{tag}\n{"="*92}')
    r = decomp(y, v9)
    print(f'  v9 blend        BSS {r[2]:8.1f}  Res {r[0]:8.1f}  Rel {r[1]:7.1f}')
    for c, p in comp.items():
        rr = decomp(y, p)
        print(f'  {c:14s}  BSS {rr[2]:8.1f}  Res {rr[0]:8.1f}  Rel {rr[1]:7.1f}')

    rows, out = [], {}
    for arm in ('base', 'noinit'):
        rid, px, per_seed = load_xgb(fold, arm, seeds)
        if not (rid == inc.row_id.to_numpy()).all():
            order = pd.Series(range(len(rid)), index=rid)
            px = px[order.reindex(inc.row_id.to_numpy()).to_numpy()]
        rr = decomp(y, px)
        sd = np.std([decomp(y, p)[2] for p in per_seed], ddof=1)
        cors = {c: float(np.corrcoef(lg(px), lg(p))[0, 1]) for c, p in comp.items()}
        mean_c = float(np.mean(list(cors.values())))
        print(f'\n  xgb_{arm:6s} (seed 평균)  BSS {rr[2]:8.1f}  Res {rr[0]:8.1f}  '
              f'Rel {rr[1]:7.1f}   seed별 BSS sd {sd:.1f}')
        print(f'    상관  A7 {cors["p_A7"]:.4f}   A9 {cors["p_A9"]:.4f}   '
              f'BCAT {cors["p_Bcat"]:.4f}   ==> 평균 {mean_c:.4f}')
        rows.append(dict(fold=fold, arm=arm, BSS=rr[2], Resolution=rr[0],
                         Reliability=rr[1], seed_bss_sd=sd, mean_corr=mean_c,
                         **{f'corr_{k}': v for k, v in cors.items()}))
        out[arm] = px

    # incumbent pairwise, for reference
    C = ['p_A7', 'p_A9', 'p_Bcat']
    M = np.column_stack([lg(comp[c]) for c in C])
    R = np.corrcoef(M.T)
    iu = np.triu_indices(3, 1)
    print(f'\n  [참조] 기존 3요소 상호 상관 평균 {R[iu].mean():.4f}   '
          f'BCAT {np.mean([R[2,0],R[2,1]]):.4f}  A7 {np.mean([R[0,1],R[0,2]]):.4f}  '
          f'A9 {np.mean([R[1,0],R[1,2]]):.4f}')

    # stacking ceilings (in-sample on this fold)
    print(f'\n  stacking 상한 (이 fold in-sample — 배포 주장 아님)')
    b3, p3 = fit_stack(y, [comp[c] for c in C])
    base_bss = decomp(y, v9)[2]
    c3 = decomp(y, p3)[2] - base_bss
    print(f'    3요소            dBSS {c3:+8.1f}   계수 {np.round(b3,3)}')
    ceil = {}
    for arm in ('base', 'noinit'):
        b4, p4 = fit_stack(y, [comp[c] for c in C] + [out[arm]])
        c4 = decomp(y, p4)[2] - base_bss
        m, se = paired(y, p3, p4)
        ceil[arm] = c4
        print(f'    4요소 (+{arm:6s}) dBSS {c4:+8.1f}   개선 {c4-c3:+7.1f}   '
              f'paired vs 3요소 {m:+.1f}±{se:.1f}   계수 {np.round(b4,3)}')
    b5, p5 = fit_stack(y, [comp[c] for c in C] + [out['base'], out['noinit']])
    c5 = decomp(y, p5)[2] - base_bss
    print(f'    5요소 (양 arm)   dBSS {c5:+8.1f}   개선 {c5-c3:+7.1f}')
    return rows, dict(c3=c3, ceil=ceil, c5=c5)


def main():
    seeds = [42, 43, 44]
    if not (OOF / 'manifest.json').exists():
        raise SystemExit('OOF 없음 — build_xgb_oof.py 를 먼저 실행하십시오')
    all_rows, stacks = [], {}
    for fold, verdict in ((2023, False), (2024, True)):
        rows, st = analyse(fold, seeds, verdict)
        all_rows += rows
        stacks[fold] = st

    df = pd.DataFrame(all_rows)
    df.to_csv(HERE / 'correlation_matrix.csv', index=False)
    (HERE / 'stacking_upper_bound.json').write_text(
        json.dumps(stacks, indent=1, default=float))

    print('\n' + '=' * 92)
    print('  GATE (2024 만으로 판정)')
    print('=' * 92)
    v = df[df.fold == 2024]
    verdicts = {}
    for _, r in v.iterrows():
        mc, bss = r.mean_corr, r.BSS
        gain = stacks[2024]['ceil'][r.arm] - stacks[2024]['c3']
        if mc >= GATE_CORR_FAIL:
            band, ok = f'>= {GATE_CORR_FAIL} 중복', False
        elif mc <= GATE_CORR_STRONG:
            band, ok = f'<= {GATE_CORR_STRONG} strong diversity', gain >= GATE_CEILING_GAIN
        else:
            band, ok = f'{GATE_CORR_STRONG}~{GATE_CORR_FAIL} 보류 -> 상한으로 판정', \
                gain >= GATE_CEILING_GAIN
        bss_ok = bss >= GATE_BSS_MIN
        print(f'\n  {r.arm}')
        print(f'    평균 상관        {mc:.4f}   [{band}]')
        print(f'    단독 BSS         {bss:.1f}   (>= {GATE_BSS_MIN:.0f}: '
              f'{"PASS" if bss_ok else "FAIL"})')
        print(f'    stacking 상한 개선 {gain:+.1f}   (>= +{GATE_CEILING_GAIN:.0f}: '
              f'{"PASS" if gain >= GATE_CEILING_GAIN else "FAIL"})')
        final = ok and (bss_ok or r.arm == 'noinit')
        verdicts[r.arm] = final
        print(f'    -> {"PASS" if final else "FAIL"}')
    overall = any(verdicts.values())
    print(f'\n  E0 종합: {"PASS" if overall else "FAIL"}')
    if not overall:
        print('  FAIL -> E1/E2 구현하지 않음. 앙상블 라인 종료, v9 유지.')
    print('=' * 92)
    print(f'  wrote {HERE}/correlation_matrix.csv, stacking_upper_bound.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
