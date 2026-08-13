"""Stage 0: is BCAT's 2023 top-bin collapse a one-off or a structural risk?

READ-ONLY. No training, no model files opened, no submission touched. Every
number comes from the existing walk-forward OOF parquets that v9's own
backtest produced.

WHAT IS BEING DECIDED
---------------------
On the 2023 fold p_Bcat's top two prediction bins carry 95% of a Reliability
of 1777.6 -- predicting 0.65-0.71 where the actual rate is the base rate. The
same model is clean on 2024 (Reliability 28.2). BCAT holds weight 0.45, the
heaviest component of v9, so the question that matters before submitting is
not "can this be improved" but "can it recur in 2025".

  A. the 2024 fold shows no such profile  -> 2023 was a one-off, stop
  B. the profile is present in 2024 too   -> structural, report and design

This is not a scoring experiment. The simulations in part 3 measure what a
defensive clip would have done; none of them is proposed for the submission.

    python experiments/bcat_next/stage0/stage0_diagnose.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from src.deep_learning_state.metrics import decomp, paired  # noqa: E402

HERE = Path(__file__).resolve().parent
W = 0.25, 0.30, 0.45          # v9 component weights, fixed


def load(season):
    o = pd.read_parquet(REPO / 'work' / 'research' / f'oof_{season}.parquet')
    a9 = REPO / 'work' / 'strat' / f'v9_a8_{season}.npy'
    if a9.exists():
        o['p_A9'] = np.load(a9)
    o['p_v9'] = W[0] * o.p_A7 + W[1] * o.p_A9 + W[2] * o.p_Bcat
    return o


def rel_table(y, p, nb=20):
    q = pd.qcut(p, nb, labels=False, duplicates='drop')
    g = pd.DataFrame({'p': p, 'y': y, 'q': q}).groupby('q').agg(
        n=('y', 'size'), pred=('p', 'mean'), actual=('y', 'mean'))
    unc = y.mean() * (1 - y.mean())
    g['gap'] = g.actual - g.pred
    g['rel'] = (g.n / len(y)) * (g.pred - g.actual) ** 2 * 100000 / unc
    return g


def concentration(d, col, k=10):
    c = d[col].value_counts()
    return dict(n_unique=int(c.size),
                top10_share=float(c.head(k).sum() / len(d)),
                max_share=float(c.iloc[0] / len(d)))


def profile(tag, d, whole, rows):
    """One row of diagnostics.csv describing a subset."""
    y = d.control_success.to_numpy(np.float64)
    r = dict(subset=tag, n=len(d), frac=len(d) / len(whole),
             p_bcat_mean=float(d.p_Bcat.mean()),
             p_bcat_min=float(d.p_Bcat.min()),
             p_bcat_max=float(d.p_Bcat.max()),
             actual=float(y.mean()),
             gap=float(y.mean() - d.p_Bcat.mean()),
             p_A7_mean=float(d.p_A7.mean()), p_A9_mean=float(d.p_A9.mean()),
             A7_gap=float(y.mean() - d.p_A7.mean()),
             A9_gap=float(y.mean() - d.p_A9.mean()),
             asof_pitcher_n_med=float(d.asof_pitcher_n.median()),
             asof_batter_n_med=float(d.asof_batter_n.median()),
             asof_p_succ_med=float(d.asof_pitcher_success_rate.median()))
    for col, nm in (('pitcher_id', 'pit'), ('batter_id', 'bat')):
        c = concentration(d, col)
        r[f'{nm}_unique'] = c['n_unique']
        r[f'{nm}_top10_share'] = c['top10_share']
    rows.append(r)
    return r


def show(r):
    print(f"  {r['subset']:<26s} n {r['n']:>7,} ({r['frac']*100:5.1f}%)  "
          f"BCAT {r['p_bcat_mean']:.4f} [{r['p_bcat_min']:.3f},{r['p_bcat_max']:.3f}]  "
          f"actual {r['actual']:.4f}  gap {r['gap']:+.4f}")
    print(f"  {'':26s}   A7 {r['p_A7_mean']:.4f} (gap {r['A7_gap']:+.4f})   "
          f"A9 {r['p_A9_mean']:.4f} (gap {r['A9_gap']:+.4f})")
    print(f"  {'':26s}   투수 {r['pit_unique']:,}명 (상위10 {r['pit_top10_share']*100:.1f}%)  "
          f"타자 {r['bat_unique']:,}명 (상위10 {r['bat_top10_share']*100:.1f}%)  "
          f"asof_p_n 중앙 {r['asof_pitcher_n_med']:,.0f}")


def main():
    rows = []
    d = {s: load(s) for s in (2023, 2024)}

    # ---------------- 1. 2023 collapse profile --------------------------
    print('=' * 100)
    print('  1. BCAT 2023 collapse profile')
    print('=' * 100)
    g23 = rel_table(d[2023].control_success.to_numpy(np.float64),
                    d[2023].p_Bcat.to_numpy(np.float64))
    tot = g23.rel.sum()
    print(f'\n  Reliability 총 {tot:.1f}   상위 3 bin 기여 '
          f'{g23.rel.nlargest(3).sum()/tot*100:.1f}%')
    print('   bin      n     pred   actual      gap      Rel')
    for i, x in g23.iterrows():
        mark = '  <--' if x.rel > tot * 0.05 else ''
        print(f'   {int(i):3d} {int(x.n):7,}   {x.pred:.4f}   {x.actual:.4f}  '
              f'{x.gap:+.4f} {x.rel:8.1f}{mark}')

    o = d[2023]
    q = pd.qcut(o.p_Bcat, 20, labels=False, duplicates='drop')
    hi23 = o[q >= 18]
    lo23 = o[q < 18]
    thr = float(hi23.p_Bcat.min())
    print(f'\n  bin18+ 임계 p_Bcat >= {thr:.4f}')
    show(profile('2023 bin18-19 (붕괴)', hi23, o, rows))
    print()
    show(profile('2023 bin0-17 (정상)', lo23, o, rows))

    print('\n  --- 엔티티 집중도: 붕괴 구간이 특정 선수에 몰려 있는가 ---')
    for col, nm in (('pitcher_id', '투수'), ('batter_id', '타자')):
        a = concentration(hi23, col)
        b = concentration(o, col)
        print(f'    {nm}  붕괴구간 {a["n_unique"]:,}명 / 전체 {b["n_unique"]:,}명   '
              f'최다 1명 점유 {a["max_share"]*100:.2f}% (전체 {b["max_share"]*100:.2f}%)')

    print('\n  --- 같은 행에서 A7/A9 는 정상인가 ---')
    yh = hi23.control_success.to_numpy(np.float64)
    for c in ('p_Bcat', 'p_A7', 'p_A9'):
        v = hi23[c].to_numpy(np.float64)
        print(f'    {c:8s} mean {v.mean():.4f}  actual {yh.mean():.4f}  '
              f'gap {yh.mean()-v.mean():+.4f}')

    # ---------------- 2. 2024 reproduction ------------------------------
    print('\n' + '=' * 100)
    print('  2. 2024 에서 같은 profile 이 재현되는가')
    print('=' * 100)
    o4 = d[2024]
    g24 = rel_table(o4.control_success.to_numpy(np.float64),
                    o4.p_Bcat.to_numpy(np.float64))
    print(f'\n  2024 Reliability 총 {g24.rel.sum():.1f}   '
          f'상위 3 bin 기여 {g24.rel.nlargest(3).sum()/g24.rel.sum()*100:.1f}%')
    print(f'  2024 p_Bcat 최대 {o4.p_Bcat.max():.4f}   '
          f'95분위 {o4.p_Bcat.quantile(.95):.4f}   '
          f'(2023: 최대 {o.p_Bcat.max():.4f}, 95분위 {o.p_Bcat.quantile(.95):.4f})')

    print(f'\n  [A] 2023 과 동일한 절대 임계 (p_Bcat >= {thr:.4f}) 를 2024 에 적용')
    hi24_abs = o4[o4.p_Bcat >= thr]
    if len(hi24_abs):
        show(profile(f'2024 p>={thr:.3f} (절대임계)', hi24_abs, o4, rows))
    else:
        print(f'    해당 행 0개 — 2024 BCAT 는 {thr:.4f} 이상을 예측하지 않음')
        rows.append(dict(subset=f'2024 p>={thr:.3f} (절대임계)', n=0, frac=0.0))

    print(f'\n  [B] 같은 분위 위치 (상위 10%, bin18-19) 를 2024 에 적용')
    q4 = pd.qcut(o4.p_Bcat, 20, labels=False, duplicates='drop')
    show(profile('2024 bin18-19 (분위)', o4[q4 >= 18], o4, rows))

    print(f'\n  [C] 2023 붕괴 구간과 같은 엔티티 프로파일이 2024 에 있는가')
    hip = set(hi23.pitcher_id.unique())
    m = o4.pitcher_id.isin(hip)
    if m.sum():
        show(profile('2024 (2023붕괴 투수 동일)', o4[m], o4, rows))

    # ---------------- 3. defensive simulation ---------------------------
    print('\n' + '=' * 100)
    print('  3. 방어선 시뮬레이션 (모델 변경 없음 · 제출 적용 금지)')
    print('=' * 100)
    prior = 0.5
    sims = ([('clip', c) for c in (0.60, 0.62, 0.65, 0.70)]
            + [('shrink', s) for s in (0.9, 0.8, 0.7)]
            + [('cap_dev', c) for c in (0.08, 0.10, 0.12)])
    for season in (2023, 2024):
        oo = d[season]
        y = oo.control_success.to_numpy(np.float64)
        b = oo.p_Bcat.to_numpy(np.float64)
        base_b = decomp(y, b)
        v9 = np.clip(oo.p_v9.to_numpy(np.float64), 1e-3, 1 - 1e-3)
        base_v = decomp(y, v9)
        print(f'\n  --- {season}   BCAT BSS {base_b[2]:.1f} (Rel {base_b[1]:.1f})   '
              f'v9 BSS {base_v[2]:.1f} ---')
        print('    방법            BCAT BSS   BCAT Rel  |  v9 BSS    dBSS   paired')
        for kind, par in sims:
            if kind == 'clip':
                bb = np.minimum(b, par)
                lab = f'clip <= {par:.2f}'
            elif kind == 'shrink':
                bb = prior + par * (b - prior)
                lab = f'shrink x{par:.1f}'
            else:
                bb = np.clip(b, prior - par, prior + par)
                lab = f'cap |p-.5|<={par:.2f}'
            bb = np.clip(bb, 1e-3, 1 - 1e-3)
            rb = decomp(y, bb)
            nv = np.clip(W[0]*oo.p_A7 + W[1]*oo.p_A9 + W[2]*bb, 1e-3, 1-1e-3)
            rv = decomp(y, nv)
            m, se = paired(y, v9, nv)
            rows.append(dict(subset=f'{season} sim {lab}', n=len(oo),
                             bcat_bss=rb[2], bcat_rel=rb[1], v9_bss=rv[2],
                             v9_dbss=rv[2]-base_v[2], paired=m, paired_se=se))
            print(f'    {lab:<16s} {rb[2]:8.1f}   {rb[1]:8.1f}  | {rv[2]:8.1f} '
                  f'{rv[2]-base_v[2]:+8.1f}  {m:+7.1f}±{se:.1f}')

    pd.DataFrame(rows).to_csv(HERE / 'diagnostics.csv', index=False)
    print(f'\n  wrote {HERE}/diagnostics.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
