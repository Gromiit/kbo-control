"""Where does v9 systematically fail? Diagnostic only.

READ-ONLY. No training, no new features, no submission artefact. Everything
comes from the walk-forward OOF that v9's own backtest already produced.

WHAT "ERROR" MEANS HERE
-----------------------
The target is binary with a base rate near 0.486 and v9 predicts inside
[0.32, 0.66] with AUC ~0.556. Per-row squared error is therefore dominated by
the coin flip and says almost nothing. The quantity that can actually be acted
on is SYSTEMATIC deviation: segments where the average outcome differs from
the average prediction.

So every segment is scored by

    bias_contrib = (n/N) * (mean_y - mean_p)^2 * 1e5 / unc

which is that segment's contribution to Reliability if the data were binned by
that variable. Summed over a partition it IS the Reliability of that binning.

NOISE CONTROL
-------------
With hundreds of pitchers, some group gaps are large by chance. Each gap is
reported with z = gap / SE, SE = sqrt(p(1-p)/n), and segments are only called
systematic at |z| >= 3. Without this the ranking would be a list of small
groups, which is how one ends up chasing noise.

2023 is included for contrast but its BCAT component collapsed on that fold
(Reliability 1777.6, BSS 0.0), so v9's 2023 numbers reflect that defect rather
than a general property. 2024 is the fold to read.

    python experiments/error_analysis/error_analysis.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.deep_learning_state.metrics import decomp  # noqa: E402

HERE = Path(__file__).resolve().parent
W = 0.25, 0.30, 0.45


def load(season):
    o = pd.read_parquet(REPO / 'work' / 'research' / f'oof_{season}.parquet')
    a9 = REPO / 'work' / 'strat' / f'v9_a8_{season}.npy'
    if a9.exists():
        o['p_A9'] = np.load(a9)
    o['p_v9'] = np.clip(W[0] * o.p_A7 + W[1] * o.p_A9 + W[2] * o.p_Bcat,
                        1e-3, 1 - 1e-3)
    o['resid'] = o.control_success - o.p_v9
    return o


def seg_table(d, key, name, unc, N):
    """One partition -> per-group bias, z, and contribution to Reliability."""
    g = d.groupby(key, observed=True).agg(
        n=('control_success', 'size'), y=('control_success', 'mean'),
        p=('p_v9', 'mean'))
    g = g[g.n >= 200]
    if not len(g):
        return None
    g['gap'] = g.y - g.p
    g['se'] = np.sqrt(np.clip(g.p * (1 - g.p), 1e-9, None) / g.n)
    g['z'] = g.gap / g.se
    g['contrib'] = (g.n / N) * g.gap ** 2 * 100000 / unc
    g['var'] = name
    return g.reset_index().rename(columns={key if isinstance(key, str)
                                           else 'index': 'level'})


def bucket(d):
    """Bucketed views of continuous columns. No new model features -- these
    exist only to slice the existing predictions."""
    b = {}
    b['count'] = (d.balls_before.astype(int).astype(str) + '-'
                  + d.strikes_before.astype(int).astype(str))
    b['inning'] = np.clip(d.inning, 1, 10).astype(int)
    b['outs'] = d.outs_before.astype(int)
    b['runners'] = d.num_runners_on.astype(int)
    b['P hand'] = d.pitcher_hand.astype(int)
    b['B hand'] = d.batter_hand.astype(int)
    b['hand pair'] = (d.pitcher_hand.astype(int).astype(str) + 'x'
                      + d.batter_hand.astype(int).astype(str))
    b['month'] = d.game_month.astype(int)
    b['game_type'] = d.game_type.astype(str)
    b['top/bot'] = d.top_bottom.astype(str)
    b['P team'] = d.pitcher_team_id.astype(int)
    b['B team'] = d.batter_team_id.astype(int)
    for c, nm in (('asof_pitcher_n', 'P asof_n'), ('asof_batter_n', 'B asof_n'),
                  ('li', 'leverage'),
                  ('asof_pitcher_success_rate', 'P succ rate'),
                  ('asof_batter_success_rate', 'B succ rate'),
                  ('asof_pitcher_fastball_rate', 'P fastball rate')):
        v = pd.to_numeric(d[c], errors='coerce')
        b[nm] = pd.qcut(v, 10, labels=False, duplicates='drop')
    b['score diff'] = np.clip(d.score_diff_pitcher_team, -5, 5).astype(int)
    return b


def main():
    rows, seg_rows = [], []
    data = {s: load(s) for s in (2023, 2024)}

    # ---------------- 1. overall ----------------------------------------
    print('=' * 104)
    print('  1. 전체 분해')
    print('=' * 104)
    for s, d in data.items():
        y = d.control_success.to_numpy(np.float64)
        p = d.p_v9.to_numpy(np.float64)
        res, rel, bss = decomp(y, p)
        note = '   <-- BCAT 붕괴 fold, 참고용' if s == 2023 else ''
        print(f'  {s}  n {len(d):,}   BSS {bss:8.1f}  Res {res:8.1f}  '
              f'Rel {rel:7.1f}   y {y.mean():.4f}  p {p.mean():.4f}  '
              f'bias {p.mean()-y.mean():+.4f}{note}')
        rows.append(dict(section='overall', season=s, n=len(d), BSS=bss,
                         Resolution=res, Reliability=rel, y=y.mean(),
                         p=p.mean(), bias=p.mean()-y.mean()))

    d = data[2024]
    y = d.control_success.to_numpy(np.float64)
    p = d.p_v9.to_numpy(np.float64)
    N = len(d)
    unc = y.mean() * (1 - y.mean())

    # ---------------- 2. reliability bins --------------------------------
    print('\n' + '=' * 104)
    print('  2. reliability bin (2024, 20분위)')
    print('=' * 104)
    q = pd.qcut(p, 20, labels=False, duplicates='drop')
    g = pd.DataFrame({'p': p, 'y': y, 'q': q}).groupby('q').agg(
        n=('y', 'size'), pred=('p', 'mean'), actual=('y', 'mean'))
    g['gap'] = g.actual - g.pred
    g['rel'] = (g.n / N) * g.gap ** 2 * 100000 / unc
    g['se'] = np.sqrt(g.pred * (1 - g.pred) / g.n)
    g['z'] = g.gap / g.se
    print('   bin      n     pred   actual      gap      z     Rel기여')
    for i, x in g.iterrows():
        mk = '  <--' if abs(x.z) >= 3 else ''
        print(f'   {int(i):3d} {int(x.n):7,}   {x.pred:.4f}   {x.actual:.4f}  '
              f'{x.gap:+.4f}  {x.z:+5.1f} {x.rel:8.1f}{mk}')
    print(f'  Reliability 합 {g.rel.sum():.1f}   |z|>=3 인 bin '
          f'{int((g.z.abs()>=3).sum())}/20')

    # ---------------- 3. segment breakdown -------------------------------
    print('\n' + '=' * 104)
    print('  3. 분할변수별 체계적 편향 (2024, n>=200 그룹만)')
    print('=' * 104)
    B = bucket(d)
    summ = []
    for name, key in B.items():
        t = seg_table(d.assign(_k=np.asarray(key)), '_k', name, unc, N)
        if t is None:
            continue
        seg_rows.append(t)
        summ.append(dict(var=name, n_groups=len(t),
                         rel_total=t.contrib.sum(),
                         n_sig=int((t.z.abs() >= 3).sum()),
                         worst_gap=t.gap.abs().max(),
                         worst_z=t.z.abs().max()))
    S = pd.DataFrame(summ).sort_values('rel_total', ascending=False)
    print(f'  {"변수":<16s} {"그룹":>5s} {"Rel기여합":>10s} {"|z|>=3":>7s} '
          f'{"최대|gap|":>9s} {"최대|z|":>8s}')
    for _, r in S.iterrows():
        print(f'  {r["var"]:<16s} {int(r.n_groups):>5d} {r.rel_total:>10.1f} '
              f'{int(r.n_sig):>7d} {r.worst_gap:>9.4f} {r.worst_z:>8.1f}')

    ALL = pd.concat(seg_rows, ignore_index=True)
    print('\n  --- 체계적 편향 상위 15 그룹 (|z|>=3 만) ---')
    top = ALL[ALL.z.abs() >= 3].nlargest(15, 'contrib')
    print(f'  {"변수":<16s} {"level":>10s} {"n":>8s} {"pred":>7s} {"actual":>7s} '
          f'{"gap":>8s} {"z":>6s} {"기여":>7s}')
    for _, r in top.iterrows():
        print(f'  {r["var"]:<16s} {str(r.level):>10s} {int(r.n):>8,} '
              f'{r.p:>7.4f} {r.y:>7.4f} {r.gap:>+8.4f} {r.z:>+6.1f} '
              f'{r.contrib:>7.2f}')
    if not len(top):
        print('    없음 — |z|>=3 인 그룹이 존재하지 않음')

    # ---------------- 4. entity level ------------------------------------
    print('\n' + '=' * 104)
    print('  4. 엔티티별 (2024)')
    print('=' * 104)
    for key, nm in (('pitcher_id', '투수'), ('batter_id', '타자')):
        t = seg_table(d, key, nm, unc, N)
        print(f'\n  --- {nm}  (n>=200 인 {len(t)}명) ---')
        print(f'    gap 평균 {t.gap.mean():+.5f}   sd {t.gap.std():.5f}   '
              f'|gap| 최대 {t.gap.abs().max():.4f}')
        print(f'    |z|>=2  {int((t.z.abs()>=2).sum())}명 '
              f'(우연 기대 {0.0455*len(t):.1f}명)   '
              f'|z|>=3  {int((t.z.abs()>=3).sum())}명 '
              f'(기대 {0.0027*len(t):.1f}명)')
        print(f'    Reliability 기여 합 {t.contrib.sum():.1f}')
        w = t.nlargest(5, 'contrib')[['level', 'n', 'p', 'y', 'gap', 'z', 'contrib']]
        print('    상위 기여 5:')
        for _, r in w.iterrows():
            print(f'      id {int(r.level):>6d}  n {int(r.n):>6,}  '
                  f'pred {r.p:.4f}  actual {r.y:.4f}  gap {r.gap:+.4f}  '
                  f'z {r.z:+.1f}  기여 {r.contrib:.2f}')
        seg_rows.append(t)

    # ---------------- 5. high-confidence errors ---------------------------
    print('\n' + '=' * 104)
    print('  5. 고확신 구간 (2024)')
    print('=' * 104)
    dev = np.abs(p - 0.5)
    for lo, hi, lab in ((0.00, 0.02, '|p-.5| < 0.02  (무확신)'),
                        (0.02, 0.05, '0.02 ~ 0.05'),
                        (0.05, 0.08, '0.05 ~ 0.08'),
                        (0.08, 1.00, '|p-.5| >= 0.08 (고확신)')):
        m = (dev >= lo) & (dev < hi)
        if m.sum() < 100:
            continue
        yy, pp = y[m], p[m]
        # 방향이 맞았는가: p>0.5 인데 y=1, p<0.5 인데 y=0
        right = ((pp > 0.5) == (yy > 0.5)).mean()
        res_m, rel_m, bss_m = decomp(yy, pp)
        print(f'  {lab:<26s} n {m.sum():>7,} ({m.mean()*100:4.1f}%)  '
              f'pred {pp.mean():.4f}  actual {yy.mean():.4f}  '
              f'gap {yy.mean()-pp.mean():+.4f}  방향적중 {right*100:.1f}%  '
              f'BSS {bss_m:7.1f}')
        rows.append(dict(section='confidence', season=2024, band=lab,
                         n=int(m.sum()), pred=pp.mean(), actual=yy.mean(),
                         gap=yy.mean()-pp.mean(), dir_acc=right, BSS=bss_m))

    print('\n  --- 고확신인데 틀린 표본의 성격 ---')
    hc = m  # last band = high confidence
    wrong = hc & (((p > 0.5) & (y < 0.5)) | ((p < 0.5) & (y > 0.5)))
    rightm = hc & ~wrong
    print(f'    고확신 {hc.sum():,} 중 방향 틀림 {wrong.sum():,} '
          f'({wrong.sum()/hc.sum()*100:.1f}%)')
    for c in ('asof_pitcher_n', 'asof_batter_n', 'li', 'inning',
              'asof_pitcher_success_rate'):
        a = pd.to_numeric(d[c], errors='coerce')
        print(f'    {c:28s} 맞은쪽 중앙 {a[rightm].median():>9.3f}   '
              f'틀린쪽 중앙 {a[wrong].median():>9.3f}')

    pd.DataFrame(rows).to_csv(HERE / 'summary.csv', index=False)
    pd.concat(seg_rows, ignore_index=True).to_csv(
        HERE / 'segment_errors.csv', index=False)
    g.assign(season=2024).to_csv(HERE / 'reliability_bins_2024.csv')
    print(f'\n  wrote summary.csv / segment_errors.csv / reliability_bins_2024.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
