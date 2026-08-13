"""[Colab GPU] FT-Transformer V0 — three models, one seed.

QUESTION
--------
Not whether this beats v9. Whether it leaves the frontier the round-17
ablation traced on this data:

    BSS = 2052 * corr(v9) - 1051        R^2 = 0.998

Every MLP variant sat on that line: accuracy and decorrelation traded off
one-for-one, so "accurate AND independent" was empty. The gate asks for
BSS >= 750 at corr < 0.85, which the line puts ~57 BSS out of reach. V0 asks
whether attention over feature tokens lands somewhere else.

THREE MODELS
------------
  A  numeric 101 only,          p = sigmoid(f(x))
  B  numeric 101 + 4 embeddings, p = sigmoid(f(x))
  C  same body as B,             p = sigmoid(logit(p_v9) + f(x))

A and B classify directly, so `init` is not used -- the model has to find the
level itself. C offsets from v9 and exists to answer one thing: does a model
handed v9's logit reproduce it, or add resolution on top?

C's constraint is structural, not a choice. Its rows need p_v9, the OOF covers
2022-2024, so C trains on 492,997 rows against A and B's 1,221,585 -- 40%. Any
comparison of C against A and B has to carry that.

The target is always control_success. A residual target is forbidden: round 17
took that route and scored dBSS -177.0 and -218.0 on 2024, with correlation to
v9 of 0.94 to 0.996.

test.csv is never opened. Training is fold_2024_tr (seasons < 2024) and
prediction is fold_2024_va (2024) -- the same walk-forward split v9's own OOF
used, so the predictions are directly comparable.

    python train_ftt.py --data-dir data --models A,B,C --seed 42
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from src.deep_learning_state.metrics import decomp  # noqa: E402
CATS = ['cat_pitcher_team_id', 'cat_batter_team_id', 'cat_count_state',
        'cat_base_state_c']
META = ['row_id', 'season', 'control_success', 'init', 'p_v9',
        'p_A7', 'p_A9', 'p_Bcat']


# ----------------------------------------------------------------- model
class FeatureTokenizer(nn.Module):
    """Each numeric feature gets its own weight and bias into d_token; each
    categorical gets an embedding table. This is what makes FT-Transformer
    different from an MLP -- features become tokens that attend to each other
    rather than being concatenated once at the input."""

    def __init__(self, n_num, cat_card, d):
        super().__init__()
        self.w = nn.Parameter(torch.empty(n_num, d))
        self.b = nn.Parameter(torch.zeros(n_num, d))
        nn.init.normal_(self.w, std=d ** -0.5)
        self.embs = nn.ModuleList([nn.Embedding(c, d) for c in cat_card])
        for e in self.embs:
            nn.init.normal_(e.weight, std=d ** -0.5)
        self.cls = nn.Parameter(torch.empty(1, 1, d))
        nn.init.normal_(self.cls, std=d ** -0.5)

    def forward(self, xn, xc):
        t = xn.unsqueeze(-1) * self.w + self.b                 # B, n_num, d
        if self.embs:
            t = torch.cat([t] + [e(xc[:, i]).unsqueeze(1)
                                 for i, e in enumerate(self.embs)], dim=1)
        return torch.cat([self.cls.expand(len(xn), -1, -1), t], dim=1)


class Block(nn.Module):
    def __init__(self, d, heads, drop):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(),
                                nn.Dropout(drop), nn.Linear(d * 2, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        h = self.n1(x)
        x = x + self.drop(self.att(h, h, h, need_weights=False)[0])
        return x + self.drop(self.ff(self.n2(x)))


class FTTransformer(nn.Module):
    def __init__(self, n_num, cat_card=(), d=64, layers=3, heads=8, drop=0.1):
        super().__init__()
        self.tok = FeatureTokenizer(n_num, cat_card, d)
        self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)     # start at the offset, like v9's GBMs

    def forward(self, xn, xc):
        x = self.tok(xn, xc)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x[:, 0])).squeeze(-1)


# ----------------------------------------------------------------- data
def scale(tr_num, va_num):
    """Median-impute + robust-scale, fitted on TRAIN only. Same recipe as
    dataset.fit_scaler; the missing-indicator expansion is deliberately not
    used, because 101 tokens keep attention affordable where 168 would not."""
    with np.errstate(all='ignore'):
        med = np.nanmedian(tr_num, axis=0)
        q1, q3 = np.nanpercentile(tr_num, [25, 75], axis=0)
    med = np.nan_to_num(med, nan=0.0)
    iqr = np.where(np.isfinite(q3 - q1) & ((q3 - q1) > 1e-9), q3 - q1, 1.0)

    def go(x):
        m = np.isnan(x)
        return np.clip((np.where(m, med, x) - med) / iqr, -8, 8).astype(np.float32)
    return go(tr_num), go(va_num)


def load(data_dir, model):
    tr = pd.read_parquet(Path(data_dir) / 'ftt_2024_tr.parquet')
    va = pd.read_parquet(Path(data_dir) / 'ftt_2024_va.parquet')
    feats = [c for c in tr.columns if c not in META + CATS]
    if model == 'C':
        n0 = len(tr)
        tr = tr[tr.p_v9.notna()].reset_index(drop=True)
        print(f'  model C: p_v9 있는 행만 -> {len(tr):,} / {n0:,} '
              f'({len(tr)/n0*100:.0f}%)   시즌 '
              f'{sorted(int(s) for s in tr.season.unique())}')
    if int(tr.season.max()) >= 2024:
        raise SystemExit('train season leaked into 2024')
    if set(va.season.unique()) != {2024}:
        raise SystemExit('valid is not season 2024')
    return tr, va, feats


def cardinalities(tr, va):
    card, maps = [], []
    for c in CATS:
        u = np.unique(np.concatenate([tr[c].to_numpy(), va[c].to_numpy()]))
        m = {int(v): i for i, v in enumerate(u)}
        maps.append(m)
        card.append(len(u))
    return card, maps


def encode(d, maps):
    return np.stack([d[c].map(m).fillna(0).to_numpy(np.int64)
                     for c, m in zip(CATS, maps)], axis=1)


# ----------------------------------------------------------------- train
def run(model, tr, va, feats, a, dev):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    use_cat = model in ('B', 'C')
    card, maps = cardinalities(tr, va)
    Xtr, Xva = scale(tr[feats].to_numpy(np.float32),
                     va[feats].to_numpy(np.float32))
    Ctr = encode(tr, maps) if use_cat else np.zeros((len(tr), 0), np.int64)
    Cva = encode(va, maps) if use_cat else np.zeros((len(va), 0), np.int64)
    ytr = tr.control_success.to_numpy(np.float32)
    yva = va.control_success.to_numpy(np.float32)

    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    if model == 'C':
        otr = lg(tr.p_v9.to_numpy(np.float64)).astype(np.float32)
        ova = lg(va.p_v9.to_numpy(np.float64)).astype(np.float32)
    else:
        otr = np.zeros(len(tr), np.float32)
        ova = np.zeros(len(va), np.float32)

    net = FTTransformer(len(feats), card if use_cat else (), a.d_token,
                        a.layers, a.heads, a.dropout).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * ((len(tr) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                                total_steps=steps, pct_start=0.2)
    amp = dev.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if amp else None
    bce = nn.BCEWithLogitsLoss()

    tX = torch.from_numpy(Xtr); tC = torch.from_numpy(Ctr)
    tY = torch.from_numpy(ytr); tO = torch.from_numpy(otr)
    vX = torch.from_numpy(Xva).to(dev); vC = torch.from_numpy(Cva).to(dev)
    vO = torch.from_numpy(ova).to(dev)

    print(f'  model {model}: {n_par:,} params  tokens {len(feats)+len(card)*use_cat+1}'
          f'  train {len(tr):,}  valid {len(va):,}')
    best, best_p, bad = -1e18, None, 0
    for ep in range(1, a.epochs + 1):
        net.train()
        perm = torch.randperm(len(tX))
        tot, seen, t0 = 0.0, 0, time.time()
        for s in range(0, len(perm), a.batch):
            idx = perm[s:s + a.batch]
            xb = tX[idx].to(dev, non_blocking=True)
            cb = tC[idx].to(dev, non_blocking=True)
            yb = tY[idx].to(dev, non_blocking=True)
            ob = tO[idx].to(dev, non_blocking=True)
            with torch.autocast('cuda', enabled=amp):
                loss = bce(ob + net(xb, cb), yb)
            opt.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
            sched.step()
            tot += loss.item() * len(idx); seen += len(idx)

        net.eval()
        with torch.no_grad():
            out = []
            for s in range(0, len(vX), 8192):
                with torch.autocast('cuda', enabled=amp):
                    out.append((vO[s:s+8192] + net(vX[s:s+8192],
                                                   vC[s:s+8192])).float().cpu())
            z = torch.cat(out).numpy()
        p = 1.0 / (1.0 + np.exp(-z.astype(np.float64)))
        res, rel, bss = decomp(yva.astype(np.float64), np.clip(p, 1e-6, 1-1e-6))
        print(f'    ep{ep:<3d} loss {tot/seen:.5f}  BSS {bss:8.1f}  Res {res:8.1f}'
              f'  Rel {rel:7.1f}  [{time.time()-t0:.0f}s]')
        if bss > best:
            best, best_p, bad = bss, p, 0
        else:
            bad += 1
            if bad >= a.patience:
                print(f'    early stop (patience {a.patience})')
                break
    return best_p, dict(model=model, seed=a.seed, params=n_par,
                        n_train=len(tr), n_valid=len(va), best_BSS=best)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='train_ftt')
    ap.add_argument('--data-dir', default=str(HERE / 'data'))
    ap.add_argument('--out-dir', default=str(HERE / 'oof'))
    ap.add_argument('--models', default='A,B,C')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--d-token', type=int, default=64)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=1e-5)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--device', default=None)
    a = ap.parse_args(argv)

    if a.device:
        dev = torch.device(a.device)
    elif torch.cuda.is_available():
        dev = torch.device('cuda')
    else:
        raise SystemExit('CUDA 가 필요합니다. Colab GPU 런타임으로 실행하십시오.\n'
                         '(MPS 는 이 실험에서 사용하지 않습니다 — 지시사항)')
    print('=' * 92)
    print(f'  FT-Transformer V0   device {dev}  seed {a.seed}  '
          f'd_token {a.d_token} layers {a.layers} heads {a.heads}')
    if dev.type == 'cuda':
        print(f'  gpu {torch.cuda.get_device_properties(0).name}')
    print('=' * 92)

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    runs = []
    for m in a.models.split(','):
        tr, va, feats = load(a.data_dir, m)
        p, meta = run(m, tr, va, feats, a, dev)
        pd.DataFrame({'row_id': va.row_id.to_numpy(),
                      'p': p.astype(np.float32)}).to_parquet(
            out / f'ftt_{m}_s{a.seed}.parquet', index=False)
        runs.append(meta)
        print(f'  -> wrote ftt_{m}_s{a.seed}.parquet   best BSS {meta["best_BSS"]:.1f}\n')
        del tr, va
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
    (out / f'manifest_s{a.seed}.json').write_text(json.dumps(
        dict(config=vars(a), runs=runs), indent=1, default=str))
    print(f'  manifest -> {out}/manifest_s{a.seed}.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
