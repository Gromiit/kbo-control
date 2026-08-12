"""Networks for the study. All of them predict a LOGIT OFFSET, not a probability.

v9's A family is a level model (logistic regression, which extrapolates across
the season drift) plus a shape model that learns a residual in log-odds on top
of it via LightGBM's init_score. A net that predicts p directly would have to
re-learn the level from scratch and would be handicapped for reasons that have
nothing to do with representational power. So every model here computes

    p = sigmoid(init + f(x))

with the same `init` v9's GBMs were given. `f` is the only thing being compared.
"""
import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    """MODEL 1. Plain net over the numeric block."""

    def __init__(self, n_in, hidden=(256, 128, 64), p_drop=(0.15, 0.10, 0.0)):
        super().__init__()
        layers = [nn.LayerNorm(n_in)]
        prev = n_in
        for h, d in zip(hidden, p_drop):
            layers += [nn.Linear(prev, h), nn.GELU()]
            if d > 0:
                layers.append(nn.Dropout(d))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)   # start exactly at the level model

    def forward(self, xnum, xcat=None):
        return self.net(xnum).squeeze(-1)


class EmbMLP(nn.Module):
    """MODEL 3. Numeric block plus entity embeddings.

    Embedding dims stay small on purpose: the point of the ablation is whether
    identity generalises across seasons, and a wide pitcher embedding would
    answer that question by memorising instead.
    """

    def __init__(self, n_in, cat_sizes, cat_dims, hidden=(256, 128, 64),
                 p_drop=(0.15, 0.10, 0.0), emb_drop=0.10):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(s, d) for s, d in
                                   zip(cat_sizes, cat_dims)])
        for e in self.embs:
            nn.init.normal_(e.weight, std=0.01)
            with torch.no_grad():
                e.weight[0].zero_()          # index 0 = unseen entity
        self.emb_drop = nn.Dropout(emb_drop)
        self.norm = nn.LayerNorm(n_in)
        prev = n_in + int(sum(cat_dims))
        layers = []
        for h, d in zip(hidden, p_drop):
            layers += [nn.Linear(prev, h), nn.GELU()]
            if d > 0:
                layers.append(nn.Dropout(d))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, xnum, xcat):
        e = self.emb_drop(torch.cat([m(xcat[:, j]) for j, m in
                                     enumerate(self.embs)], dim=1))
        return self.net(torch.cat([self.norm(xnum), e], dim=1)).squeeze(-1)


class ResBlock(nn.Module):
    def __init__(self, h, p):
        super().__init__()
        self.f = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(),
                               nn.Dropout(p), nn.Linear(h, h))

    def forward(self, x):
        return x + self.f(x)


class ResidualMLP(nn.Module):
    """MODEL 2. Two projections (base state vs context) summed, then residual
    blocks. The split is what lets the net form context-conditioned corrections
    without the numeric block having to carry them through every layer."""

    def __init__(self, n_base, n_ctx, h=128, blocks=3, p=0.10,
                 cat_sizes=None, cat_dims=None):
        super().__init__()
        self.n_base = n_base
        self.pb = nn.Sequential(nn.LayerNorm(n_base), nn.Linear(n_base, h))
        self.pc = nn.Sequential(nn.LayerNorm(n_ctx), nn.Linear(n_ctx, h))
        self.use_cat = cat_sizes is not None
        if self.use_cat:
            self.embs = nn.ModuleList([nn.Embedding(s, d) for s, d in
                                       zip(cat_sizes, cat_dims)])
            for e in self.embs:
                nn.init.normal_(e.weight, std=0.01)
                with torch.no_grad():
                    e.weight[0].zero_()
            self.pe = nn.Linear(int(sum(cat_dims)), h)
        self.blocks = nn.Sequential(*[ResBlock(h, p) for _ in range(blocks)])
        self.head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, xnum, xcat=None):
        z = self.pb(xnum[:, :self.n_base]) + self.pc(xnum[:, self.n_base:])
        if self.use_cat:
            z = z + self.pe(torch.cat([m(xcat[:, j]) for j, m in
                                       enumerate(self.embs)], dim=1))
        return self.head(self.blocks(z)).squeeze(-1)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
