"""Models. No device branching lives here -- these are plain nn.Modules.

Both predict a LOGIT OFFSET on top of `init`, the level model's log-odds, for
the same reason round 17 did: v9's A family is level + shape, and a net asked
to predict p from scratch would be re-learning the season drift instead of the
thing under test.

    p = sigmoid(init + f(static, sequence))
"""
import torch
import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(self, n_in, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(n_in), nn.Linear(n_in, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)     # start exactly at the level model

    def forward(self, z):
        return self.net(z).squeeze(-1)


class StaticMLP(nn.Module):
    """Baseline: ignores the sequence. The control the GRU has to beat."""

    def __init__(self, n_static, n_channels=0, hidden=64, num_layers=1,
                 head_hidden=64, dropout=0.1, sequence_length=0):
        super().__init__()
        self.head = MLPHead(n_static, head_hidden, dropout)

    def forward(self, x, seq, length):
        return self.head(x)


class GRUState(nn.Module):
    """The pitcher's recent pitches, summarised, concatenated with the row.

    The GRU runs over the padded window and the final state is taken at each
    row's own length, so padded steps never reach the head. Rows with an empty
    history (first pitch of a pitcher's season) contribute a zero state, which
    the zero-initialised head turns into no correction at all.
    """

    def __init__(self, n_static, n_channels, hidden=32, num_layers=1,
                 head_hidden=64, dropout=0.1, sequence_length=16):
        super().__init__()
        self.gru = nn.GRU(n_channels, hidden, num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.seq_norm = nn.LayerNorm(hidden)
        self.head = MLPHead(n_static + hidden, head_hidden, dropout)

    def forward(self, x, seq, length):
        h, _ = self.gru(seq)                                  # B, L, H
        idx = (length.clamp(min=1) - 1).view(-1, 1, 1).expand(-1, 1, h.size(-1))
        last = h.gather(1, idx).squeeze(1)
        last = torch.where((length > 0).unsqueeze(-1), last,
                           torch.zeros_like(last))
        return self.head(torch.cat([x, self.seq_norm(last)], dim=1))


REGISTRY = {'gru': GRUState, 'mlp': StaticMLP}


def build(cfg, n_static, n_channels):
    if cfg.model not in REGISTRY:
        raise SystemExit(f'unknown model {cfg.model!r}, have {list(REGISTRY)}')
    return REGISTRY[cfg.model](
        n_static=n_static, n_channels=n_channels, hidden=cfg.hidden_size,
        num_layers=cfg.num_layers, head_hidden=cfg.head_hidden,
        dropout=cfg.dropout, sequence_length=cfg.sequence_length)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
