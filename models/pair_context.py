from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


PAIRS = [(i, j) for i in range(8) for j in range(i + 1, 8)]
PI = torch.tensor([i for i, _ in PAIRS], dtype=torch.long)
PJ = torch.tensor([j for _, j in PAIRS], dtype=torch.long)


def preprocess(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
    xn = (x - mean) / std
    dx = torch.diff(xn, dim=-1, prepend=xn[..., :1])
    dstd = dx.std(dim=-1, keepdim=True).clamp_min(1e-6)
    dxn = dx / dstd
    t = torch.linspace(-1.0, 1.0, x.shape[-1], device=x.device, dtype=x.dtype)
    t = t.view(1, 1, -1).expand_as(x)
    features = torch.stack([xn, dxn, t], dim=2)
    stats = torch.cat([torch.log(std), torch.log(dstd), mean / std], dim=-1)
    return features, stats


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        padding = 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, 5, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 5, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.gelu(x + h)


class EdgeAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.05) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        return x + self.ff(self.norm2(x))


def lag_cross_correlation(xi: torch.Tensor, xj: torch.Tensor, max_lag: int = 16) -> torch.Tensor:
    values = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = xi[..., :lag], xj[..., -lag:]
        elif lag > 0:
            a, b = xi[..., lag:], xj[..., :-lag]
        else:
            a, b = xi, xj
        values.append((a * b).mean(dim=-1))
    return torch.stack(values, dim=-1)


class PairContextNet(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        pair_channels: int = 128,
        dim: int = 192,
        edge_layers: int = 6,
        heads: int = 8,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.node_in = nn.Conv1d(3, channels, 5, padding=2)
        self.node_blocks = nn.Sequential(
            ResidualTemporalBlock(channels, 1),
            ResidualTemporalBlock(channels, 2),
            nn.Conv1d(channels, channels, 4, stride=2, padding=1),
            nn.GELU(),
            ResidualTemporalBlock(channels, 4),
        )
        self.stat_proj = nn.Linear(3, channels)
        self.pair_in = nn.Conv1d(3 * channels, pair_channels, 1)
        self.pair_blocks = nn.Sequential(
            ResidualTemporalBlock(pair_channels, 1),
            ResidualTemporalBlock(pair_channels, 2),
            ResidualTemporalBlock(pair_channels, 4),
        )
        self.pair_proj = nn.Linear(2 * pair_channels, dim)
        self.xcorr_proj = nn.Linear(66, dim)
        self.edge_layers = nn.ModuleList(
            [EdgeAttentionBlock(dim, heads=heads, dropout=dropout) for _ in range(edge_layers)]
        )
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.register_buffer("pi", PI, persistent=False)
        self.register_buffer("pj", PJ, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, stats = preprocess(x)
        batch, nodes, _, times = features.shape
        node = features.reshape(batch * nodes, 3, times)
        node = self.node_blocks(self.node_in(node))
        node = node.reshape(batch, nodes, node.shape[1], node.shape[2])
        stat = self.stat_proj(stats).unsqueeze(-1)
        node = node + stat

        ni = node[:, self.pi]
        nj = node[:, self.pj]
        pair_sequence = torch.cat([ni, nj, (ni - nj).abs()], dim=2)
        pair_sequence = pair_sequence.reshape(batch * len(PAIRS), pair_sequence.shape[2], pair_sequence.shape[3])
        pair_sequence = self.pair_blocks(self.pair_in(pair_sequence))
        mean_pool = pair_sequence.mean(dim=-1)
        max_pool = pair_sequence.amax(dim=-1)
        pair = self.pair_proj(torch.cat([mean_pool, max_pool], dim=-1)).reshape(batch, len(PAIRS), -1)

        xn = features[:, :, 0]
        dxn = features[:, :, 1]
        corr_x = lag_cross_correlation(xn[:, self.pi], xn[:, self.pj])
        corr_dx = lag_cross_correlation(dxn[:, self.pi], dxn[:, self.pj])
        pair = pair + self.xcorr_proj(torch.cat([corr_x, corr_dx], dim=-1))

        for layer in self.edge_layers:
            pair = layer(pair)
        return torch.sigmoid(self.head(pair).squeeze(-1))
