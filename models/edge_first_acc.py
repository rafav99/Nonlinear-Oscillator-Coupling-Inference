from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


UNDIRECTED = [(i, j) for i in range(8) for j in range(i + 1, 8)]
DIRECTED = [(i, j) for i in range(8) for j in range(8) if i != j]


def smooth_second_difference(x: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0], device=x.device, dtype=x.dtype)
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    batch, nodes, times = x.shape
    flat = x.reshape(batch * nodes, 1, times)
    smooth = F.conv1d(flat, kernel, padding=2).reshape(batch, nodes, times)
    dx = torch.diff(smooth, dim=-1, prepend=smooth[..., :1])
    ddx = torch.diff(dx, dim=-1, prepend=dx[..., :1])
    return ddx


def acc_features(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
    xn = (x - mean) / std
    dx = torch.diff(xn, dim=-1, prepend=xn[..., :1])
    ddx = smooth_second_difference(xn)
    t = torch.linspace(-1.0, 1.0, x.shape[-1], device=x.device, dtype=x.dtype)
    t = t.view(1, 1, -1).expand_as(x)
    return torch.stack([xn, dx, ddx, t], dim=2)


class TemporalAttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.softmax(self.score(x).squeeze(-1), dim=-1)
        return torch.sum(weight.unsqueeze(-1) * x, dim=-2)


class EdgeFirstAccNet(nn.Module):
    def __init__(
        self,
        c: int = 32,
        e: int = 64,
        d: int = 192,
        layers: int = 4,
        heads: int = 8,
        drop: float = 0.10,
    ) -> None:
        super().__init__()
        self.node = nn.Sequential(
            nn.Conv1d(4, c, 7, padding=3),
            nn.GELU(),
            nn.Conv1d(c, c, 5, padding=2),
            nn.GELU(),
        )
        self.edge = nn.Sequential(
            nn.Conv1d(4 * c, e, 1),
            nn.GELU(),
            nn.Conv1d(e, d, 3, padding=1),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=heads,
            dim_feedforward=4 * d,
            dropout=drop,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.edge_context = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.temporal_pool = TemporalAttentionPool(d)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = acc_features(x)
        batch, nodes, _, times = features.shape
        encoded_nodes = self.node(features.reshape(batch * nodes, 4, times))
        encoded_nodes = encoded_nodes.reshape(batch, nodes, -1, times)

        directed = []
        for i, j in DIRECTED:
            hi = encoded_nodes[:, i]
            hj = encoded_nodes[:, j]
            directed.append(torch.cat([hi, hj, hj - hi, hi * hj], dim=1))
        edge_seq = torch.stack(directed, dim=1)
        edge_seq = edge_seq.reshape(batch * len(DIRECTED), edge_seq.shape[2], times)
        edge_seq = self.edge(edge_seq).transpose(1, 2)
        pooled = self.temporal_pool(edge_seq).reshape(batch, len(DIRECTED), -1)
        contextual = self.edge_context(pooled)

        index = {(i, j): k for k, (i, j) in enumerate(DIRECTED)}
        symmetric = []
        for i, j in UNDIRECTED:
            symmetric.append(0.5 * (contextual[:, index[(i, j)]] + contextual[:, index[(j, i)]]))
        symmetric = torch.stack(symmetric, dim=1)
        return torch.sigmoid(self.head(symmetric).squeeze(-1))
