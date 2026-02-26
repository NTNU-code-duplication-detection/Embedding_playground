"""
Module for model
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GnnOut:
    graph_emb: torch.Tensor  # [H]


class EdgeTypeGNNLayer(nn.Module):
    """
    GNN class
    """
    def __init__(self, hidden_dim: int, edge_type_dim: int, num_edge_types: int, dropout: float):
        super().__init__()
        self.edge_emb = nn.Embedding(num_edge_types, edge_type_dim)
        self.msg = nn.Linear(hidden_dim + edge_type_dim, hidden_dim, bias=True)
        self.upd = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """
        x: [N, H]
        edge_index: [2, E] (src,dst)
        edge_type: [E]
        """
        if edge_index.numel() == 0:
            return x

        src = edge_index[0]  # [E]
        dst = edge_index[1]  # [E]

        et = self.edge_emb(edge_type)                # [E, T]
        m_in = torch.cat([x[src], et], dim=-1)       # [E, H+T]
        m = self.msg(m_in)                           # [E, H]
        m = F.relu(m)

        # mean aggregate to dst
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, m)

        deg = torch.zeros((x.size(0),), device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones((dst.size(0),), device=x.device, dtype=x.dtype))
        deg = deg.clamp(min=1.0).unsqueeze(-1)

        agg = agg / deg

        out = self.upd(agg)
        out = self.dropout(out)

        x2 = self.norm(x + out)  # residual
        return x2


class MethodEncoder(nn.Module):
    """
    Encodes entire method
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, num_edge_types: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            EdgeTypeGNNLayer(hidden_dim, edge_type_dim=16, num_edge_types=num_edge_types, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """
        Returns graph embedding [H]
        """
        h = self.proj(x)
        h = F.relu(h)

        for layer in self.layers:
            h = layer(h, edge_index, edge_type)

        # mean pool
        g = h.mean(dim=0)
        return g


class PairClassifier(nn.Module):
    """
    Classifier for pairs, simple 2layer MLP
    """
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        a,b: [H]
        returns logits: [1]
        """
        feat = torch.cat([a, b, (a - b).abs(), a * b], dim=-1)
        return self.mlp(feat).squeeze(-1)
