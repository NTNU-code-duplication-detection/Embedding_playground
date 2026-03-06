"""GNN encoder and pair classification model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PairForwardOut:
    """Forward output bundle."""

    logit: torch.Tensor
    emb_a: torch.Tensor
    emb_b: torch.Tensor


class EdgeTypeGNNLayer(nn.Module):
    """Message passing layer with edge-type embeddings."""

    def __init__(self, hidden_dim: int, edge_type_dim: int, num_edge_types: int, dropout: float):
        super().__init__()
        self.edge_emb = nn.Embedding(num_edge_types, edge_type_dim)
        self.msg = nn.Linear(hidden_dim + edge_type_dim, hidden_dim)
        self.upd = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.edge_emb.weight)
        nn.init.xavier_uniform_(self.msg.weight)
        nn.init.zeros_(self.msg.bias)
        nn.init.xavier_uniform_(self.upd.weight)
        nn.init.zeros_(self.upd.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0 or x.numel() == 0:
            return x

        src = edge_index[0]
        dst = edge_index[1]

        edge_feat = self.edge_emb(edge_type)
        msg_in = torch.cat([x[src], edge_feat], dim=-1)
        msg = F.relu(self.msg(msg_in))

        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)

        deg = torch.zeros((x.shape[0],), device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones((dst.shape[0],), device=x.device, dtype=x.dtype))
        deg = deg.clamp(min=1.0).unsqueeze(-1)

        upd = self.upd(agg / deg)
        upd = self.dropout(upd)
        return self.norm(x + upd)


class MethodEncoder(nn.Module):
    """Encode one method graph to a fixed-size embedding."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_edge_types: int,
        edge_type_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                EdgeTypeGNNLayer(
                    hidden_dim=hidden_dim,
                    edge_type_dim=edge_type_dim,
                    num_edge_types=num_edge_types,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return torch.zeros((self.hidden_dim,), device=x.device)

        h = F.relu(self.in_proj(x))
        for layer in self.layers:
            h = layer(h, edge_index, edge_type)
            h = self.dropout(h)

        return h.mean(dim=0)


class PairMLPHead(nn.Module):
    """Final pair classifier head over two program embeddings."""

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
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([emb_a, emb_b, (emb_a - emb_b).abs(), emb_a * emb_b], dim=-1)
        return self.mlp(feat).squeeze(-1)


class ProgramCloneModel(nn.Module):
    """Program-level clone detector: method GNN + mean pooling + MLP head."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_edge_types: int,
        edge_type_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.method_encoder = MethodEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_edge_types=num_edge_types,
            edge_type_dim=edge_type_dim,
            dropout=dropout,
        )
        self.pair_head = PairMLPHead(hidden_dim=hidden_dim, dropout=dropout)
        self.hidden_dim = hidden_dim

    def encode_program(self, method_records: list[dict], device: str) -> torch.Tensor:
        if not method_records:
            return torch.zeros((self.hidden_dim,), device=device)

        method_embs = []
        for record in method_records:
            x = record["x"].to(device)
            edge_index = record["edge_index"].to(device)
            edge_type = record["edge_type"].to(device)
            method_embs.append(self.method_encoder(x, edge_index, edge_type))

        return torch.stack(method_embs, dim=0).mean(dim=0)

    def forward_pair(self, methods_a: list[dict], methods_b: list[dict], device: str) -> PairForwardOut:
        emb_a = self.encode_program(methods_a, device)
        emb_b = self.encode_program(methods_b, device)
        logit = self.pair_head(emb_a, emb_b)
        return PairForwardOut(logit=logit, emb_a=emb_a, emb_b=emb_b)

    @staticmethod
    def cosine_similarity(emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0), dim=-1).squeeze(0)
