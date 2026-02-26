"""
Chunk-level GNN encoder.

A lightweight GNN that processes chunk-level code graphs into fixed-size
embeddings. Each node's features are pre-computed UniXcoder embeddings
(768-dim), so the GNN's job is to propagate structural information between
chunks via message passing and produce a single graph-level embedding.

Supports two GNN types (selectable via config "gnn_type"):
  - "GCNConv": Standard graph convolution (uniform neighbor aggregation)
  - "GATConv": Graph attention (learned attention weights per neighbor)

Architecture:
  Input (768-dim per node)
    -> [Optional] Learned input projection (768 -> hidden_dim)
    -> GNN layers (hidden -> hidden) with optional residual connections
    -> ReLU + Dropout after each layer
    -> Global mean pool over all nodes
    -> Linear projection to output dim (128)
    -> [Optional] L2 normalization
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


def _make_conv(
    gnn_type: str, in_dim: int, out_dim: int, heads: int = 4,
) -> nn.Module:
    """Create a single GNN convolution layer."""
    if gnn_type == "GCNConv":
        return GCNConv(in_dim, out_dim)
    if gnn_type == "GATConv":
        # GATConv outputs heads * out_channels, so we set out_channels
        # such that heads * out_channels = out_dim
        assert out_dim % heads == 0, (
            f"hidden_dim ({out_dim}) must be divisible by num_heads ({heads})"
        )
        per_head = out_dim // heads
        return GATConv(in_dim, per_head, heads=heads, concat=True)
    raise ValueError(f"Unknown gnn_type: {gnn_type!r}. Use 'GCNConv' or 'GATConv'.")


class ChunkGNNEncoder(nn.Module):
    """GNN that encodes a chunk-level graph into a fixed-size embedding."""

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 256,
        out_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        input_projection: bool = False,
        residual: bool = False,
        l2_normalize: bool = False,
        gnn_type: str = "GCNConv",
        num_heads: int = 4,
    ):
        super().__init__()

        self.residual = residual
        self.l2_normalize = l2_normalize

        # Optional learned projection from UniXcoder space to GNN space
        if input_projection:
            self.input_proj = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            )
            gnn_in_dim = hidden_dim
        else:
            self.input_proj = None
            gnn_in_dim = in_dim

        self.convs = nn.ModuleList()
        self.convs.append(_make_conv(gnn_type, gnn_in_dim, hidden_dim, num_heads))
        for _ in range(num_layers - 1):
            self.convs.append(_make_conv(gnn_type, hidden_dim, hidden_dim, num_heads))

        # Residual projection for first layer when input dim != hidden dim
        if residual and gnn_in_dim != hidden_dim:
            self.residual_proj = nn.Linear(gnn_in_dim, hidden_dim)
        else:
            self.residual_proj = None

        self.dropout = nn.Dropout(dropout)
        self.project = nn.Linear(hidden_dim, out_dim)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """Encode a batched graph into per-graph embeddings.

        Args:
            x: Node features, shape (total_nodes_in_batch, in_dim)
            edge_index: Edge connectivity, shape (2, total_edges_in_batch)
            batch: Batch vector mapping each node to its graph index,
                   shape (total_nodes_in_batch,)

        Returns:
            Graph-level embeddings, shape (num_graphs_in_batch, out_dim)
        """
        # Optional input projection
        if self.input_proj is not None:
            x = self.input_proj(x)

        for i, conv in enumerate(self.convs):
            identity = x
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

            # Residual connection
            if self.residual:
                if i == 0 and self.residual_proj is not None:
                    identity = self.residual_proj(identity)
                x = x + identity

        # Pool all node embeddings into one vector per graph
        x = global_mean_pool(x, batch)

        # Project to output dimension
        x = self.project(x)

        # L2 normalize to unit hypersphere (spreads cosine similarity range)
        if self.l2_normalize:
            x = F.normalize(x, dim=-1)

        return x
