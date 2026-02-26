"""
Chunk-level graph builder.

Converts a list of Chunks (with embeddings) into a torch_geometric Data
object suitable for GNN processing.

Graph structure:
  - Nodes: one per chunk, features = UniXcoder embedding (768-dim)
  - Edges (all bidirectional):
    - Sequential: chunk_i <-> chunk_{i+1} (natural code order)
    - Parent-child: from chunk.parent_index relationships
    - Self-loops: each node connects to itself

Deferred to Phase 2 (not in MVP):
  - Data-flow edges (variable def-use tracking across chunks)
  - Sibling edges
  - Typed edge attributes (all edges are homogeneous in MVP)
"""

from __future__ import annotations

import logging

import torch
from torch_geometric.data import Data

from chunk_gnn.data.chunker import Chunk

log = logging.getLogger(__name__)


class ChunkGraphBuilder:
    """Builds torch_geometric Data objects from chunks + embeddings."""

    def __init__(self, add_self_loops: bool = True):
        self.add_self_loops = add_self_loops

    def build_graph(
        self,
        chunks: list[Chunk],
        embeddings: torch.Tensor,
    ) -> Data:
        """Build a graph Data object from chunks and their embeddings.

        Args:
            chunks: List of Chunk objects (with parent_index set)
            embeddings: Tensor of shape (num_chunks, embedding_dim)

        Returns:
            torch_geometric.data.Data with:
              - x: node features (num_chunks, embedding_dim), float16
              - edge_index: (2, num_edges), long
              - num_nodes: int
              - chunk_kinds: list of "straight"/"control" strings
              - chunk_depths: list of depth ints
        """
        num_nodes = len(chunks)
        assert embeddings.shape[0] == num_nodes, (
            f"Mismatch: {num_nodes} chunks but {embeddings.shape[0]} embeddings"
        )

        # Build edge list
        edges = []

        # Sequential edges: chunk_i <-> chunk_{i+1}
        for i in range(num_nodes - 1):
            edges.append((i, i + 1))
            edges.append((i + 1, i))

        # Parent-child edges: from chunk.parent_index
        for i, chunk in enumerate(chunks):
            if chunk.parent_index is not None:
                edges.append((chunk.parent_index, i))
                edges.append((i, chunk.parent_index))

        # Self-loops
        if self.add_self_loops:
            for i in range(num_nodes):
                edges.append((i, i))

        # Deduplicate edges
        edges = list(set(edges))

        # Build edge_index tensor
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Store embeddings as float16 to save disk space
        x = embeddings.to(dtype=torch.float16)

        data = Data(
            x=x,
            edge_index=edge_index,
            num_nodes=num_nodes,
        )

        # Store metadata as plain Python lists (serializable)
        data.chunk_kinds = [c.kind.value for c in chunks]
        data.chunk_depths = [c.depth for c in chunks]

        return data


def build_single_node_graph(embedding: torch.Tensor) -> Data:
    """Build a graph with a single node (for functions with 1 chunk).

    Still a valid graph — the GNN just returns the projected node embedding.
    """
    x = embedding.unsqueeze(0).to(dtype=torch.float16)  # (1, 768)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # self-loop only
    return Data(
        x=x,
        edge_index=edge_index,
        num_nodes=1,
        chunk_kinds=["straight"],
        chunk_depths=[0],
    )
