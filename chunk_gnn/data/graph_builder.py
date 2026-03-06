"""
Chunk-level graph builder.

Converts a list of Chunks (with embeddings) into a torch_geometric Data
object suitable for GNN processing.

Graph structure:
  - Nodes: one per chunk, features = UniXcoder embedding (768-dim)
  - Edges (all bidirectional, with typed attributes):
    - Self-loops: each node connects to itself (type 0)
    - Sequential: chunk_i <-> chunk_{i+1} (type 1)
    - Parent-child: from chunk.parent_index relationships (type 2)
    - Data-flow: chunks sharing variable names (type 3)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from itertools import combinations

import torch
from torch_geometric.data import Data

from chunk_gnn.data.chunker import Chunk

log = logging.getLogger(__name__)

# Edge type constants (stored in data.edge_type tensor)
EDGE_SELF_LOOP = 0
EDGE_SEQUENTIAL = 1
EDGE_PARENT_CHILD = 2
EDGE_DATA_FLOW = 3
NUM_EDGE_TYPES = 4


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
            chunks: List of Chunk objects (with parent_index and variables set)
            embeddings: Tensor of shape (num_chunks, embedding_dim)

        Returns:
            torch_geometric.data.Data with:
              - x: node features (num_chunks, embedding_dim), float16
              - edge_index: (2, num_edges), long
              - edge_type: (num_edges,), long — type per edge
              - num_nodes: int
              - chunk_kinds: list of "straight"/"control" strings
              - chunk_depths: list of depth ints
        """
        num_nodes = len(chunks)
        assert embeddings.shape[0] == num_nodes, (
            f"Mismatch: {num_nodes} chunks but {embeddings.shape[0]} embeddings"
        )

        # Build typed edge list: (src, dst, edge_type)
        typed_edges: set[tuple[int, int, int]] = set()

        # Sequential edges: chunk_i <-> chunk_{i+1}
        for i in range(num_nodes - 1):
            typed_edges.add((i, i + 1, EDGE_SEQUENTIAL))
            typed_edges.add((i + 1, i, EDGE_SEQUENTIAL))

        # Parent-child edges: from chunk.parent_index
        for i, chunk in enumerate(chunks):
            if chunk.parent_index is not None:
                typed_edges.add((chunk.parent_index, i, EDGE_PARENT_CHILD))
                typed_edges.add((i, chunk.parent_index, EDGE_PARENT_CHILD))

        # Self-loops
        if self.add_self_loops:
            for i in range(num_nodes):
                typed_edges.add((i, i, EDGE_SELF_LOOP))

        # Data flow edges: connect chunks that share variable names
        var_to_chunks: dict[str, list[int]] = defaultdict(list)
        for i, chunk in enumerate(chunks):
            for var in chunk.variables:
                var_to_chunks[var].append(i)

        for var, chunk_indices in var_to_chunks.items():
            if len(chunk_indices) < 2:
                continue
            for a, b in combinations(chunk_indices, 2):
                typed_edges.add((a, b, EDGE_DATA_FLOW))
                typed_edges.add((b, a, EDGE_DATA_FLOW))

        # Build tensors
        if typed_edges:
            edge_list = sorted(typed_edges)  # Deterministic ordering
            srcs, dsts, types = zip(*edge_list)
            edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
            edge_type = torch.tensor(types, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty((0,), dtype=torch.long)

        # Store embeddings as float16 to save disk space
        x = embeddings.to(dtype=torch.float16)

        data = Data(
            x=x,
            edge_index=edge_index,
            num_nodes=num_nodes,
        )
        data.edge_type = edge_type

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
    edge_type = torch.tensor([EDGE_SELF_LOOP], dtype=torch.long)
    data = Data(
        x=x,
        edge_index=edge_index,
        num_nodes=1,
        chunk_kinds=["straight"],
        chunk_depths=[0],
    )
    data.edge_type = edge_type
    return data
