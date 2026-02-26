"""
Tree-sitter + UniXCoder Siamese clone detection pipeline.

This script implements:
1. Java parsing with Tree-sitter.
2. Statement/control-header chunk extraction.
3. Optional AST+token/control-flow graph construction.
4. UniXCoder chunk embedding via AutoTokenizer + AutoModel.
5. Siamese training on CodeXGLUE BigCloneBench pairs.
6. Contrastive or InfoNCE-based training objective.
7. Inference by cosine similarity + thresholding.
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
import sys
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from tree_sitter import Language, Parser
import tree_sitter_java
from transformers import AutoModel, AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from data.data_generators.codexglue_dataset_gen import bigclonebench_hf_generator
from data.data_generators.schema import CodeSample


class EdgeType(IntEnum):
    """Typed edges used in the function graph."""

    PARENT_TO_CHILD = 0
    CHILD_TO_PARENT = 1
    NEXT_SIBLING = 2
    NEXT_TOKEN = 3
    NEXT_USE = 4
    IF_COND_TO_THEN = 5
    IF_COND_TO_ELSE = 6
    LOOP_COND_TO_BODY = 7
    NEXT_STMT = 8


@dataclass(frozen=True)
class StatementChunk:
    """A statement-level semantic unit extracted from a Java method."""

    text: str
    node_type: str
    start_byte: int
    end_byte: int
    ast_depth: int
    method_index: int
    node_id: int


@dataclass(frozen=True)
class FunctionGraph:
    """Chunk nodes and edges for a function snippet."""

    chunks: List[StatementChunk]
    edge_index: torch.Tensor  # [2, E], dtype long
    edge_types: torch.Tensor  # [E], dtype long


def iter_named_nodes(root) -> Iterator:
    """Depth-first traversal of named Tree-sitter nodes."""
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        for child in reversed(node.named_children):
            stack.append(child)


class JavaTreeSitterParser:
    """Creates and uses a Tree-sitter Java parser."""

    def __init__(self) -> None:
        language = Language(tree_sitter_java.language())
        self.parser = Parser(language)

    def parse_method_like_code(self, code: str) -> Tuple[bytes, object]:
        """
        Parse `code` directly; if no method is found, wrap it in a class.

        Returns:
            (source_bytes, tree)
        """
        direct_bytes = code.encode("utf-8", errors="ignore")
        direct_tree = self.parser.parse(direct_bytes)
        if self._contains_method(direct_tree.root_node):
            return direct_bytes, direct_tree

        wrapped_src = f"class __Wrapper__ {{\n{code}\n}}\n"
        wrapped_bytes = wrapped_src.encode("utf-8", errors="ignore")
        wrapped_tree = self.parser.parse(wrapped_bytes)
        if self._contains_method(wrapped_tree.root_node):
            return wrapped_bytes, wrapped_tree

        return direct_bytes, direct_tree

    @staticmethod
    def _contains_method(root) -> bool:
        for node in iter_named_nodes(root):
            if node.type == "method_declaration":
                return True
        return False


class TreeSitterJavaChunkGraphBuilder:
    """
    Extracts statement-level chunks and builds a typed graph.

    Inspired by existing notebook chunkers:
    - CONTROL_NODES / STRAIGHT_NODES split
    - control-header extraction
    - Tree-sitter AST traversal
    """

    CONTROL_NODES = {
        "if_statement",
        "for_statement",
        "enhanced_for_statement",
        "while_statement",
        "do_statement",
        "switch_statement",
        "try_statement",
        "try_with_resources_statement",
        "catch_clause",
        "synchronized_statement",
    }

    STRAIGHT_NODES = {
        "local_variable_declaration",
        "expression_statement",
        "return_statement",
        "throw_statement",
        "assert_statement",
        "break_statement",
        "continue_statement",
        "yield_statement",
        "labeled_statement",
    }

    LOOP_NODES = {
        "for_statement",
        "enhanced_for_statement",
        "while_statement",
        "do_statement",
    }

    CHUNK_NODE_TYPES = CONTROL_NODES | STRAIGHT_NODES

    def __init__(self, parser: JavaTreeSitterParser, max_chunks: int = 128) -> None:
        self.parser = parser
        self.max_chunks = max_chunks

    def build(self, code: str) -> FunctionGraph:
        source_bytes, tree = self.parser.parse_method_like_code(code)
        method_nodes = [n for n in iter_named_nodes(tree.root_node) if n.type == "method_declaration"]

        if not method_nodes:
            return self._fallback_graph(code)

        all_chunks: List[StatementChunk] = []
        all_edges: List[Tuple[int, int, int]] = []
        offset = 0

        for method_index, method_node in enumerate(method_nodes):
            local_chunks, local_edges = self._build_method_subgraph(
                source_bytes=source_bytes,
                method_node=method_node,
                method_index=method_index,
            )
            if not local_chunks:
                continue
            all_chunks.extend(local_chunks)
            all_edges.extend([(a + offset, b + offset, t) for (a, b, t) in local_edges])
            offset += len(local_chunks)
            if len(all_chunks) >= self.max_chunks:
                break

        if not all_chunks:
            return self._fallback_graph(code)

        if len(all_chunks) > self.max_chunks:
            all_chunks = all_chunks[: self.max_chunks]
            max_idx = len(all_chunks) - 1
            all_edges = [(a, b, t) for (a, b, t) in all_edges if a <= max_idx and b <= max_idx]

        edge_index, edge_types = self._to_edge_tensors(all_edges)
        return FunctionGraph(chunks=all_chunks, edge_index=edge_index, edge_types=edge_types)

    def _build_method_subgraph(
        self,
        source_bytes: bytes,
        method_node,
        method_index: int,
    ) -> Tuple[List[StatementChunk], List[Tuple[int, int, int]]]:
        entries: List[Tuple] = []
        stack = [(method_node, 0)]
        while stack:
            node, depth = stack.pop()
            if node.type in self.CHUNK_NODE_TYPES:
                text = self._extract_chunk_text(source_bytes, node)
                if text:
                    entries.append((node, text, depth))
            for child in reversed(node.named_children):
                stack.append((child, depth + 1))

        if not entries:
            return [], []

        entries.sort(key=lambda e: (e[0].start_byte, e[0].end_byte))
        chunks = [
            StatementChunk(
                text=text,
                node_type=node.type,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                ast_depth=depth,
                method_index=method_index,
                node_id=node.id,
            )
            for (node, text, depth) in entries
        ]

        node_to_idx = {node.id: idx for idx, (node, _, _) in enumerate(entries)}
        edge_set: set[Tuple[int, int, int]] = set()

        self._add_parent_and_sibling_edges(entries, node_to_idx, edge_set)
        self._add_next_stmt_edges(chunks, edge_set)
        self._add_identifier_edges(method_node, entries, node_to_idx, source_bytes, edge_set)
        self._add_control_flow_edges(entries, node_to_idx, edge_set)

        return chunks, sorted(edge_set)

    @staticmethod
    def _extract_chunk_text(source_bytes: bytes, node) -> str:
        if node.type in TreeSitterJavaChunkGraphBuilder.CONTROL_NODES:
            # Prefer only the control header, not the full block body.
            if node.type == "if_statement":
                consequence = node.child_by_field_name("consequence")
                end_byte = consequence.start_byte if consequence is not None else node.end_byte
            else:
                body = node.child_by_field_name("body")
                end_byte = body.start_byte if body is not None else node.end_byte

            header = source_bytes[node.start_byte:end_byte].decode("utf-8", errors="ignore").strip()
            if header.endswith("{"):
                header = header[:-1].rstrip()
            if header:
                return header

            full = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
            return full.split("{")[0].strip()

        return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore").strip()

    @staticmethod
    def _add_parent_and_sibling_edges(entries, node_to_idx, edge_set: set[Tuple[int, int, int]]) -> None:
        for child_idx, (node, _, _) in enumerate(entries):
            parent = node.parent
            while parent is not None and parent.id not in node_to_idx:
                parent = parent.parent
            if parent is not None:
                parent_idx = node_to_idx[parent.id]
                edge_set.add((parent_idx, child_idx, int(EdgeType.PARENT_TO_CHILD)))
                edge_set.add((child_idx, parent_idx, int(EdgeType.CHILD_TO_PARENT)))

            sibling = node.next_named_sibling
            while sibling is not None and sibling.id not in node_to_idx:
                sibling = sibling.next_named_sibling
            if sibling is not None:
                sib_idx = node_to_idx[sibling.id]
                edge_set.add((child_idx, sib_idx, int(EdgeType.NEXT_SIBLING)))

    @staticmethod
    def _add_next_stmt_edges(chunks: List[StatementChunk], edge_set: set[Tuple[int, int, int]]) -> None:
        if len(chunks) < 2:
            return
        ordered = sorted(range(len(chunks)), key=lambda i: chunks[i].start_byte)
        for i in range(len(ordered) - 1):
            edge_set.add((ordered[i], ordered[i + 1], int(EdgeType.NEXT_STMT)))

    @staticmethod
    def _first_descendant_chunk_idx(node, node_to_idx: Dict[int, int]) -> Optional[int]:
        if node is None:
            return None
        queue = [node]
        best_idx = None
        best_start = None
        while queue:
            cur = queue.pop(0)
            idx = node_to_idx.get(cur.id)
            if idx is not None:
                if best_start is None or cur.start_byte < best_start:
                    best_start = cur.start_byte
                    best_idx = idx
            queue.extend(cur.named_children)
        return best_idx

    def _add_control_flow_edges(self, entries, node_to_idx, edge_set: set[Tuple[int, int, int]]) -> None:
        for node, _, _ in entries:
            src_idx = node_to_idx[node.id]

            if node.type == "if_statement":
                then_idx = self._first_descendant_chunk_idx(node.child_by_field_name("consequence"), node_to_idx)
                else_idx = self._first_descendant_chunk_idx(node.child_by_field_name("alternative"), node_to_idx)
                if then_idx is not None:
                    edge_set.add((src_idx, then_idx, int(EdgeType.IF_COND_TO_THEN)))
                if else_idx is not None:
                    edge_set.add((src_idx, else_idx, int(EdgeType.IF_COND_TO_ELSE)))

            if node.type in self.LOOP_NODES:
                body_idx = self._first_descendant_chunk_idx(node.child_by_field_name("body"), node_to_idx)
                if body_idx is not None:
                    edge_set.add((src_idx, body_idx, int(EdgeType.LOOP_COND_TO_BODY)))

    def _add_identifier_edges(
        self,
        method_node,
        entries,
        node_to_idx,
        source_bytes: bytes,
        edge_set: set[Tuple[int, int, int]],
    ) -> None:
        def covering_chunk_idx(byte_pos: int) -> Optional[int]:
            best_idx = None
            best_span = None
            for idx, (node, _, _) in enumerate(entries):
                if node.start_byte <= byte_pos < node.end_byte:
                    span = node.end_byte - node.start_byte
                    if best_span is None or span < best_span:
                        best_span = span
                        best_idx = idx
            return best_idx

        identifiers = [n for n in iter_named_nodes(method_node) if n.type == "identifier"]
        identifiers.sort(key=lambda n: n.start_byte)

        prev_chunk_idx: Optional[int] = None
        last_use: Dict[str, int] = {}

        for ident in identifiers:
            chunk_idx = covering_chunk_idx(ident.start_byte)
            if chunk_idx is None:
                continue

            if prev_chunk_idx is not None and prev_chunk_idx != chunk_idx:
                edge_set.add((prev_chunk_idx, chunk_idx, int(EdgeType.NEXT_TOKEN)))
            prev_chunk_idx = chunk_idx

            name = source_bytes[ident.start_byte:ident.end_byte].decode("utf-8", errors="ignore")
            prior_use = last_use.get(name)
            if prior_use is not None and prior_use != chunk_idx:
                edge_set.add((prior_use, chunk_idx, int(EdgeType.NEXT_USE)))
            last_use[name] = chunk_idx

    @staticmethod
    def _to_edge_tensors(edges: Iterable[Tuple[int, int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_list = list(edges)
        if not edge_list:
            return (
                torch.empty((2, 0), dtype=torch.long),
                torch.empty((0,), dtype=torch.long),
            )

        src = [e[0] for e in edge_list]
        dst = [e[1] for e in edge_list]
        types = [e[2] for e in edge_list]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_types = torch.tensor(types, dtype=torch.long)
        return edge_index, edge_types

    @staticmethod
    def _fallback_graph(code: str) -> FunctionGraph:
        text = code.strip() or "// empty"
        chunk = StatementChunk(
            text=text,
            node_type="fallback_chunk",
            start_byte=0,
            end_byte=len(text.encode("utf-8", errors="ignore")),
            ast_depth=0,
            method_index=0,
            node_id=0,
        )
        return FunctionGraph(
            chunks=[chunk],
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_types=torch.empty((0,), dtype=torch.long),
        )


class UniXCoderChunkEmbedder(nn.Module):
    """Embeds chunk texts with HuggingFace AutoTokenizer + AutoModel."""

    def __init__(
        self,
        model_name: str = "microsoft/unixcoder-base",
        max_length: int = 256,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    def forward(self, texts: Sequence[str], device: torch.device, batch_size: int = 16) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self.hidden_size), dtype=torch.float32, device=device)

        vectors: List[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = list(texts[start : start + batch_size])
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            if self.freeze_backbone:
                with torch.no_grad():
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

            # Fixed-size chunk embedding from [CLS] token state.
            cls = outputs.last_hidden_state[:, 0, :]
            vectors.append(cls)

        return torch.cat(vectors, dim=0)


class TypedMessagePassingLayer(nn.Module):
    """A small edge-type aware message passing layer."""

    def __init__(self, hidden_size: int, num_edge_types: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.edge_embedding = nn.Embedding(num_edge_types, hidden_size)
        self.self_linear = nn.Linear(hidden_size, hidden_size)
        self.msg_linear = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_types: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return x

        src = edge_index[0]
        dst = edge_index[1]
        typed_messages = x[src] + self.edge_embedding(edge_types)

        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, dst, typed_messages)

        out = self.self_linear(x) + self.msg_linear(aggregated)
        out = F.relu(out)
        out = self.norm(out)
        return self.dropout(out)


class GraphAggregator(nn.Module):
    """Stacked message passing over chunk graph."""

    def __init__(
        self,
        hidden_size: int,
        num_edge_types: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TypedMessagePassingLayer(hidden_size, num_edge_types, dropout) for _ in range(num_layers)]
        )

    def forward(self, node_embeddings: torch.Tensor, edge_index: torch.Tensor, edge_types: torch.Tensor) -> torch.Tensor:
        x = node_embeddings
        for layer in self.layers:
            x = layer(x, edge_index=edge_index, edge_types=edge_types)
        return x


class ChunkPooler(nn.Module):
    """Pools chunk embeddings into one function vector."""

    def __init__(self, hidden_size: int, pooling: str = "mean") -> None:
        super().__init__()
        if pooling not in {"mean", "attention"}:
            raise ValueError(f"Unknown pooling strategy: {pooling}")
        self.pooling = pooling
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size, 1) if pooling == "attention" else None

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        if node_embeddings.numel() == 0:
            return torch.zeros(self.hidden_size, dtype=torch.float32, device=node_embeddings.device)

        if self.pooling == "mean":
            return node_embeddings.mean(dim=0)

        scores = self.attn(node_embeddings).squeeze(-1)
        weights = torch.softmax(scores, dim=0)
        return torch.sum(node_embeddings * weights.unsqueeze(-1), dim=0)


class FunctionEncoder(nn.Module):
    """
    Shared function encoder:
    chunk extraction -> chunk embeddings -> graph aggregation -> function pooling.
    """

    def __init__(
        self,
        model_name: str = "microsoft/unixcoder-base",
        max_chunk_tokens: int = 256,
        chunk_batch_size: int = 16,
        max_chunks: int = 128,
        use_graph: bool = False,
        gnn_layers: int = 2,
        pooling: str = "mean",
        projection_dim: int = 256,
        freeze_unixcoder: bool = True,
        graph_cache_size: int = 20000,
    ) -> None:
        super().__init__()
        self.chunk_batch_size = chunk_batch_size
        self.use_graph = use_graph
        self.graph_cache_size = max(0, graph_cache_size)
        self._graph_cache: Dict[str, FunctionGraph] = {}

        self.ts_parser = JavaTreeSitterParser()
        self.graph_builder = TreeSitterJavaChunkGraphBuilder(self.ts_parser, max_chunks=max_chunks)
        self.embedder = UniXCoderChunkEmbedder(
            model_name=model_name,
            max_length=max_chunk_tokens,
            freeze_backbone=freeze_unixcoder,
        )

        hidden_size = self.embedder.hidden_size
        self.graph_aggregator = (
            GraphAggregator(
                hidden_size=hidden_size,
                num_edge_types=len(EdgeType),
                num_layers=gnn_layers,
                dropout=0.1,
            )
            if use_graph
            else None
        )
        self.pooler = ChunkPooler(hidden_size=hidden_size, pooling=pooling)
        self.projection = (
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, projection_dim),
            )
            if projection_dim > 0
            else nn.Identity()
        )

    def _get_or_build_graph(self, code: str) -> FunctionGraph:
        if self.graph_cache_size and code in self._graph_cache:
            return self._graph_cache[code]

        graph = self.graph_builder.build(code)

        if self.graph_cache_size:
            if len(self._graph_cache) >= self.graph_cache_size:
                # Remove oldest inserted item (insertion-order dict in modern Python).
                oldest_key = next(iter(self._graph_cache))
                self._graph_cache.pop(oldest_key)
            self._graph_cache[code] = graph

        return graph

    def encode(self, codes: Sequence[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        graphs = [self._get_or_build_graph(code) for code in codes]

        all_texts = [chunk.text for graph in graphs for chunk in graph.chunks]
        chunk_vectors = self.embedder(all_texts, device=device, batch_size=self.chunk_batch_size)

        outputs: List[torch.Tensor] = []
        offset = 0
        for graph in graphs:
            n_chunks = len(graph.chunks)
            local_vectors = chunk_vectors[offset : offset + n_chunks]
            offset += n_chunks

            if self.use_graph and self.graph_aggregator is not None and graph.edge_index.numel() > 0:
                edge_index = graph.edge_index.to(device)
                edge_types = graph.edge_types.to(device)
                local_vectors = self.graph_aggregator(local_vectors, edge_index=edge_index, edge_types=edge_types)

            pooled = self.pooler(local_vectors)
            projected = self.projection(pooled)
            outputs.append(F.normalize(projected, p=2, dim=-1))

        return torch.stack(outputs, dim=0)

    def forward(self, codes: Sequence[str]) -> torch.Tensor:
        return self.encode(codes)


class SiameseCloneDetector(nn.Module):
    """Two-tower Siamese model with shared function encoder."""

    def __init__(self, encoder: FunctionEncoder) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, codes_a: Sequence[str], codes_b: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        z_a = self.encoder(codes_a)
        z_b = self.encoder(codes_b)
        return z_a, z_b

    @torch.no_grad()
    def embed_function(self, code: str) -> torch.Tensor:
        self.eval()
        return self.encoder([code])[0]

    @torch.no_grad()
    def similarity(self, code_a: str, code_b: str) -> float:
        self.eval()
        z_a = self.embed_function(code_a)
        z_b = self.embed_function(code_b)
        return float(F.cosine_similarity(z_a.unsqueeze(0), z_b.unsqueeze(0)).item())


def pairwise_contrastive_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Classic cosine contrastive loss for pair labels."""
    sims = F.cosine_similarity(z_a, z_b)
    labels = labels.float()
    pos = labels * (1.0 - sims).pow(2)
    neg = (1.0 - labels) * F.relu(sims - margin).pow(2)
    return (pos + neg).mean(), sims.detach()


def info_nce_loss_with_negatives(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    neg_margin: float = 0.4,
    neg_weight: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    InfoNCE on positive clone pairs + explicit margin penalty on non-clones.
    """
    labels = labels.float()
    sims = F.cosine_similarity(z_a, z_b)
    pos_idx = torch.nonzero(labels > 0.5, as_tuple=False).squeeze(-1)

    if pos_idx.numel() > 0:
        a = F.normalize(z_a, dim=-1)
        b = F.normalize(z_b, dim=-1)
        logits_ab = (a[pos_idx] @ b.T) / temperature
        logits_ba = (b[pos_idx] @ a.T) / temperature
        targets = pos_idx.to(torch.long)
        nce_ab = F.cross_entropy(logits_ab, targets)
        nce_ba = F.cross_entropy(logits_ba, targets)
        nce = 0.5 * (nce_ab + nce_ba)
    else:
        nce = torch.zeros((), dtype=z_a.dtype, device=z_a.device)

    neg_penalty = ((1.0 - labels) * F.relu(sims - neg_margin).pow(2)).mean()
    loss = nce + neg_weight * neg_penalty
    return loss, sims.detach()


def batchify(samples: Sequence[CodeSample], batch_size: int) -> Iterator[List[CodeSample]]:
    for start in range(0, len(samples), batch_size):
        yield list(samples[start : start + batch_size])


def load_samples(split: str, max_samples: Optional[int], seed: int, shuffle: bool = True) -> List[CodeSample]:
    generator = bigclonebench_hf_generator(split=split)
    if max_samples is not None:
        generator = itertools.islice(generator, max_samples)
    samples = list(generator)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(samples)
    return samples


@dataclass
class TrainConfig:
    epochs: int = 1
    batch_size: int = 8
    lr: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    loss_type: str = "contrastive"  # "contrastive" or "infonce"
    margin: float = 0.4
    temperature: float = 0.07
    neg_weight: float = 0.25
    threshold: float = 0.75


def evaluate(
    model: SiameseCloneDetector,
    samples: Sequence[CodeSample],
    batch_size: int,
    threshold: float,
    margin: float = 0.4,
) -> Dict[str, float]:
    model.eval()
    device = next(model.parameters()).device

    all_labels: List[float] = []
    all_sims: List[float] = []
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for batch in batchify(samples, batch_size=batch_size):
            codes_a = [s.code_a for s in batch]
            codes_b = [s.code_b or "" for s in batch]
            labels = torch.tensor([s.label for s in batch], dtype=torch.float32, device=device)

            z_a, z_b = model(codes_a, codes_b)
            loss, sims = pairwise_contrastive_loss(z_a, z_b, labels=labels, margin=margin)
            total_loss += float(loss.item()) * len(batch)
            total_items += len(batch)

            all_labels.extend(labels.cpu().tolist())
            all_sims.extend(sims.cpu().tolist())

    if not all_labels:
        return {"loss": 0.0, "accuracy": 0.0, "mean_similarity": 0.0}

    preds = [1.0 if sim >= threshold else 0.0 for sim in all_sims]
    correct = sum(float(p == y) for p, y in zip(preds, all_labels))
    acc = correct / len(all_labels)

    return {
        "loss": total_loss / max(total_items, 1),
        "accuracy": acc,
        "mean_similarity": float(sum(all_sims) / len(all_sims)),
    }


def train(
    model: SiameseCloneDetector,
    train_samples: Sequence[CodeSample],
    config: TrainConfig,
    val_samples: Optional[Sequence[CodeSample]] = None,
) -> List[Dict[str, float]]:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters. Disable freeze or enable projection/GNN layers.")

    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    device = next(model.parameters()).device
    history: List[Dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        running_items = 0

        random.shuffle(train_samples)  # in-place epoch shuffle

        for batch in batchify(train_samples, batch_size=config.batch_size):
            codes_a = [s.code_a for s in batch]
            codes_b = [s.code_b or "" for s in batch]
            labels = torch.tensor([s.label for s in batch], dtype=torch.float32, device=device)

            z_a, z_b = model(codes_a, codes_b)

            if config.loss_type == "infonce":
                loss, _ = info_nce_loss_with_negatives(
                    z_a,
                    z_b,
                    labels=labels,
                    temperature=config.temperature,
                    neg_margin=config.margin,
                    neg_weight=config.neg_weight,
                )
            else:
                loss, _ = pairwise_contrastive_loss(z_a, z_b, labels=labels, margin=config.margin)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=config.max_grad_norm)
            optimizer.step()

            running_loss += float(loss.item()) * len(batch)
            running_items += len(batch)

        train_loss = running_loss / max(running_items, 1)
        epoch_log: Dict[str, float] = {"epoch": float(epoch), "train_loss": train_loss}

        if val_samples:
            val_metrics = evaluate(
                model=model,
                samples=val_samples,
                batch_size=config.batch_size,
                threshold=config.threshold,
                margin=config.margin,
            )
            epoch_log.update(
                {
                    "val_loss": float(val_metrics["loss"]),
                    "val_accuracy": float(val_metrics["accuracy"]),
                    "val_mean_similarity": float(val_metrics["mean_similarity"]),
                }
            )

        history.append(epoch_log)
        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"train_loss={train_loss:.4f}"
            + (
                f" | val_loss={epoch_log['val_loss']:.4f} "
                f"| val_acc={epoch_log['val_accuracy']:.4f}"
                if "val_loss" in epoch_log
                else ""
            )
        )

    return history


@torch.no_grad()
def predict_clone(
    model: SiameseCloneDetector,
    code_a: str,
    code_b: str,
    threshold: float = 0.75,
) -> Tuple[int, float]:
    sim = model.similarity(code_a, code_b)
    pred = 1 if sim >= threshold else 0
    return pred, sim


def save_checkpoint(path: str, model: SiameseCloneDetector, args: argparse.Namespace) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "args": vars(args),
    }
    torch.save(payload, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tree-sitter + UniXCoder Siamese clone training")
    parser.add_argument("--model-name", type=str, default="microsoft/unixcoder-base")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--use-graph", action="store_true", help="Enable AST graph message passing")
    parser.add_argument("--pooling", choices=["mean", "attention"], default="mean")
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--max-chunks", type=int, default=128)
    parser.add_argument("--max-chunk-tokens", type=int, default=256)
    parser.add_argument("--chunk-batch-size", type=int, default=16)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--freeze-unixcoder", action="store_true")
    parser.add_argument("--graph-cache-size", type=int, default=20000)

    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="validation")
    parser.add_argument("--max-train-samples", type=int, default=2000)
    parser.add_argument("--max-val-samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--loss-type", choices=["contrastive", "infonce"], default="contrastive")
    parser.add_argument("--margin", type=float, default=0.4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--neg-weight", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--save-path", type=str, default="models/siamese_unixcoder_treesitter.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading CodeXGLUE BigCloneBench samples...")
    train_samples = load_samples(
        split=args.train_split,
        max_samples=args.max_train_samples,
        seed=args.seed,
        shuffle=True,
    )
    val_samples = load_samples(
        split=args.val_split,
        max_samples=args.max_val_samples,
        seed=args.seed + 1,
        shuffle=False,
    )
    print(f"Train samples: {len(train_samples)} | Validation samples: {len(val_samples)}")

    encoder = FunctionEncoder(
        model_name=args.model_name,
        max_chunk_tokens=args.max_chunk_tokens,
        chunk_batch_size=args.chunk_batch_size,
        max_chunks=args.max_chunks,
        use_graph=args.use_graph,
        gnn_layers=args.gnn_layers,
        pooling=args.pooling,
        projection_dim=args.projection_dim,
        freeze_unixcoder=args.freeze_unixcoder,
        graph_cache_size=args.graph_cache_size,
    )
    model = SiameseCloneDetector(encoder).to(torch.device(args.device))

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss_type=args.loss_type,
        margin=args.margin,
        temperature=args.temperature,
        neg_weight=args.neg_weight,
        threshold=args.threshold,
    )

    if args.epochs > 0 and train_samples:
        train(model=model, train_samples=train_samples, config=config, val_samples=val_samples)

    metrics = evaluate(
        model=model,
        samples=val_samples,
        batch_size=args.batch_size,
        threshold=args.threshold,
        margin=args.margin,
    )
    print(
        "Validation metrics | "
        f"loss={metrics['loss']:.4f} | "
        f"accuracy={metrics['accuracy']:.4f} | "
        f"mean_similarity={metrics['mean_similarity']:.4f}"
    )

    save_checkpoint(path=args.save_path, model=model, args=args)
    print(f"Saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
