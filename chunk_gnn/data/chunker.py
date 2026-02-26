"""
Tree-sitter based Java chunker for the chunk-level GNN.

Adapted from the TreeSitterJavaChunker in the weighted_sum notebooks (V2,
header-only variant). Key improvements for chunk-GNN:

  1. Tracks parent-child relationships between chunks (for graph edges)
  2. Adds enhanced_for_statement to CONTROL_NODES (bug fix)
  3. Normalizes depth relative to method body (not CST root)
  4. Records start/end lines for debugging
  5. Returns chunks in DFS order (natural sequential ordering for edges)

Design decisions:
  - Uses V2 (header-only) for CONTROL chunks to avoid text overlap
  - STRAIGHT chunks get full statement text
  - Empty chunks (from parse errors or empty methods) return None
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

log = logging.getLogger(__name__)

# Initialize tree-sitter Java parser (module-level singleton)
_JAVA_LANGUAGE = Language(tsjava.language())
_parser = Parser(_JAVA_LANGUAGE)

# ---------------------------------------------------------------------------
# Node type classification
# ---------------------------------------------------------------------------

CONTROL_NODES = frozenset({
    "if_statement",
    "for_statement",
    "enhanced_for_statement",  # for-each loops — missing in notebook version
    "while_statement",
    "do_statement",
    "switch_statement",
    "try_statement",
    "catch_clause",
})

STRAIGHT_NODES = frozenset({
    "local_variable_declaration",
    "expression_statement",
    "return_statement",
    "throw_statement",
})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ChunkKind(Enum):
    """Classification of code chunks as straight-line or control-flow."""
    STRAIGHT = "straight"
    CONTROL = "control"


@dataclass
class Chunk:
    """A single code chunk (node in the chunk-level graph)."""

    text: str
    kind: ChunkKind
    depth: int              # Normalized depth (relative to method body)
    start_line: int
    end_line: int
    parent_index: int | None  # Index of parent chunk in the chunks list, or None


# ---------------------------------------------------------------------------
# Header extraction for control nodes
# ---------------------------------------------------------------------------

def _extract_control_header(code_bytes: bytes, node) -> str:
    """Extract only the condition/header of a control statement.

    For `if (x > 0) { ... }` returns `if (x > 0)`.
    For `for (int i = 0; i < n; i++) { ... }` returns `for (int i = 0; i < n; i++)`.
    For `try { ... }` returns `try`.

    This avoids text overlap between control chunks and their child
    statement chunks.
    """
    # Strategy: find the body block and take everything before it
    for child in node.children:
        if child.type == "block":
            header_text = code_bytes[node.start_byte:child.start_byte].decode(
                "utf8", errors="replace"
            ).strip()
            if header_text:
                return header_text

    # Fallback for single-statement bodies (no block, e.g. `if (x) return;`)
    # Take text up to the first child that is a statement
    for child in node.children:
        if child.type in CONTROL_NODES or child.type in STRAIGHT_NODES:
            header_text = code_bytes[node.start_byte:child.start_byte].decode(
                "utf8", errors="replace"
            ).strip()
            if header_text:
                return header_text

    # Last resort: first line
    full_text = code_bytes[node.start_byte:node.end_byte].decode(
        "utf8", errors="replace"
    )
    first_line = full_text.split("\n")[0].strip()
    return first_line


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

def _find_body_depth(root_node) -> int:
    """Find the CST depth of the method body's block node.

    BCB files are bare methods, so the tree looks like:
      program -> method_declaration -> block -> statements...
    or with errors:
      program -> ERROR -> ... -> block -> statements...

    We find the first `block` node and return its depth, so we can
    normalize chunk depths relative to it.
    """
    # BFS to find the first block that likely represents the method body
    queue = [(root_node, 0)]
    while queue:
        node, depth = queue.pop(0)
        # The method body block is typically a child of method_declaration
        # or constructor_declaration
        if node.type == "block" and node.parent and node.parent.type in (
            "method_declaration",
            "constructor_declaration",
            # Error recovery: sometimes tree-sitter wraps in ERROR
        ):
            return depth
        # Also accept if parent is a program-level node (bare block)
        if node.type == "block" and depth <= 3:
            return depth
        for child in node.children:
            queue.append((child, depth + 1))

    # If no block found, use depth 0 as baseline
    return 0


class TreeSitterChunker:
    """Chunks a Java function into STRAIGHT and CONTROL segments.

    Each chunk becomes a node in the chunk-level graph. Parent-child
    relationships between chunks are tracked for graph edge construction.
    """

    def __init__(self) -> None:
        # State is per-call, not per-instance, so the chunker is reusable
        pass

    def chunk_function(
        self, source: str, max_chunks: int = 50
    ) -> list[Chunk] | None:
        """Parse a Java function and return ordered list of chunks.

        Args:
            source: Java source code (single method, no class wrapper needed)
            max_chunks: Maximum chunks to return (truncate if exceeded)

        Returns:
            List of Chunk objects in DFS order, or None if parsing fails
            or produces no chunks.
        """
        code_bytes = source.encode("utf-8")
        tree = _parser.parse(code_bytes)

        if tree.root_node.child_count == 0:
            log.debug("Tree-sitter produced empty tree")
            return None

        # Find the method body depth for normalization
        body_depth = _find_body_depth(tree.root_node)

        # Extract chunks via DFS
        chunks: list[Chunk] = []
        self._visit(
            node=tree.root_node,
            raw_depth=0,
            body_depth=body_depth,
            code_bytes=code_bytes,
            chunks=chunks,
            parent_chunk_index=None,
        )

        if not chunks:
            log.debug("No chunks extracted (empty or unparseable method)")
            return None

        # Truncate if too many chunks
        if len(chunks) > max_chunks:
            log.debug(
                "Truncating %d chunks to %d", len(chunks), max_chunks
            )
            chunks = chunks[:max_chunks]

        return chunks

    def _visit(
        self,
        node,
        raw_depth: int,
        body_depth: int,
        code_bytes: bytes,
        chunks: list[Chunk],
        parent_chunk_index: int | None,
    ) -> None:
        """DFS traversal that extracts chunks and tracks parent relationships."""

        current_chunk_index = None
        norm_depth = max(0, raw_depth - body_depth)

        if node.type in CONTROL_NODES:
            header = _extract_control_header(code_bytes, node)
            if header:
                current_chunk_index = len(chunks)
                chunks.append(
                    Chunk(
                        text=header,
                        kind=ChunkKind.CONTROL,
                        depth=norm_depth,
                        start_line=node.start_point[0],
                        end_line=node.end_point[0],
                        parent_index=parent_chunk_index,
                    )
                )
                # Children of this control node get this chunk as parent
                parent_chunk_index = current_chunk_index

        elif node.type in STRAIGHT_NODES:
            text = code_bytes[node.start_byte:node.end_byte].decode(
                "utf-8", errors="replace"
            ).strip()
            if text:
                current_chunk_index = len(chunks)
                chunks.append(
                    Chunk(
                        text=text,
                        kind=ChunkKind.STRAIGHT,
                        depth=norm_depth,
                        start_line=node.start_point[0],
                        end_line=node.end_point[0],
                        parent_index=parent_chunk_index,
                    )
                )

        # Recurse into children
        for child in node.children:
            self._visit(
                node=child,
                raw_depth=raw_depth + 1,
                body_depth=body_depth,
                code_bytes=code_bytes,
                chunks=chunks,
                parent_chunk_index=parent_chunk_index,
            )
