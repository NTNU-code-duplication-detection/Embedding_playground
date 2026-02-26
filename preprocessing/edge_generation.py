"""
JAVA -> AST-CHUNK GRAPH -> EDGE TENSORS

Input:
  java_source: str

Output:
  nodes: List[dict]   (each node has id, type, ast_type, code, tokens, ...)
  edge_index: torch.LongTensor [2, E]
  edge_type:  torch.LongTensor [E]
  (optional) node_ids already 0..N-1, so node row i corresponds to node id i.

Edges generated here:
  - AST_CHILD / AST_PARENT between *selected chunk nodes* (control + straight)
  - AST_SIBLING_NEXT / AST_SIBLING_PREV between chunk siblings in source order

No embeddings here. This is purely the AST edge generation and tensorization.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import tree_sitter_java
from tree_sitter import Language, Parser


# ----------------------------
# 0) Tree-sitter Java setup
# ----------------------------

JAVA_LANGUAGE = Language(tree_sitter_java.language())
parser = Parser(JAVA_LANGUAGE)


# ----------------------------
# 1) Chunk selection policy (what becomes a node)
# ----------------------------

CONTROL_NODES = (
    "if_statement",
    "for_statement",
    "while_statement",
    "do_statement",
    "switch_statement",
    "try_statement",
    "catch_clause",
)

STRAIGHT_NODES = (
    "local_variable_declaration",
    "expression_statement",
    "return_statement",
    "throw_statement",
)


# ----------------------------
# 2) Graph schema
# ----------------------------

class ChunkKind(Enum):
    STRAIGHT = "straight"
    CONTROL = "control"


@dataclass
class GraphNode:
    id: int
    type: str                 # "Control" or "Statement" (you can extend)
    ast_type: str             # raw tree-sitter node.type
    code: str                 # snippet to embed later
    tokens: Optional[List[str]] = None

    start_byte: Optional[int] = None
    end_byte: Optional[int] = None
    start_point: Optional[Tuple[int, int]] = None
    end_point: Optional[Tuple[int, int]] = None
    ast_depth: Optional[int] = None
    kind: Optional[str] = None


@dataclass
class GraphEdge:
    src: int
    dst: int
    type: str


EDGE_TYPE_TO_ID = {
    "AST_CHILD": 0,
    "AST_PARENT": 1,
    "AST_SIBLING_NEXT": 2,
    "AST_SIBLING_PREV": 3,
    "AST_NEXT": 4,
    "AST_PREV": 5,
    "IF_THEN": 6,
    "IF_THEN_REV": 7,
    "IF_ELSE": 8,
    "IF_ELSE_REV": 9,
}

# ----------------------------
# 3) Helpers
# ----------------------------

def _safe_text(code_bytes: bytes, node) -> str:
    return code_bytes[node.start_byte:node.end_byte].decode("utf8", errors="replace").strip()


def _simple_tokenize(text: str) -> List[str]:
    out: List[str] = []
    buf: List[str] = []
    seps = set("(){}[];,.+-*/%=&|!<>?:~^")
    for ch in text:
        if ch.isspace():
            if buf:
                out.append("".join(buf))
                buf = []
            continue
        if ch in seps:
            if buf:
                out.append("".join(buf))
                buf = []
            out.append(ch)
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out


def extract_control_header(code: bytes, node) -> str:
    """
    For control nodes, we often want just the header (condition/signature).
    This is a heuristic: adapt it as needed.
    """
    # Try common patterns first
    for child in node.children:
        if child.type in ("condition", "parenthesized_expression"):
            return code[child.start_byte:child.end_byte].decode("utf8", errors="replace").strip()

    # Fallback: until first '{' if present
    text = code[node.start_byte:node.end_byte].decode("utf8", errors="replace")
    return text.split("{")[0].strip()


# ----------------------------
# 4) Builder: Java code -> nodes + edges
# ----------------------------

class JavaAstChunkGraph:
    def __init__(self, java_source: str):
        self.code_bytes = java_source.encode("utf8")
        self.nodes: List[GraphNode] = []
        self.edges: List[GraphEdge] = []
        self._next_id = 0
        self._tsnode_to_graphid: Dict[int, int] = {}
        self._span_to_graphid: Dict[Tuple[int, int, str], int] = {}

    def build(self) -> Dict[str, Any]:
        tree = parser.parse(self.code_bytes)
        self._visit(tree.root_node, depth=0, parent_selected_id=None)

        self._add_if_role_edges(add_reverse=True)
        self._add_sibling_edges()
        self._add_next_prev_edges()

        return {
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
        }

    def _visit(self, root_node, depth: int, parent_selected_id: Optional[int]):
        """
        Iterative DFS version to avoid RecursionError.
        Preserves the same semantics as the recursive _visit.
        """
        # stack items: (ts_node, depth, parent_selected_id)
        stack: List[Tuple[Any, int, Optional[int]]] = [(root_node, depth, parent_selected_id)]

        while stack:
            node, d, parent_sel = stack.pop()

            node_type = node.type
            is_control = node_type in CONTROL_NODES
            is_straight = node_type in STRAIGHT_NODES
            is_selected = is_control or is_straight

            current_selected_id: Optional[int] = None
            next_parent_sel = parent_sel

            if is_selected:
                if is_control:
                    code = extract_control_header(self.code_bytes, node)
                    high_type = "Control"
                    kind = ChunkKind.CONTROL.value
                else:
                    code = _safe_text(self.code_bytes, node)
                    high_type = "Statement"
                    kind = ChunkKind.STRAIGHT.value

                if code:
                    current_selected_id = self._add_node(node, d, high_type, node_type, code, kind)

                    # Parent-child edges between selected nodes
                    if parent_sel is not None:
                        self.edges.append(GraphEdge(src=parent_sel, dst=current_selected_id, type="AST_CHILD"))
                        self.edges.append(GraphEdge(src=current_selected_id, dst=parent_sel, type="AST_PARENT"))

                    # This selected node becomes the parent_selected_id for descendants
                    next_parent_sel = current_selected_id

            # Push children in reverse order so traversal order is roughly preserved
            # (not strictly required, but good for deterministic debugging)
            for child in reversed(node.children):
                if len(child.children) == 0 and child.type in (";", "{", "}", "(", ")", ","):
                    continue
                stack.append((child, d + 1, next_parent_sel))

    def _add_node(self, node, depth: int, high_type: str, ast_type: str, code: str, kind: str) -> int:
        node_id = self._next_id
        self._next_id += 1
        self.nodes.append(
            GraphNode(
                id=node_id,
                type=high_type,
                ast_type=ast_type,
                code=code,
                tokens=_simple_tokenize(code),
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                start_point=(node.start_point[0], node.start_point[1]) if node.start_point else None,
                end_point=(node.end_point[0], node.end_point[1]) if node.end_point else None,
                ast_depth=depth,
                kind=kind,
            )
        )
        self._tsnode_to_graphid[node.id] = node_id
        self._span_to_graphid[(node.start_byte, node.end_byte, node.type)] = node_id
        return node_id

    def _add_sibling_edges(self):
        """
        Adds sibling edges among selected nodes under the same selected parent.
        We derive sibling groups from AST_CHILD edges.
        """
        children_by_parent: Dict[int, List[int]] = {}
        for e in self.edges:
            if e.type == "AST_CHILD":
                children_by_parent.setdefault(e.src, []).append(e.dst)

        id_to_start = {n.id: (n.start_byte if n.start_byte is not None else 10**18) for n in self.nodes}

        for parent_id, child_ids in children_by_parent.items():
            # NEW: skip sibling edges under if_statement (prevents then->else chaining)
            parent_ast_type = self.nodes[parent_id].ast_type
            if parent_ast_type == "if_statement":
                continue

            child_ids_sorted = sorted(child_ids, key=lambda cid: id_to_start.get(cid, 10**18))
            for i in range(len(child_ids_sorted) - 1):
                a = child_ids_sorted[i]
                b = child_ids_sorted[i + 1]
                self.edges.append(GraphEdge(src=a, dst=b, type="AST_SIBLING_NEXT"))
                self.edges.append(GraphEdge(src=b, dst=a, type="AST_SIBLING_PREV"))

    def _add_next_prev_edges(self):
        """
        Adds linear edges ONLY among "top-level" selected nodes.

        Heuristic:
        - find min ast_depth among selected nodes (usually block statement depth)
        - only connect nodes at that min depth in source order
        """
        if len(self.nodes) <= 1:
            return

        # Determine "top-level" depth among selected nodes
        depths = [n.ast_depth for n in self.nodes if n.ast_depth is not None]
        if not depths:
            return
        top_depth = min(depths)

        # Keep only top-level nodes
        top_nodes = [n for n in self.nodes if n.ast_depth == top_depth]

        if len(top_nodes) <= 1:
            return

        top_nodes_sorted = sorted(
            top_nodes,
            key=lambda n: (n.start_byte if n.start_byte is not None else 10**18,
                        n.end_byte if n.end_byte is not None else 10**18),
        )

        for a, b in zip(top_nodes_sorted, top_nodes_sorted[1:]):
            # Overlap guard (shouldn't happen here, but safe)
            if a.end_byte is not None and b.start_byte is not None and a.end_byte > b.start_byte:
                continue

            self.edges.append(GraphEdge(src=a.id, dst=b.id, type="AST_NEXT"))
            self.edges.append(GraphEdge(src=b.id, dst=a.id, type="AST_PREV"))

    def _selected_nodes_in_span(self, start_byte: int, end_byte: int) -> List[int]:
        """
        Return ALL selected node ids whose byte span is fully inside [start_byte, end_byte].
        This is what you need for block bodies { ... } that contain multiple statements.
        """
        ids: List[int] = []
        for n in self.nodes:
            if n.start_byte is None or n.end_byte is None:
                continue
            if n.start_byte >= start_byte and n.end_byte <= end_byte:
                ids.append(n.id)
        return ids


    def _add_if_role_edges(self, add_reverse: bool = True):
        tree = parser.parse(self.code_bytes)

        stack = [tree.root_node]
        while stack:
            ts_node = stack.pop()

            if ts_node.type == "if_statement":
                if_id = self._span_to_graphid.get(
                    (ts_node.start_byte, ts_node.end_byte, ts_node.type),
                    None,
                )

                if if_id is not None:
                    then_node = ts_node.child_by_field_name("consequence")
                    else_node = ts_node.child_by_field_name("alternative")

                    if then_node is not None:
                        then_ids = self._selected_nodes_in_span(then_node.start_byte, then_node.end_byte)
                        for tid in then_ids:
                            if tid == if_id:
                                continue
                            self.edges.append(GraphEdge(src=if_id, dst=tid, type="IF_THEN"))
                            if add_reverse:
                                self.edges.append(GraphEdge(src=tid, dst=if_id, type="IF_THEN_REV"))

                    if else_node is not None:
                        else_ids = self._selected_nodes_in_span(else_node.start_byte, else_node.end_byte)
                        for eid in else_ids:
                            if eid == if_id:
                                continue
                            self.edges.append(GraphEdge(src=if_id, dst=eid, type="IF_ELSE"))
                            if add_reverse:
                                self.edges.append(GraphEdge(src=eid, dst=if_id, type="IF_ELSE_REV"))

            # push children (reverse order optional)
            for ch in reversed(ts_node.children):
                if len(ch.children) == 0 and ch.type in (";", "{", "}", "(", ")", ","):
                    continue
                stack.append(ch)

# ----------------------------
# 5) Convert edges to tensors
# ----------------------------

def edges_to_tensors(edges: List[Dict[str, Any]], edge_type_to_id: Dict[str, int] = EDGE_TYPE_TO_ID) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      edge_index: [2, E] long
      edge_type:  [E] long
    """
    src: List[int] = []
    dst: List[int] = []
    et: List[int] = []

    for e in edges:
        t = e["type"]
        if t not in edge_type_to_id:
            continue
        src.append(int(e["src"]))
        dst.append(int(e["dst"]))
        et.append(int(edge_type_to_id[t]))

    if len(src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_type = torch.tensor(et, dtype=torch.long)

    return edge_index, edge_type


def java_to_nodes_and_edge_tensors(java_source: str) -> Tuple[List[Dict[str, Any]], torch.Tensor, torch.Tensor]:
    """
    Main function you asked for:
      java code -> nodes list + edge_index tensor + edge_type tensor
    """
    graph = JavaAstChunkGraph(java_source).build()
    nodes = graph["nodes"]
    edges = graph["edges"]
    edge_index, edge_type = edges_to_tensors(edges)
    return nodes, edge_index, edge_type


# ----------------------------
# 6) Quick test
# ----------------------------

if __name__ == "__main__":
    JAVA = """
class DemoBlockIf {
    int foo(int a, int b) {
        int x = a;

        if (a > b) {
            x = x + 10;
            x = x + 1;
        } else {
            x = x - 10;
            x = x - 1;
        }

        return x;
    }
}
    """.strip()

    nodes, edge_index, edge_type = java_to_nodes_and_edge_tensors(JAVA)
    print("Code:")
    print(JAVA)
    print("N nodes =", len(nodes))
    print("edge_index shape =", tuple(edge_index.shape))
    print("edge_type shape  =", tuple(edge_type.shape))

    # Print a compact view
    for n in nodes:
        code = n["code"].replace("\n", " ")
        if len(code) > 60:
            code = code[:57] + "..."
        print(f'node {n["id"]:>2} {n["type"]:<9} ast={n["ast_type"]:<25} depth={n["ast_depth"]} code="{code}"')

    print("\nEdges:")
    # show first ~30 edges
    for i in range(min(edge_index.shape[1], 300)):
        s = int(edge_index[0, i])
        d = int(edge_index[1, i])
        t = int(edge_type[i])
        inv = {v: k for k, v in EDGE_TYPE_TO_ID.items()}
        print(f"{inv[t]:<16} {s} -> {d}")
