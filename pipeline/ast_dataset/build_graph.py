"""Module pipeline/ast_dataset/build_graph.py."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from tree_sitter import Node

# ----------------------------
# Chunk selection policy
# ----------------------------

CONTROL_NODES = {
    "if_statement",
    "for_statement",
    "while_statement",
    "do_statement",
    "switch_statement",
    "try_statement",
    "catch_clause",
}

STRAIGHT_NODES = {
    "local_variable_declaration",
    "expression_statement",
    "return_statement",
    "throw_statement",
}

# Minimal edge type IDs (optional; keep as strings in JSON if you prefer)
EDGE_TYPE_TO_ID = {
    "SEQ": 0,
    "AST": 1,
    "IF_THEN": 2,
    "IF_ELSE": 3,
}


# ----------------------------
# Helpers
# ----------------------------

def _text(code: bytes, node: Node) -> str:
    return code[node.start_byte:node.end_byte].decode("utf-8", errors="replace").strip()


def _is_punct_leaf(n: Node) -> bool:
    return (len(n.children) == 0) and (n.type in (";", "{", "}", "(", ")", ",", "."))


def extract_control_header(code: bytes, node: Node) -> str:
    """
    Header-only text for control nodes.
    Heuristics:
      - prefer condition/parenthesized_expression if present
      - otherwise, take text up to first '{'
    """
    for child in node.children:
        if child.type in ("condition", "parenthesized_expression"):
            return _text(code, child)

    full = _text(code, node)
    if "{" in full:
        return full.split("{", 1)[0].strip()
    return full


def _span_key(node: Node) -> Tuple[int, int, str]:
    return (node.start_byte, node.end_byte, node.type)


def _node_in_span(n_start: int, n_end: int, span_start: int, span_end: int) -> bool:
    return (n_start >= span_start) and (n_end <= span_end)


# ----------------------------
# Graph builder
# ----------------------------

@dataclass
class ChunkNode:
    id: int
    kind: str         # "control" or "straight"
    ast_type: str     # raw tree-sitter node.type
    code: str
    start_byte: int
    end_byte: int
    depth: int


def build_method_chunk_graph(
    method_node: Node,
    file_bytes: bytes,
    undirected: bool = True,
    add_if_role_edges: bool = True,
    seq_top_level_only: bool = True,
) -> Dict[str, object]:
    """
    Build a chunk graph for a single method/constructor node.

    Returns:
      nodes: List[dict]
      edges: List[dict]  where dict has src,dst,type
    """

    nodes: List[ChunkNode] = []
    edges: List[Dict[str, object]] = []

    # Maps for role edge lookup
    span_to_id: Dict[Tuple[int, int, str], int] = {}

    def add_edge(src: int, dst: int, etype: str):
        edges.append({"src": src, "dst": dst, "type": etype})
        if undirected and src != dst:
            edges.append({"src": dst, "dst": src, "type": etype})

    def add_node(ts_node: Node, depth: int, kind: str, code: str) -> int:
        nid = len(nodes)
        nodes.append(
            ChunkNode(
                id=nid,
                kind=kind,
                ast_type=ts_node.type,
                code=code,
                start_byte=ts_node.start_byte,
                end_byte=ts_node.end_byte,
                depth=depth,
            )
        )
        span_to_id[_span_key(ts_node)] = nid
        return nid

    # Iterative DFS over the METHOD subtree
    # parent_sel_id tracks nearest selected ancestor (for AST edges)
    stack: List[Tuple[Node, int, Optional[int]]] = [(method_node, 0, None)]

    while stack:
        ts_node, depth, parent_sel = stack.pop()

        t = ts_node.type
        is_control = t in CONTROL_NODES
        is_straight = t in STRAIGHT_NODES
        is_selected = is_control or is_straight

        current_sel: Optional[int] = None
        next_parent_sel = parent_sel

        if is_selected:
            if is_control:
                code = extract_control_header(file_bytes, ts_node)
                kind = "control"
            else:
                code = _text(file_bytes, ts_node)
                kind = "straight"

            if code:
                current_sel = add_node(ts_node, depth, kind, code)

                # AST containment (undirected) between selected nodes                
                if parent_sel is not None:
                    parent_is_if = nodes[parent_sel].ast_type == "if_statement"
                    if not parent_is_if:
                        add_edge(parent_sel, current_sel, "AST")

                next_parent_sel = current_sel

        # push children
        for ch in reversed(ts_node.children):
            if _is_punct_leaf(ch):
                continue
            stack.append((ch, depth + 1, next_parent_sel))

    # SEQ edges: connect "top-level" selected nodes in source order
    if len(nodes) >= 2:
        if seq_top_level_only:
            top_depth = min(n.depth for n in nodes)
            top_nodes = [n for n in nodes if n.depth == top_depth]
        else:
            top_nodes = list(nodes)

        top_nodes = sorted(top_nodes, key=lambda n: (n.start_byte, n.end_byte))
        for a, b in zip(top_nodes, top_nodes[1:]):
            # overlap guard
            if a.end_byte > b.start_byte:
                continue
            add_edge(a.id, b.id, "SEQ")

    # IF role edges (then/else)
    if add_if_role_edges and nodes:
        # helper: find all selected node ids fully inside a span
        def selected_ids_in_span(span_start: int, span_end: int) -> List[int]:
            out: List[int] = []
            for n in nodes:
                if _node_in_span(n.start_byte, n.end_byte, span_start, span_end):
                    out.append(n.id)
            return out

        # Traverse only method subtree again, look for if_statement
        stack2 = [method_node]
        while stack2:
            n = stack2.pop()
            if n.type == "if_statement":
                if_id = span_to_id.get(_span_key(n))
                if if_id is not None:
                    then_node = n.child_by_field_name("consequence")
                    else_node = n.child_by_field_name("alternative")

                    if then_node is not None:
                        for tid in selected_ids_in_span(then_node.start_byte, then_node.end_byte):
                            if tid != if_id:
                                add_edge(if_id, tid, "IF_THEN")

                    if else_node is not None:
                        for eid in selected_ids_in_span(else_node.start_byte, else_node.end_byte):
                            if eid != if_id:
                                add_edge(if_id, eid, "IF_ELSE")

            for ch in reversed(n.children):
                if _is_punct_leaf(ch):
                    continue
                stack2.append(ch)

    return {
        "nodes": [
            {
                "id": n.id,
                "kind": n.kind,
                "ast_type": n.ast_type,
                "code": n.code,
                "start_byte": n.start_byte,
                "end_byte": n.end_byte,
                "depth": n.depth,
            }
            for n in nodes
        ],
        "edges": edges,
    }
