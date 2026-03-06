"""AST statement graph extraction with optional edge construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from tree_sitter import Language, Node, Parser
import tree_sitter_java

from config import ASTConfig


JAVA_LANGUAGE = Language(tree_sitter_java.language())

EDGE_TYPES = ("AST", "SEQ", "IF_THEN", "IF_ELSE", "DATA_FLOW")

CONTROL_NODES = {
    "if_statement",
    "for_statement",
    "while_statement",
    "do_statement",
    "switch_statement",
    "try_statement",
    "catch_clause",
    "enhanced_for_statement",
}

STRAIGHT_NODES = {
    "local_variable_declaration",
    "expression_statement",
    "return_statement",
    "throw_statement",
    "break_statement",
    "continue_statement",
    "assert_statement",
}

PUNCT_LEAF_TYPES = {";", "{", "}", "(", ")", ",", "."}


@dataclass
class StatementNode:
    """Selected statement node tracked in a method graph."""

    id: int
    kind: str
    ast_type: str
    code: str
    start_byte: int
    end_byte: int
    depth: int


class ASTGraphBuilder:
    """Build method-level statement graphs from Java source."""

    def __init__(self, cfg: ASTConfig):
        self.cfg = cfg
        self.parser = Parser(JAVA_LANGUAGE)
        self.enabled = set(cfg.enabled_edge_types)

    @staticmethod
    def edge_type_to_id(enabled_edge_types: Iterable[str]) -> dict[str, int]:
        """Edge type to numeric id mapping for tensor conversion."""

        uniq = []
        for edge_type in EDGE_TYPES:
            if edge_type in enabled_edge_types:
                uniq.append(edge_type)
        return {edge_type: i for i, edge_type in enumerate(uniq)}

    @staticmethod
    def _node_text(code: bytes, node: Node) -> str:
        return code[node.start_byte : node.end_byte].decode("utf-8", errors="replace").strip()

    @staticmethod
    def _is_punct_leaf(node: Node) -> bool:
        return (len(node.children) == 0) and (node.type in PUNCT_LEAF_TYPES)

    def _extract_control_header(self, code: bytes, node: Node) -> str:
        for child in node.children:
            if child.type in ("condition", "parenthesized_expression"):
                return self._node_text(code, child)

        full = self._node_text(code, node)
        return full.split("{", 1)[0].strip() if "{" in full else full

    @staticmethod
    def _span_key(node: Node) -> tuple[int, int, str]:
        return (node.start_byte, node.end_byte, node.type)

    @staticmethod
    def _find_descendants(node: Node, node_type: str) -> list[Node]:
        stack = [node]
        found: list[Node] = []
        while stack:
            current = stack.pop()
            if current.type == node_type:
                found.append(current)
            for child in reversed(current.children):
                stack.append(child)
        return found

    def _identifier_names(self, code: bytes, node: Optional[Node]) -> list[str]:
        if node is None:
            return []
        names: list[str] = []
        stack = [node]
        while stack:
            current = stack.pop()
            if current.type == "identifier":
                text = self._node_text(code, current)
                if text:
                    names.append(text)
            for child in reversed(current.children):
                stack.append(child)
        return names

    def _statement_defs_uses(self, code: bytes, statement_node: Node) -> tuple[set[str], list[str]]:
        defs: set[str] = set()
        uses: list[str] = []

        if statement_node.type == "local_variable_declaration":
            for decl in self._find_descendants(statement_node, "variable_declarator"):
                name_node = decl.child_by_field_name("name")
                value_node = decl.child_by_field_name("value")
                for var in self._identifier_names(code, name_node):
                    defs.add(var)
                uses.extend(self._identifier_names(code, value_node))

        for assign in self._find_descendants(statement_node, "assignment_expression"):
            left = assign.child_by_field_name("left")
            right = assign.child_by_field_name("right")
            defs.update(self._identifier_names(code, left))
            uses.extend(self._identifier_names(code, right))

        for update_expr in self._find_descendants(statement_node, "update_expression"):
            update_ids = self._identifier_names(code, update_expr)
            defs.update(update_ids)
            uses.extend(update_ids)

        if not defs and not uses:
            uses.extend(self._identifier_names(code, statement_node))

        # Keep stable order while deduplicating uses.
        seen: set[str] = set()
        dedup_uses: list[str] = []
        for name in uses:
            if name not in seen:
                seen.add(name)
                dedup_uses.append(name)
        return defs, dedup_uses

    def _build_method_graph(self, method_node: Node, file_bytes: bytes) -> dict[str, Any]:
        nodes: list[StatementNode] = []
        edges: list[dict[str, Any]] = []
        ts_by_id: dict[int, Node] = {}
        span_to_id: dict[tuple[int, int, str], int] = {}

        def add_edge(src: int, dst: int, edge_type: str) -> None:
            if edge_type not in self.enabled:
                return
            edges.append({"src": src, "dst": dst, "type": edge_type})
            if self.cfg.undirected_edges and src != dst:
                edges.append({"src": dst, "dst": src, "type": edge_type})

        def add_node(ts_node: Node, depth: int) -> Optional[int]:
            is_control = ts_node.type in CONTROL_NODES
            is_straight = ts_node.type in STRAIGHT_NODES
            if not (is_control or is_straight):
                return None

            if is_control:
                code = self._extract_control_header(file_bytes, ts_node)
                kind = "control"
            else:
                code = self._node_text(file_bytes, ts_node)
                kind = "straight"

            if not code:
                return None

            node_id = len(nodes)
            nodes.append(
                StatementNode(
                    id=node_id,
                    kind=kind,
                    ast_type=ts_node.type,
                    code=code,
                    start_byte=ts_node.start_byte,
                    end_byte=ts_node.end_byte,
                    depth=depth,
                )
            )
            ts_by_id[node_id] = ts_node
            span_to_id[self._span_key(ts_node)] = node_id
            return node_id

        stack: list[tuple[Node, int, Optional[int]]] = [(method_node, 0, None)]
        while stack:
            ts_node, depth, parent_selected = stack.pop()
            current_selected = add_node(ts_node, depth)

            next_parent = parent_selected
            if current_selected is not None:
                if parent_selected is not None:
                    # AST is built as child -> parent by default.
                    add_edge(current_selected, parent_selected, "AST")
                next_parent = current_selected

            for child in reversed(ts_node.children):
                if self._is_punct_leaf(child):
                    continue
                stack.append((child, depth + 1, next_parent))

        if "SEQ" in self.enabled and len(nodes) >= 2:
            if self.cfg.seq_top_level_only:
                top_depth = min(node.depth for node in nodes)
                ordered = [node for node in nodes if node.depth == top_depth]
            else:
                ordered = list(nodes)

            ordered.sort(key=lambda node: (node.start_byte, node.end_byte))
            for left, right in zip(ordered, ordered[1:]):
                if left.end_byte <= right.start_byte:
                    add_edge(left.id, right.id, "SEQ")

        if "IF_THEN" in self.enabled or "IF_ELSE" in self.enabled:
            def selected_ids_in_span(span_start: int, span_end: int) -> list[int]:
                selected_ids: list[int] = []
                for node in nodes:
                    if node.start_byte >= span_start and node.end_byte <= span_end:
                        selected_ids.append(node.id)
                return selected_ids

            stack_if = [method_node]
            while stack_if:
                current = stack_if.pop()
                if current.type == "if_statement":
                    if_id = span_to_id.get(self._span_key(current))
                    if if_id is not None:
                        then_node = current.child_by_field_name("consequence")
                        else_node = current.child_by_field_name("alternative")

                        if then_node is not None:
                            for target_id in selected_ids_in_span(then_node.start_byte, then_node.end_byte):
                                if target_id != if_id:
                                    add_edge(if_id, target_id, "IF_THEN")

                        if else_node is not None:
                            for target_id in selected_ids_in_span(else_node.start_byte, else_node.end_byte):
                                if target_id != if_id:
                                    add_edge(if_id, target_id, "IF_ELSE")

                for child in reversed(current.children):
                    if self._is_punct_leaf(child):
                        continue
                    stack_if.append(child)

        if "DATA_FLOW" in self.enabled:
            last_def: dict[str, int] = {}
            sorted_nodes = sorted(nodes, key=lambda node: (node.start_byte, node.end_byte))
            for node in sorted_nodes:
                ts_node = ts_by_id[node.id]
                defs, uses = self._statement_defs_uses(file_bytes, ts_node)
                for var in uses:
                    if var in last_def and last_def[var] != node.id:
                        add_edge(last_def[var], node.id, "DATA_FLOW")
                for var in defs:
                    last_def[var] = node.id

        return {
            "nodes": [
                {
                    "id": node.id,
                    "kind": node.kind,
                    "ast_type": node.ast_type,
                    "code": node.code,
                    "start_byte": node.start_byte,
                    "end_byte": node.end_byte,
                    "depth": node.depth,
                }
                for node in nodes
            ],
            "edges": edges,
        }

    def _extract_methods(self, tree, src: bytes, file_path: Path) -> list[dict[str, Any]]:
        methods: list[dict[str, Any]] = []

        def visit(node: Node) -> None:
            if node.type in ("method_declaration", "constructor_declaration"):
                method_name = "unknown"
                for child in node.children:
                    if child.type == "identifier":
                        method_name = self._node_text(src, child)
                        break

                methods.append(
                    {
                        "method_id": f"{file_path}:{method_name}:{node.start_point[0]}",
                        "method_name": method_name,
                        "node": node,
                    }
                )

            for child in node.children:
                visit(child)

        visit(tree.root_node)
        return methods

    @staticmethod
    def find_java_files(input_path: Path) -> list[Path]:
        """Resolve either one `.java` file or all java files under a directory."""

        input_path = input_path.expanduser().resolve()
        if input_path.is_file() and input_path.suffix.lower() == ".java":
            return [input_path]
        if input_path.is_dir():
            return sorted(p for p in input_path.rglob("*.java") if p.is_file())
        return []

    def build_records_for_file(self, java_file: Path) -> list[dict[str, Any]]:
        """Build method graph records for one Java file."""

        java_file = java_file.expanduser().resolve()
        src = java_file.read_bytes()
        tree = self.parser.parse(src)
        methods = self._extract_methods(tree, src, java_file)

        if not methods and self.cfg.include_file_fallback_method:
            fallback_graph = self._build_method_graph(tree.root_node, src)
            if fallback_graph["nodes"]:
                return [
                    {
                        "method_id": f"{java_file}:FILE:0",
                        "method_name": "FILE",
                        "file": str(java_file),
                        "nodes": fallback_graph["nodes"],
                        "edges": fallback_graph["edges"],
                    }
                ]
            return []

        rows: list[dict[str, Any]] = []
        for method in methods:
            graph = self._build_method_graph(method["node"], src)
            if not graph["nodes"]:
                continue
            rows.append(
                {
                    "method_id": method["method_id"],
                    "method_name": method["method_name"],
                    "file": str(java_file),
                    "nodes": graph["nodes"],
                    "edges": graph["edges"],
                }
            )

        return rows

    def build_records_from_path(self, input_path: Path) -> list[dict[str, Any]]:
        """Build method graph records for a file or recursively for a directory."""

        rows: list[dict[str, Any]] = []
        for java_file in self.find_java_files(input_path):
            try:
                rows.extend(self.build_records_for_file(java_file))
            except Exception as exc:  # pragma: no cover - keeps pipeline resilient
                rows.append(
                    {
                        "method_id": f"{java_file}:ERROR:0",
                        "method_name": "ERROR",
                        "file": str(java_file),
                        "nodes": [],
                        "edges": [],
                        "error": str(exc),
                    }
                )
        return rows
