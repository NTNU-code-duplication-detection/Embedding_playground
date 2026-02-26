"""
Module for extracting methods from Java AST
"""
from tree_sitter import Node
from pathlib import Path
from typing import List, Dict


def _node_text(src: bytes, node: Node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def extract_methods(tree, src: bytes, file_path: Path) -> List[Dict]:
    """
    Returns list of:
    {
        "method_id",
        "method_name",
        "code",
        "node"
    }
    """
    methods = []

    cursor = tree.walk()

    def visit(node: Node):
        if node.type in ("method_declaration", "constructor_declaration"):
            name = None
            for child in node.children:
                if child.type == "identifier":
                    name = _node_text(src, child)
                    break

            if name is None:
                name = "unknown"

            methods.append({
                "method_id": f"{file_path}:{name}:{node.start_point[0]}",
                "method_name": name,
                "code": _node_text(src, node),
                "node": node,
            })

        for c in node.children:
            visit(c)

    visit(tree.root_node)
    return methods
