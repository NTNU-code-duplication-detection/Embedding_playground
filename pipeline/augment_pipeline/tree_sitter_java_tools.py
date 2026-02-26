"""Module pipeline/augment_pipeline/tree_sitter_java_tools.py."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import tree_sitter_java
from tree_sitter import Language, Parser


JAVA_LANGUAGE = Language(tree_sitter_java.language())
_PARSER = Parser(JAVA_LANGUAGE)


@dataclass(frozen=True)
class IdentSpan:
    start: int
    end: int
    text: str


def parse_java(code_bytes: bytes):
    return _PARSER.parse(code_bytes)


def _node_text(code_bytes: bytes, node) -> str:
    return code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def collect_declared_local_names(code_bytes: bytes) -> Set[str]:
    """
    Collect candidate names from:
      - formal parameters
      - local variable declarators
      - catch parameters
      - enhanced for variables

    We only collect names; actual replacement happens in collect_identifier_occurrences().
    """
    tree = parse_java(code_bytes)
    root = tree.root_node

    names: Set[str] = set()
    stack = [root]

    # Nodes that “declare” a variable name often contain an identifier child.
    DECL_NODES = {
        "variable_declarator",
        "formal_parameter",
        "catch_formal_parameter",
        "enhanced_for_statement",
    }

    while stack:
        node = stack.pop()
        for ch in reversed(node.children):
            stack.append(ch)

        if node.type not in DECL_NODES:
            continue

        # Find identifier under this decl node (first identifier is usually the declared name)
        for ch in node.children:
            if ch.type == "identifier":
                nm = _node_text(code_bytes, ch)
                if nm and nm[0].isalpha():
                    names.add(nm)
                break

    return names


def _is_risky_identifier_position(node) -> bool:
    """
    Returns True if this identifier is in a position we should NOT rename.
    This is intentionally conservative.
    """
    parent = node.parent
    if parent is None:
        return True

    # These parent types strongly suggest the identifier is a definition/name of something global-ish
    RISKY_PARENTS = {
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "method_declaration",
        "constructor_declaration",
        "field_declaration",
        "annotation",
        "package_declaration",
        "import_declaration",
        "scoped_identifier",
        "type_identifier",
        "scoped_type_identifier",
        "label_statement",
    }
    if parent.type in RISKY_PARENTS:
        return True

    # Tree-sitter uses field names; many nodes have a `name:` field for defs/calls.
    # We avoid renaming identifiers that are the `name` field of common constructs.
    try:
        if parent.child_by_field_name("name") == node:
            # This catches method names, class names, method invocation name, etc.
            # Too broad, but very safe.
            return True
    except Exception:
        pass

    # Do not rename if identifier is used as member access: obj.<identifier>
    # Often represented as field_access with a field_identifier, but be safe:
    if parent.type in {"field_access", "method_invocation", "scoped_field_access"}:
        return True

    return False


def collect_identifier_occurrences(
    code_bytes: bytes,
    target_names: Set[str],
    *,
    blacklist: Optional[Set[str]] = None,
) -> List[IdentSpan]:
    """
    Collect ALL identifier spans whose text is in target_names,
    excluding risky positions.
    """
    if not target_names:
        return []

    if blacklist is None:
        blacklist = set()

    tree = parse_java(code_bytes)
    root = tree.root_node

    spans: List[IdentSpan] = []
    stack = [root]

    while stack:
        node = stack.pop()
        for ch in reversed(node.children):
            stack.append(ch)

        if node.type != "identifier":
            continue

        txt = _node_text(code_bytes, node)
        if txt not in target_names:
            continue
        if txt in blacklist:
            continue
        if _is_risky_identifier_position(node):
            continue

        spans.append(IdentSpan(node.start_byte, node.end_byte, txt))

    return spans


def apply_replacements(code_bytes: bytes, replacements: List[Tuple[int, int, bytes]]) -> bytes:
    out = bytearray(code_bytes)
    for start, end, rep in sorted(replacements, key=lambda x: x[0], reverse=True):
        out[start:end] = rep
    return bytes(out)


def make_rename_map(names: List[str], *, prefix: str = "v") -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    i = 0
    for nm in names:
        if nm in mapping:
            continue
        i += 1
        mapping[nm] = f"{prefix}{i}"
    return mapping
