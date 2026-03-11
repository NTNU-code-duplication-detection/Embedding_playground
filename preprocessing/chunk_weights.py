"""
Chunk importance weights based on control-flow complexity and nesting depth.

Two public functions:
  - chunk_weight(ast_dict)          → float  (single chunk)
  - normalize_chunk_weights(chunks) → list[float]  (file-level normalisation)
"""

import math

# All control-flow node types produced by clean_ast_node / javalang
CONTROL_NODE_TYPES = frozenset({
    "IfStatement",
    "ForStatement",
    "WhileStatement",
    "DoStatement",
    "SwitchStatement",
    "TryStatement",
})


# ---------------------------------------------------------------------------
# Internal AST walker
# ---------------------------------------------------------------------------

def _walk(node, ctrl_depth, results):
    """
    Recursively walk *node* (an ast_dict, list, or scalar).

    Whenever a control-flow node is encountered its current *ctrl_depth* is
    compared against the running maximum, then children are walked at
    ctrl_depth + 1 so nested structures accumulate depth correctly.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if key in CONTROL_NODE_TYPES:
                results["found"] = True
                if ctrl_depth > results["max_depth"]:
                    results["max_depth"] = ctrl_depth
                _walk(value, ctrl_depth + 1, results)
            else:
                _walk(value, ctrl_depth, results)
    elif isinstance(node, list):
        for item in node:
            _walk(item, ctrl_depth, results)
    # scalars (str, int, bool, None) → nothing to walk


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def has_control_flow(ast_dict) -> int:
    """
    Return ``1`` if *ast_dict* contains any control-flow statement
    (if, for, while, do, switch, try), else ``0``.

    Parameters
    ----------
    ast_dict : dict
        Cleaned AST dict as produced by ``clean_ast_node``.
    """
    results = {"found": False, "max_depth": 0}
    _walk(ast_dict, 0, results)
    return 1 if results["found"] else 0


def max_control_nesting_depth(ast_dict) -> int:
    """
    Return the maximum control-flow nesting depth inside *ast_dict*.

    Depth is **0-based at the outermost control statement**:

    * ``if { }``              → 0
    * ``for { if { } }``     → 1
    * ``for { if { if { } } }`` → 2

    Returns 0 for chunks with no control flow.
    """
    results = {"found": False, "max_depth": 0}
    _walk(ast_dict, 0, results)
    return results["max_depth"] if results["found"] else 0


# ---------------------------------------------------------------------------
# Weight calculation
# ---------------------------------------------------------------------------

def chunk_weight(ast_dict) -> float:
    """
    Compute the importance weight for a single chunk.

    Formula::

        depth_bonus = 1 + log(1 + max_nesting_depth)

    Worked examples
    ---------------
    * No control flow or a single flat ``if``  → 1.0   (depth 0, log(1) = 0)
    * ``for → if``                             → ≈1.69  (depth 1, log(2) ≈ 0.69)
    * ``for → if → if``                        → ≈2.10  (depth 2, log(3) ≈ 1.10)

    Parameters
    ----------
    ast_dict : dict
        Cleaned AST dict as produced by ``clean_ast_node``.
    """
    depth = max_control_nesting_depth(ast_dict)
    return 1.0 + math.log(1 + depth)


def normalize_chunk_weights(chunks) -> list:
    """
    Compute and normalize importance weights for a list of ``ContextChunk``\\s.

    Weights are scaled so they **sum to 1.0**, making them directly usable
    as mixing coefficients when combining per-chunk embeddings into a single
    method-level embedding.

    Parameters
    ----------
    chunks : list[tuple[str, str, dict]]
        All chunks belonging to one file, each a ``(method_name, code, ast_dict)``
        tuple as returned by ``get_ready_to_embed_chunks``.

    Returns
    -------
    list[float]
        Normalized weights, parallel to *chunks*, summing to 1.0.
        Falls back to uniform weights if all raw weights are zero.
    """
    if not chunks:
        return []

    weights = [chunk_weight(c[2]) for c in chunks]
    total = sum(weights)

    if total == 0.0:
        uniform = 1.0 / len(weights)
        return [uniform] * len(weights)

    return [w / total for w in weights]
