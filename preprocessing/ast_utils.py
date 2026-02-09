"""
Clean ast module
"""


def clean_ast_node(node):
    """
    Remove empty lists and decorators in AST tree
    """
    if isinstance(node, list):
        return [clean_ast_node(n) for n in node if n not in (None, [], {}, set())]

    if hasattr(node, "__dict__"):
        cleaned = {}
        for k, v in node.__dict__.items():
            if k.startswith("_") or v in (None, [], {}, set()):
                continue
            cleaned[k] = clean_ast_node(v)
        return {node.__class__.__name__: cleaned}

    return node
