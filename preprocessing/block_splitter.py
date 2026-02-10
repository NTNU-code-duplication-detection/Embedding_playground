"""
Split blocks into smaller sices
"""

import javalang

MAX_SEQ_STMTS = 5

# pylint: disable=too-few-public-methods
class SyntheticBlock:
    """
    Dataclass for statements
    """
    def __init__(self, statements):
        self.statements = statements
        self.n_statements = len(statements)

def is_control_statement(stmt):
    """
    Checks if statement defines control
    """
    return isinstance(stmt, (
        javalang.tree.IfStatement,
        javalang.tree.ForStatement,
        javalang.tree.WhileStatement,
        javalang.tree.DoStatement,
        javalang.tree.SwitchStatement,
        javalang.tree.TryStatement
    ))


def build_block_subtrees(method_decl):
    """
    Connect blocks into larger pieces
    """
    blocks = []
    current_seq = []

    for stmt in method_decl.body:
        if is_control_statement(stmt):
            if current_seq:
                blocks.append(SyntheticBlock(current_seq))
                current_seq = []
            blocks.append(stmt)
        else:
            current_seq.append(stmt)
            if len(current_seq) >= MAX_SEQ_STMTS:
                blocks.append(SyntheticBlock(current_seq))
                current_seq = []

    if current_seq:
        blocks.append(SyntheticBlock(current_seq))

    return blocks



def _indent(text, spaces=4):
    """Indent each line of text by the given number of spaces."""
    pad = " " * spaces
    return "\n".join(pad + line if line else line for line in text.splitlines())


def _get_type_name(type_node):
    """Extract type name from a type dict like {'BasicType': {'name': 'int'}}."""
    if isinstance(type_node, dict):
        for tval in type_node.values():
            if isinstance(tval, dict):
                return tval.get("name", "?")
            return str(tval)
    return str(type_node)


def _first_or_all(nodes, convert):
    """Convert first element of a list, or the node itself."""
    if isinstance(nodes, list) and nodes:
        return convert(nodes[0])
    return convert(nodes)


# ---- Individual node handlers ----
# Each takes (attrs, convert) where convert is the recursive node_to_str.

def _handle_synthetic_block(attrs, convert):
    stmts = attrs.get("statements", [])
    inner = "\n".join(convert(s) for s in stmts)
    return f"(Block\n{_indent(inner)}\n)"


def _handle_for(attrs, convert):
    ctrl = attrs.get("control", {})
    fc = ctrl.get("ForControl", ctrl)
    init = _first_or_all(fc.get("init", []), convert)
    cond = convert(fc.get("condition"))
    update = _first_or_all(fc.get("update", []), convert)
    body = convert(attrs.get("body"))
    return (
        "(For\n"
        f"  (Init {init})\n"
        f"  (Cond {cond})\n"
        f"  (Update {update})\n"
        f"  (Body\n{_indent(body)}\n  )\n"
        ")"
    )


def _handle_while(attrs, convert):
    cond = convert(attrs.get("condition"))
    body = convert(attrs.get("body"))
    return f"(While\n  (Cond {cond})\n  (Body\n{_indent(body)}\n  )\n)"


def _handle_do(attrs, convert):
    cond = convert(attrs.get("condition"))
    body = convert(attrs.get("body"))
    return f"(Do\n  (Body\n{_indent(body)}\n  )\n  (Cond {cond})\n)"


def _handle_if(attrs, convert):
    cond = convert(attrs.get("condition"))
    then = convert(attrs.get("then_statement"))
    els = attrs.get("else_statement")
    result = f"(If\n  (Cond {cond})\n  (Then\n{_indent(then)}\n  )\n"
    if els:
        result += f"  (Else\n{_indent(convert(els))}\n  )\n"
    return result + ")"


def _handle_switch(attrs, convert):
    expr = convert(attrs.get("expression"))
    cases_str = "\n".join(convert(c) for c in attrs.get("cases", []))
    return f"(Switch {expr}\n{_indent(cases_str)}\n)"


def _handle_try(attrs, convert):
    block = convert(attrs.get("block"))
    catches_str = "\n".join(convert(c) for c in attrs.get("catches", []))
    result = f"(Try\n{_indent(block)}\n"
    if catches_str:
        result += f"{_indent(catches_str)}\n"
    return result + ")"


def _handle_block_statement(attrs, convert):
    return "\n".join(convert(s) for s in attrs.get("statements", []))


def _handle_var_decl(attrs, convert):
    type_name = _get_type_name(attrs.get("type", {}))
    parts = []
    for decl in attrs.get("declarators", []):
        d = decl.get("VariableDeclarator", decl)
        name = d.get("name", "?")
        init = convert(d.get("initializer")) if d.get("initializer") else ""
        parts.append(f"(Var {type_name} {name} {init})" if init
                     else f"(Var {type_name} {name})")
    return " ".join(parts)


def _handle_assignment(attrs, convert):
    return f"(Assign {convert(attrs.get('expressionl'))} {convert(attrs.get('value'))})"


def _handle_binary_op(attrs, convert):
    return (f"({attrs.get('operator', '?')} "
            f"{convert(attrs.get('operandl'))} {convert(attrs.get('operandr'))})")


def _handle_method_invocation(attrs, convert):
    qual = attrs.get("qualifier", "")
    member = attrs.get("member", "?")
    args_str = " ".join(convert(a) for a in attrs.get("arguments", []))
    prefix = f"{qual} " if qual else ""
    return f"(Call {prefix}{member} {args_str})".rstrip()


def _handle_member_ref(attrs, convert):
    member = attrs.get("member", "?")
    selectors = attrs.get("selectors", [])
    sel_str = ""
    for sel in selectors:
        if isinstance(sel, dict) and "ArraySelector" in sel:
            sel_str += f"[{convert(sel['ArraySelector'].get('index'))}]"
    result = member + sel_str
    for op in attrs.get("postfix_operators", []):
        result = f"({op} {result})"
    for op in reversed(attrs.get("prefix_operators", [])):
        result = f"({op} {result})"
    return result


def _handle_class_creator(attrs, convert):
    type_name = _get_type_name(attrs.get("type", {}))
    args_str = " ".join(convert(a) for a in attrs.get("arguments", []))
    return f"(New {type_name} {args_str})".rstrip()


def _handle_array_creator(attrs, convert):
    type_name = _get_type_name(attrs.get("type", {}))
    dims_str = "".join(f"[{convert(d)}]" for d in attrs.get("dimensions", []))
    return f"(NewArray {type_name}{dims_str})"


def _handle_cast(attrs, convert):
    return f"(Cast {_get_type_name(attrs.get('type', {}))} {convert(attrs.get('expression'))})"


def _handle_ternary(attrs, convert):
    return (f"(Ternary {convert(attrs.get('condition'))} "
            f"{convert(attrs.get('if_true'))} {convert(attrs.get('if_false'))})")


_NODE_HANDLERS = {
    "SyntheticBlock":             _handle_synthetic_block,
    "ForStatement":               _handle_for,
    "WhileStatement":             _handle_while,
    "DoStatement":                _handle_do,
    "IfStatement":                _handle_if,
    "SwitchStatement":            _handle_switch,
    "TryStatement":               _handle_try,
    "BlockStatement":             _handle_block_statement,
    "LocalVariableDeclaration":   _handle_var_decl,
    "VariableDeclaration":        _handle_var_decl,
    "Assignment":                 _handle_assignment,
    "BinaryOperation":            _handle_binary_op,
    "MethodInvocation":           _handle_method_invocation,
    "MemberReference":            _handle_member_ref,
    "ClassCreator":               _handle_class_creator,
    "ArrayCreator":               _handle_array_creator,
    "Cast":                       _handle_cast,
    "TernaryExpression":          _handle_ternary,
}

# Simple nodes that return a single attr value
_SIMPLE_NODES = {
    "Literal":            "value",
    "This":               None,
    "StatementExpression": "expression",
}

# pylint: disable=too-many-branches
def _node_to_str(node):
    """Recursively convert an AST dict node to a compact string."""
    if node is None:
        result = ""
    elif isinstance(node, (str, int, float, bool)):
        result = str(node)
    elif isinstance(node, list):
        result = " ".join(_node_to_str(n) for n in node)
    elif isinstance(node, SyntheticBlock):
        inner = "\n".join(_node_to_str(s) for s in node.statements)
        result = f"(Block\n{_indent(inner)}\n)"
    elif not isinstance(node, dict):
        result = str(node)
    else:
        keys = list(node.keys())
        if len(keys) != 1:
            result = str(node)
        else:
            node_type = keys[0]
            attrs = node[node_type]

            if not isinstance(attrs, dict):
                result = str(attrs)
            elif handler := _NODE_HANDLERS.get(node_type):
                result = handler(attrs, _node_to_str)
            elif node_type in _SIMPLE_NODES:
                attr_key = _SIMPLE_NODES[node_type]
                result = node_type.lower() if attr_key is None \
                    else _node_to_str(attrs.get(attr_key))
            elif node_type == "ReturnStatement":
                expr = _node_to_str(attrs.get("expression"))
                result = f"(Return {expr})" if expr else "(Return)"
            else:
                children = [
                    _node_to_str(val) if isinstance(val, (dict, list)) else val
                    for val in attrs.values()
                    if isinstance(val, (dict, list, str))
                ]
                result = f"({node_type} {' '.join(children)})" if children else f"({node_type})"

    return result


def deverbose_ast(ast) -> str:
    """
    Convert an AST dict (from clean_ast_node) or SyntheticBlock into a
    compact, symbolic AST string.
    """
    return _node_to_str(ast)
