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
