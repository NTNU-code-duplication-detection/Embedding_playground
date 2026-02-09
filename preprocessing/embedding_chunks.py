"""
Module for creating chunks that is easy for bimodal encoders to process
"""

import javalang
from preprocessing.block_splitter import build_block_subtrees, SyntheticBlock
from preprocessing.ast_utils import clean_ast_node
from preprocessing.parser import parse_java

def get_ready_to_embed_chunks(java_code: str):
    """
    Splits java code into smaller chunks

    Returns (method name, code chunk, ast chunk)
    """
    chunks = []
    lines = java_code.splitlines()

    tree = parse_java(java_code)

    for _, method_decl in tree.filter(javalang.tree.MethodDeclaration):
        method_name = method_decl.name
        blocks = build_block_subtrees(method_decl)

        for block in blocks:
            ast_chunk = clean_ast_node(block)

            if isinstance(block, SyntheticBlock):
                code_lines = []
                for stmt in block.statements:
                    if stmt.position:
                        code_lines.append(lines[stmt.position.line - 1])
                code_chunk = "\n".join(code_lines)

            elif block.position:
                code_chunk = lines[block.position.line - 1]

            else:
                code_chunk = ""

            chunks.append((method_name, code_chunk, ast_chunk))

    return chunks
