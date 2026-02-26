"""
Module for creating parser
"""
from tree_sitter import Language, Parser
import tree_sitter_java

JAVA_LANGUAGE = Language(tree_sitter_java.language())


def make_parser() -> Parser:
    """
    Makes JAVA parser
    """
    parser = Parser(JAVA_LANGUAGE)
    return parser
