"""Handler for AST conversion endpoint."""

import javalang.parser
import javalang.tokenizer
from flask import Blueprint, request, Response

from preprocessing.parser import parse_java
from preprocessing.ast_utils import clean_ast_node

ast_bp = Blueprint("ast", __name__)


@ast_bp.route("/ast", methods=["POST"])
def parse_to_ast():
    """
    POST /ast

    Request body: raw Java source code (Content-Type: text/plain)

    Response: raw output of clean_ast_node as plain text — on success
              error message as plain text              — on failure (400)
    """
    code = request.get_data(as_text=True)

    if not code.strip():
        return Response(
            "Request body must contain raw Java source code.",
            status=400,
            mimetype="text/plain",
        )

    try:
        tree = parse_java(code)
        ast_dict = clean_ast_node(tree)
        return Response(str(ast_dict), status=200, mimetype="text/plain")
    except (javalang.parser.JavaSyntaxError, javalang.tokenizer.LexerError) as exc:
        return Response(f"Failed to parse code: {str(exc)}", status=400, mimetype="text/plain")
