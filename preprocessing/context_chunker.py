"""
Context-aware code chunker for Java methods.

Produces chunks that preserve full control-structure bodies.
Drop-in replacement for the get_ready_to_embed_chunks pipeline.
"""

from dataclasses import dataclass, field
import javalang

from preprocessing.block_splitter import SyntheticBlock, is_control_statement
from preprocessing.ast_utils import clean_ast_node
from preprocessing.parser import parse_java


@dataclass
class ContextChunk:
    """One embeddable chunk of a method."""
    method_name: str
    code: str
    ast_dict: dict
    statements: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _update_best(best, candidate):
    """Return the larger of *best* and *candidate*, treating None as -inf."""
    if candidate is None:
        return best
    if best is None:
        return candidate
    return max(best, candidate)


def _deepest_position_line(node):
    """
    Walk a javalang AST subtree and return the largest .position.line found.
    Returns None when no position exists anywhere in the subtree.
    """
    best = None
    if hasattr(node, 'position') and node.position is not None:
        best = node.position.line

    if not hasattr(node, 'children'):
        return best

    for child in node.children:
        if child is None:
            continue
        items = child if isinstance(child, list) else [child]
        for item in items:
            if hasattr(item, 'position') or hasattr(item, 'children'):
                best = _update_best(best, _deepest_position_line(item))

    return best


def _find_closing_brace(lines, start_line_idx):
    """
    Starting from *start_line_idx* (0-based), scan forward using
    brace-balancing to find the line containing the matching '}'.
    Returns the 0-based line index of the closing brace.
    If no opening brace is found or balancing fails, returns start_line_idx.
    """
    depth = 0
    found_open = False

    for idx in range(start_line_idx, len(lines)):
        for ch in lines[idx]:
            if ch == '{':
                depth += 1
                found_open = True
            elif ch == '}':
                depth -= 1
                if found_open and depth == 0:
                    return idx

    return start_line_idx


def _extract_atom_lines(stmt, lines):
    """
    Return the source lines (0-based indices) that belong to *stmt*.

    For control statements the range extends from the statement's start line
    to its closing brace (found via brace-balancing).
    For plain statements only the deepest-position line is used.
    """
    if stmt.position is None:
        return []

    start = stmt.position.line - 1  # 0-based

    if is_control_statement(stmt):
        deepest = _deepest_position_line(stmt)
        deepest_idx = (deepest - 1) if deepest else start
        # Scan from the deepest known line to find the final '}'
        end = _find_closing_brace(lines, start)
        # Make sure we at least cover the deepest AST position
        end = max(end, deepest_idx)
        return list(range(start, end + 1))

    # Non-control: may span multiple lines (e.g. chained method calls)
    deepest = _deepest_position_line(stmt)
    end = (deepest - 1) if deepest else start
    return list(range(start, end + 1))


def _estimate_tokens(text, tokenizer=None):
    """Estimate token count. Uses tokenizer if provided, else heuristic."""
    if tokenizer is not None:
        return tokenizer(text)
    return len(text) // 4 + 1


# ---------------------------------------------------------------------------
# Core chunking logic
# ---------------------------------------------------------------------------

def _flush_chunk(line_idxs, stmts, method_name, lines):
    """Build a ContextChunk from accumulated line indices and statements."""
    if not line_idxs:
        return None
    ordered = sorted(set(line_idxs))
    code = "\n".join(lines[i] for i in ordered)

    if len(stmts) == 1:
        ast_dict = clean_ast_node(stmts[0])
    else:
        ast_dict = clean_ast_node(SyntheticBlock(stmts))

    return ContextChunk(
        method_name=method_name,
        code=code,
        ast_dict=ast_dict,
        statements=list(stmts),
    )


def _build_atoms(method_decl, lines):
    """Return list of (line_indices, stmt, is_control) for each body stmt."""
    return [
        (_extract_atom_lines(stmt, lines), stmt, is_control_statement(stmt))
        for stmt in method_decl.body
    ]


def _lines_to_text(line_idxs, lines):
    """Join source lines referenced by 0-based indices."""
    return "\n".join(lines[i] for i in line_idxs) if line_idxs else ""


def _chunk_method(method_decl, lines, max_tokens, tokenizer):
    """Greedy-pack a single method's statements into chunks."""
    atoms = _build_atoms(method_decl, lines)
    if not atoms:
        return []

    chunks = []
    cur_lines = []
    cur_stmts = []

    for atom_lines, stmt, _ in atoms:
        if cur_lines:
            combined_tokens = _estimate_tokens(
                _lines_to_text(sorted(set(cur_lines + atom_lines)), lines),
                tokenizer)
        else:
            combined_tokens = _estimate_tokens(
                _lines_to_text(atom_lines, lines), tokenizer)

        if cur_lines and combined_tokens > max_tokens:
            chunk = _flush_chunk(cur_lines, cur_stmts,
                                 method_decl.name, lines)
            if chunk:
                chunks.append(chunk)
            cur_lines = []
            cur_stmts = []

        cur_lines.extend(atom_lines)
        cur_stmts.append(stmt)

    chunk = _flush_chunk(cur_lines, cur_stmts, method_decl.name, lines)
    if chunk:
        chunks.append(chunk)

    return chunks


def get_context_chunks(java_code, max_tokens=480, tokenizer=None):
    """
    Split every method in *java_code* into context-aware chunks.

    Parameters
    ----------
    java_code : str
        Complete Java source (class with methods).
    max_tokens : int
        Soft token budget per chunk (~480 for RoBERTa-512 with room for
        special tokens).
    tokenizer : callable or None
        ``tokenizer(text) -> int`` returning token count.  Falls back to a
        conservative ``len(text) // 4 + 1`` heuristic.

    Returns
    -------
    list[ContextChunk]
    """
    lines = java_code.splitlines()
    tree = parse_java(java_code)
    all_chunks = []

    for _, method_decl in tree.filter(javalang.tree.MethodDeclaration):
        if method_decl.body is None:
            continue
        all_chunks.extend(
            _chunk_method(method_decl, lines, max_tokens, tokenizer)
        )

    return all_chunks


def get_ready_to_embed_context_chunks(java_code, max_tokens=480,
                                      tokenizer=None):
    """
    Drop-in replacement for ``get_ready_to_embed_chunks``.

    Returns
    -------
    list[tuple[str, str, dict]]
        Each tuple is ``(method_name, code_string, ast_dict)``.
    """
    chunks = get_context_chunks(
        java_code,
        max_tokens=max_tokens,
        tokenizer=tokenizer,
    )
    return [(c.method_name, c.code, c.ast_dict) for c in chunks]


def safe_get_ready_to_embed_context_chunks(java_code, max_tokens=480,
                                           context_lines=2, tokenizer=None):
    """
    Drop-in replacement for ``get_ready_to_embed_chunks``.

    Returns
    -------
    list[tuple[str, str, dict]]
        Each tuple is ``(method_name, code_string, ast_dict)``.
    """
    try:
        return get_ready_to_embed_context_chunks(java_code, max_tokens,# pylint: disable=(too-many-function-args)
                                                 context_lines, tokenizer)
    except javalang.tokenizer.LexerError as e:
        print(f"[SKIP] LexerError in context chunking: {e}")
        return []
    except javalang.parser.JavaSyntaxError as e:
        print(f"[SKIP] JavaSyntaxError in context chunking: {e}")
        return []

    except Exception as e:# pylint: disable=(broad-exception-caught)
        print(f"[SKIP] Unexpected error in context chunking: {type(e).__name__} - {e}")
        return []
