"""Module preprocessing/augmentation.py."""

import random
from typing import Dict, List, Tuple, Optional

import tree_sitter_java
from tree_sitter import Language, Parser

JAVA_LANGUAGE = Language(tree_sitter_java.language())
parser = Parser(JAVA_LANGUAGE)


PUNCT_LEAVES = (";", "{", "}", "(", ")", ",")
KEYWORDS = {
    "abstract","assert","boolean","break","byte","case","catch","char","class","const","continue",
    "default","do","double","else","enum","extends","final","finally","float","for","goto","if",
    "implements","import","instanceof","int","interface","long","native","new","package","private",
    "protected","public","return","short","static","strictfp","super","switch","synchronized","this",
    "throw","throws","transient","try","void","volatile","while","var","record","sealed","permits",
    "non-sealed","yield"
}

def augment_java(code: str, *, seed: Optional[int] = None) -> str:
    """
    Semantics-preserving augmentation for contrastive learning.
    Implements 3 "starter" transforms (conservative):
      1) Scope-limited local variable renaming (inside each method body block)
      2) i++  <->  i += 1   (only when it's a standalone statement)
      3) Brace normalization: wrap single-statement bodies of if/else/for/while/do in { ... }

    Returns augmented code (or original if something goes wrong).
    """
    rng = random.Random(seed)

    src_bytes = code.encode("utf8")
    tree = parser.parse(src_bytes)
    root = tree.root_node

    edits: List[Tuple[int, int, bytes]] = []  # (start_byte, end_byte, replacement_bytes)

    # -------------------------
    # Helpers
    # -------------------------
    def node_text(n) -> str:
        return src_bytes[n.start_byte:n.end_byte].decode("utf8", errors="replace")

    def add_insert(pos: int, s: str):
        edits.append((pos, pos, s.encode("utf8")))

    def add_replace(start: int, end: int, s: str):
        edits.append((start, end, s.encode("utf8")))

    def is_leaf_punct(n) -> bool:
        return len(n.children) == 0 and n.type in PUNCT_LEAVES

    def walk(n, fn):
        """
        Iterative DFS to avoid RecursionError on deep ASTs.
        """
        stack = [n]
        while stack:
            cur = stack.pop()
            fn(cur)

            # push children in reverse so traversal order is roughly preserved
            chs = cur.children
            for ch in reversed(chs):
                if is_leaf_punct(ch):
                    continue
                stack.append(ch)

    def apply_edits(original: bytes, edits_list: List[Tuple[int, int, bytes]]) -> bytes:
        # Apply from back to front so offsets don't shift
        out = original
        for s, e, rep in sorted(edits_list, key=lambda t: (t[0], t[1]), reverse=True):
            out = out[:s] + rep + out[e:]
        return out

    def parse_ok(b: bytes) -> bool:
        try:
            t = parser.parse(b)
            return t is not None and t.root_node is not None and len(t.root_node.children) > 0
        except Exception:
            return False

    # -------------------------
    # 3) Brace normalization
    # -------------------------
    # We wrap non-block bodies for: if/else/for/while/do (conservative rules).
    def add_brace_wrap(stmt_node):
        # Insert { before stmt, } after stmt
        add_insert(stmt_node.start_byte, "{\n")
        add_insert(stmt_node.end_byte, "\n}")

    def braces_transform():
        local_edits: List[Tuple[int, int, bytes]] = []

        def add_insert_local(pos: int, s: str):
            local_edits.append((pos, pos, s.encode("utf8")))

        def wrap_local(stmt_node):
            add_insert_local(stmt_node.start_byte, "{\n")
            add_insert_local(stmt_node.end_byte, "\n}")

        def visit(n):
            # if_statement: consequence + alternative
            if n.type == "if_statement":
                cons = n.child_by_field_name("consequence")
                alt = n.child_by_field_name("alternative")

                if cons is not None and cons.type != "block":
                    # Wrap only if it's truly a statement node
                    wrap_local(cons)

                if alt is not None:
                    # Avoid wrapping "else if" (alternative is another if_statement)
                    if alt.type != "block" and alt.type != "if_statement":
                        wrap_local(alt)

            # loops
            elif n.type in ("for_statement", "while_statement", "do_statement", "enhanced_for_statement"):
                body = n.child_by_field_name("body")
                if body is not None and body.type != "block":
                    wrap_local(body)

        walk(root, visit)

        # Randomly decide to apply braces transform (often helps)
        # To ensure diversity, we apply it with ~50% probability.
        if not local_edits or rng.random() > 0.5:
            return

        # Apply to a temp buffer and re-parse for safety
        tmp = apply_edits(src_bytes, local_edits)
        if parse_ok(tmp):
            edits.extend(local_edits)

    # -------------------------
    # 2) i++ <-> i += 1  (standalone statement only)
    # -------------------------
    def inc_transform():
        candidates: List[Tuple[int, int, str]] = []  # (start, end, replacement)

        def norm_no_space(s: str) -> str:
            return "".join(s.split())

        def is_simple_ident(s: str) -> bool:
            return s.isidentifier() and s not in KEYWORDS

        def consider_update_expr(expr_node):
            """
            Add candidates for:
            - update_expression: i++ / i--
            - assignment_expression: i += 1 / i -= 1 (reverse)
            """
            if expr_node is None:
                return

            if expr_node.type == "update_expression":
                txt = norm_no_space(node_text(expr_node))
                if txt.endswith("++"):
                    var = txt[:-2]
                    if is_simple_ident(var):
                        candidates.append((expr_node.start_byte, expr_node.end_byte, f"{var} += 1"))
                elif txt.endswith("--"):
                    var = txt[:-2]
                    if is_simple_ident(var):
                        candidates.append((expr_node.start_byte, expr_node.end_byte, f"{var} -= 1"))

            elif expr_node.type == "assignment_expression":
                op = expr_node.child_by_field_name("operator")
                left = expr_node.child_by_field_name("left")
                right = expr_node.child_by_field_name("right")
                if op is None or left is None or right is None:
                    return

                op_txt = node_text(op).strip()
                l_txt = norm_no_space(node_text(left))
                r_txt = norm_no_space(node_text(right))

                if op_txt == "+=" and r_txt == "1" and is_simple_ident(l_txt):
                    candidates.append((expr_node.start_byte, expr_node.end_byte, f"{l_txt}++"))
                elif op_txt == "-=" and r_txt == "1" and is_simple_ident(l_txt):
                    candidates.append((expr_node.start_byte, expr_node.end_byte, f"{l_txt}--"))

        def visit(n):
            # A) standalone statement: i++;  (expression_statement -> update_expression)
            if n.type == "expression_statement":
                expr = n.child_by_field_name("expression")
                consider_update_expr(expr)

            # B) for-loop update slot: for (...; ...; update)
            if n.type == "for_statement":
                upd = n.child_by_field_name("update")
                # update can be: update_expression, assignment_expression, or comma_expression
                if upd is None:
                    return

                if upd.type in ("update_expression", "assignment_expression"):
                    consider_update_expr(upd)

                elif upd.type == "comma_expression":
                    # handle: i++, j++  (apply to each child expression)
                    for ch in upd.children:
                        if len(ch.children) == 0 and ch.type in PUNCT_LEAVES:
                            continue
                        if ch.type in ("update_expression", "assignment_expression"):
                            consider_update_expr(ch)

        walk(root, visit)

        if not candidates:
            return

        rng.shuffle(candidates)
        k = 1 if len(candidates) == 1 else rng.choice([1, 2])
        chosen = candidates[:k]

        local_edits = [(s, e, rep.encode("utf8")) for (s, e, rep) in chosen]
        tmp = apply_edits(src_bytes, local_edits)
        if parse_ok(tmp):
            edits.extend(local_edits)

    # -------------------------
    # 1) Local variable renaming (scope-limited to method body blocks)
    # -------------------------
    def rename_transform():
        local_edits: List[Tuple[int, int, bytes]] = []

        # Collect all method bodies
        method_bodies = []

        def collect_methods(n):
            if n.type == "method_declaration":
                body = n.child_by_field_name("body")
                if body is not None and body.type == "block":
                    method_bodies.append(body)

        walk(root, collect_methods)

        if not method_bodies:
            return

        # For each method body, rename some locals declared within
        for body in method_bodies:
            body_start, body_end = body.start_byte, body.end_byte

            # Gather declared local variable names within this body (conservative)
            declared: List[str] = []
            decl_counts: Dict[str, int] = {}

            def collect_decl(n):
                if n.type == "formal_parameter":
                    name_node = n.child_by_field_name("name")
                    if name_node is not None and name_node.type == "identifier":
                        name = node_text(name_node)
                        if name and name.isidentifier() and name not in KEYWORDS:
                            declared.append(name)
                            decl_counts[name] = decl_counts.get(name, 0) + 1
                if n.type == "local_variable_declaration":
                    # variable_declarator nodes contain name field
                    for ch in n.children:
                        if ch.type == "variable_declarator":
                            name_node = ch.child_by_field_name("name")
                            if name_node is not None and name_node.type == "identifier":
                                name = node_text(name_node)
                                if name and name.isidentifier() and name not in KEYWORDS:
                                    declared.append(name)
                                    decl_counts[name] = decl_counts.get(name, 0) + 1

            walk(body, collect_decl)

            # Skip shadowed/redeclared names to avoid scope bugs
            declared_unique = [x for x in declared if decl_counts.get(x, 0) == 1]
            if not declared_unique:
                continue

            # Build a set of all identifiers in the body to avoid collisions
            all_idents = set()

            def collect_idents(n):
                if n.type == "identifier":
                    all_idents.add(node_text(n))

            walk(body, collect_idents)

            # Pick 1-3 vars to rename
            rng.shuffle(declared_unique)
            num_to_rename = min(len(declared_unique), rng.choice([1, 2, 3]))
            to_rename = declared_unique[:num_to_rename]

            # Create new names
            rename_map: Dict[str, str] = {}
            ctr = 0
            for old in to_rename:
                while True:
                    new = f"v{rng.randrange(1000, 9999)}"
                    if new not in all_idents and new not in KEYWORDS:
                        break
                rename_map[old] = new
                all_idents.add(new)

            if not rename_map:
                continue

            # Apply rename edits on identifier tokens inside the body span.
            # Conservative skip: do not rename if immediately preceded by '.' (field access)
            # and do not rename if it looks like a type (UpperCamel) — optional extra safety.
            def add_rename_edits(n):
                if n.type != "identifier":
                    return
                if not (body_start <= n.start_byte and n.end_byte <= body_end):
                    return
                txt = node_text(n)
                if txt not in rename_map:
                    return

                # Skip if preceded by '.' => likely field/method selector
                if n.start_byte > 0 and src_bytes[n.start_byte - 1] == ord('.'):
                    return

                # Optional: skip UpperCamelCase to avoid renaming type identifiers
                if txt[:1].isupper():
                    return

                new_txt = rename_map[txt]
                local_edits.append((n.start_byte, n.end_byte, new_txt.encode("utf8")))

            walk(body, add_rename_edits)

        if not local_edits:
            return

        # Apply and validate
        tmp = apply_edits(src_bytes, local_edits)
        if parse_ok(tmp):
            edits.extend(local_edits)

    # -------------------------
    # Run transforms (in random order)
    # -------------------------
    transforms = [rename_transform, inc_transform, braces_transform]
    rng.shuffle(transforms)
    for t in transforms:
        t()

    if not edits:
        return code

    out_bytes = apply_edits(src_bytes, edits)

    # Final safety: ensure valid parse
    if not parse_ok(out_bytes):
        return code

    return out_bytes.decode("utf8", errors="replace")


if __name__ == '__main__':
    # aug1, aug2 are your positive pair views

    code="""
class DemoInc {
    int sum(int n) {
        int s = 0;
        for (int i = 0; i < n; i++)
            s = s + i;
        return s;
    }
}
    """
    aug1 = augment_java(code, seed=1)
    aug2 = augment_java(code, seed=2)
    print("Original:")
    print(code)
    print("-"*20)
    print("Augmentation 1:")
    print(aug1)
    print("-"*20)
    print("Augmentation 2:")
    print(aug2)
