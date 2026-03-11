# pylint: disable=invalid-name
"""Utilities for transforming Java code received from the frontend JSON payload."""
import re


def transform_code(file_data: dict) -> str:
    """
    Takes {name, content} and returns transformed Java code:
    - Class name gets 'Impl' suffix
    - Non-main method names get 'do' prefix (first letter capitalised)
    - All references (calls, comments, instantiations) are updated
    """
    code = file_data.get("content", "")
    code = code.replace('\r\n', '\n').replace('\r', '\n')

    # ── 1. Find class name ────────────────────────────────────────────────
    class_match = re.search(r'\bclass\s+(\w+)', code)
    if not class_match:
        return code

    class_name = class_match.group(1)
    new_class_name = class_name + 'Impl'

    # ── 2. Find non-main, non-constructor method names ────────────────────
    # Matches: [public|private|protected] [static] <returnType> <methodName>(
    method_pattern = re.compile(
        r'\b(?:public|private|protected)\s+(?:static\s+)?'
        r'(?:\w+(?:\[\])?)\s+(\w+)\s*\('
    )
    method_names = {
        m.group(1)
        for m in method_pattern.finditer(code)
        if m.group(1) not in ('main', class_name)
    }

    # ── 3. Build rename map ───────────────────────────────────────────────
    rename_map = {class_name: new_class_name}
    for method in method_names:
        rename_map[method] = 'do' + method[0].upper() + method[1:]

    # ── 4. Apply renames with word boundaries ────────────────────────────
    # Longest names first to avoid partial replacements
    for old, new in sorted(rename_map.items(), key=lambda x: -len(x[0])):
        code = re.sub(r'\b' + re.escape(old) + r'\b', new, code)

    return code
