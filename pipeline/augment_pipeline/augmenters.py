"""Module pipeline/augment_pipeline/augmenters.py."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Tuple

from .tree_sitter_java_tools import (
    apply_replacements,
    collect_declared_local_names,
    collect_identifier_occurrences,
    make_rename_map,
)

AugmentFn = Callable[[bytes, random.Random], bytes]


@dataclass(frozen=True)
class AugmentStats:
    applied: List[str]


def augment_identifier_renaming(code_bytes: bytes, rng: random.Random) -> bytes:
    """
    Rename local/param identifiers consistently:
      - pick names from declarations
      - replace ALL safe occurrences of those names
    """
    declared = collect_declared_local_names(code_bytes)
    if not declared:
        return code_bytes

    # Avoid renaming common types and library symbols
    blacklist = {
        "String", "System", "Math", "Integer", "Long", "Double", "Float", "Object",
        "IOException", "Exception", "BufferedReader", "FileReader", "PrintWriter",
        "File", "Arrays", "Collections", "List", "Map", "Set",
    }

    # Stable mapping for all declared locals/params
    names = sorted(declared)
    mapping = make_rename_map(names, prefix="v")

    spans = collect_identifier_occurrences(code_bytes, set(mapping.keys()), blacklist=blacklist)
    if not spans:
        return code_bytes

    repls: List[Tuple[int, int, bytes]] = []

    for sp in spans:
        new = mapping.get(sp.text)
        if new is None:
            continue

        start = sp.start

        # ---- context guards to keep code compilable ----

        # 1) Skip inside import/package lines
        line_start = code_bytes.rfind(b"\n", 0, start) + 1
        line_end = code_bytes.find(b"\n", start)
        if line_end == -1:
            line_end = len(code_bytes)
        line = code_bytes[line_start:line_end].lstrip()
        if line.startswith(b"import ") or line.startswith(b"package "):
            continue

        # 2) Skip member access (obj.field or obj.method)
        i = start - 1
        while i >= 0 and code_bytes[i] in b" \t":
            i -= 1
        if i >= 0 and code_bytes[i] == ord('.'):
            continue

        # 3) Skip class/interface/enum declarations
        prefix = code_bytes[max(0, start - 20):start]
        if b"class " in prefix or b"interface " in prefix or b"enum " in prefix:
            continue

        repls.append((sp.start, sp.end, new.encode("utf-8")))

    if not repls:
        return code_bytes

    return apply_replacements(code_bytes, repls)


def augment_whitespace_noise(code_bytes: bytes, rng: random.Random) -> bytes:
    """
    Add small whitespace noise: double some spaces, add line breaks after ';' sometimes.
    Keeps compilation stable.
    """
    text = code_bytes.decode("utf-8", errors="replace")
    out_chars: List[str] = []
    for ch in text:
        out_chars.append(ch)
        if ch == " " and rng.random() < 0.05:
            out_chars.append(" ")
        if ch == ";" and rng.random() < 0.10:
            out_chars.append("\n")
    return "".join(out_chars).encode("utf-8")


def apply_augmentations(
    code_bytes: bytes,
    *,
    rng: random.Random,
    do_rename: bool,
    rename_prob: float,
    do_ws: bool,
    ws_prob: float,
) -> Tuple[bytes, AugmentStats]:
    """
    Apply a subset of augmentations based on probabilities.
    """
    applied: List[str] = []
    out = code_bytes

    if do_rename and rng.random() < max(0.0, min(1.0, rename_prob)):
        out2 = augment_identifier_renaming(out, rng)
        if out2 != out:
            applied.append("rename_identifiers")
        out = out2

    if do_ws and rng.random() < max(0.0, min(1.0, ws_prob)):
        out2 = augment_whitespace_noise(out, rng)
        if out2 != out:
            applied.append("whitespace_noise")
        out = out2

    return out, AugmentStats(applied=applied)
