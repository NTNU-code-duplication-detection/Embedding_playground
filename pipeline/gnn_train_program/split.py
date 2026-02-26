"""Module pipeline/gnn_train_program/split.py."""

from __future__ import annotations

import re
from typing import Optional, Set

# Matches .../dataset/base/09/main.java or .../dataset/type-3/09/2.java
_IDX_RE = re.compile(r"/dataset/(?:base|type-\d)/([^/]+)/", re.IGNORECASE)

def extract_idx_from_path(p: str) -> Optional[str]:
    m = _IDX_RE.search(p.replace("\\", "/"))
    if not m:
        return None
    return m.group(1)

def make_index_sets(limit_indices: int | None, val_ratio: float, seed: int = 0) -> tuple[Set[str], Set[str]]:
    """
    Returns (train_idx_set, val_idx_set) as strings like "09", "16", ...
    Uses deterministic ordering, then splits by ratio.
    """
    # Indices are assumed to be 00..99 but we only include up to limit_indices if given.
    # Keep "09" formatting (2 digits) to match your dataset.
    all_idxs = [f"{i:02d}" for i in range(100)]
    if limit_indices is not None:
        all_idxs = all_idxs[:limit_indices]

    # Deterministic split: last fraction is val
    n = len(all_idxs)
    n_val = max(1, int(round(n * val_ratio))) if n > 1 else 1
    n_val = min(n_val, n - 1) if n > 1 else 0

    train = set(all_idxs[: n - n_val])
    val = set(all_idxs[n - n_val :])
    return train, val
