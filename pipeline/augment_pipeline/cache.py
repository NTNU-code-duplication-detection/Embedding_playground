"""Module pipeline/augment_pipeline/cache.py."""

from __future__ import annotations

import hashlib
from pathlib import Path


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def augmented_path(out_dir: Path, source_path: Path, aug_key: str) -> Path:
    """
    Create a stable path for an augmented copy, derived from:
      - absolute source path
      - augmentation key
    """
    h = hashlib.sha1((str(source_path.resolve()) + "::" + aug_key).encode("utf-8")).hexdigest()
    return out_dir / f"{h}.java"


def write_augmented(
    *,
    out_dir: Path,
    source_path: Path,
    aug_key: str,
    content: bytes,
    cache: bool,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = augmented_path(out_dir, source_path, aug_key)
    if cache and dst.exists():
        return dst
    dst.write_bytes(content)
    return dst
