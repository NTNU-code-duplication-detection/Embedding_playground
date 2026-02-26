"""Module pipeline/augment_pipeline/config.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class AugmentConfig:
    """
    Configuration for Java augmentation.

    root:
      Directory containing Java programs (recursively).
      For GCJ compiled dataset, you can point to ../data/gcj_compiled
      and use glob="*.java".

    out_dir:
      Where augmented Java files are written.
      Augmented files are created on demand and cached.
    """

    root: Path
    out_dir: Path

    seed: int = 0
    glob: str = "*.java"

    # Controls how many buckets to include for GCJ-style directory layouts.
    # If None: include all buckets under root.
    limit_buckets: Optional[int] = None
    max_files_per_bucket: Optional[int] = None

    # Augmentation knobs
    rename_identifiers: bool = True
    rename_prob: float = 1.0  # probability of applying renaming when chosen
    whitespace_noise: bool = True
    whitespace_prob: float = 0.5

    # If True, do not overwrite existing augmented files (cache on disk).
    cache_augmented: bool = True
