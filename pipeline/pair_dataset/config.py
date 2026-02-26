"""
Module for configs
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

CloneType = Literal["type-1", "type-2", "type-3"]

@dataclass(frozen=True)
class PairDatasetConfig:
    """
    Config class
    """
    root: Path  # project_root/data/code-clone-dataset/dataset
    clone_type: CloneType = "type-3"

    # Filenames (change if needed)
    anchor_filename: str = "main.java"

    # Neg sampling
    negative_pool: Literal["base", "same_clone_type"] = "same_clone_type"
    seed: int = 0

    # Optional: restrict to a subset of indices
    limit_indices: Optional[int] = None
