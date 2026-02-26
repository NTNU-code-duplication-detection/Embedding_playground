"""
Config module
"""
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DatasetConfig:
    """
    Config for src and out paths
    """
    src_root: Path
    out_root: Path
