"""
Module for scanning
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from pair_dataset.config import PairDatasetConfig


@dataclass(frozen=True)
class IndexEntry:
    """
    Index entry
    """
    idx: str
    anchor: Path
    clones: List[Path]  # clone files for cfg.clone_type


def _is_index_dir(p: Path) -> bool:
    return p.is_dir()


def _sorted_index_dirs(parent: Path) -> List[Path]:
    if not parent.exists():
        return []
    dirs = [d for d in parent.iterdir() if _is_index_dir(d)]
    # Prefer numeric sort if folder names are numeric
    def key(d: Path):
        try:
            return (0, int(d.name))
        except Exception:
            return (1, d.name)
    return sorted(dirs, key=key)


def scan_dataset(cfg: PairDatasetConfig) -> Dict[str, IndexEntry]:
    """
    Returns: {idx -> IndexEntry(anchor, clones)}
    Requires:
      root/base/<idx>/<anchor_filename>
      root/<clone_type>/<idx>/(one or more .java files)
    """
    root = cfg.root
    base_dir = root / "base"
    clone_dir = root / cfg.clone_type

    out: Dict[str, IndexEntry] = {}

    base_indices = _sorted_index_dirs(base_dir)
    if cfg.limit_indices is not None:
        base_indices = base_indices[: cfg.limit_indices]

    for d in base_indices:
        idx = d.name
        anchor = d / cfg.anchor_filename
        if not anchor.exists():
            # skip if missing anchor
            continue

        clones_folder = clone_dir / idx
        clones: List[Path] = []
        if clones_folder.exists():
            # collect all .java files (excluding anchor_filename if present)
            for p in sorted(clones_folder.glob("*.java")):
                if p.name == cfg.anchor_filename:
                    continue
                clones.append(p)

        out[idx] = IndexEntry(idx=idx, anchor=anchor, clones=clones)

    return out


def summarize_index(entries: Dict[str, IndexEntry]) -> Tuple[int, int, int]:
    """
    returns: (num_indices, num_with_clones, total_clones)
    """
    num_indices = len(entries)
    num_with_clones = sum(1 for e in entries.values() if len(e.clones) > 0)
    total_clones = sum(len(e.clones) for e in entries.values())
    return num_indices, num_with_clones, total_clones
