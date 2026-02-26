"""
Scan dataset module
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from program_project.config import ProgramProjectConfig


@dataclass(frozen=True)
class ProgramEntry:
    """
    Program entry dataclass

    For GoogleJam each .java file is treated as a standalone program (no explicit clones).
    """
    idx: str
    anchor: Path
    clones: List[Path]  # clones for cfg.clone_type


def _sorted_index_dirs(parent: Path) -> List[Path]:
    if not parent.exists():
        return []
    dirs = [d for d in parent.iterdir() if d.is_dir()]
    def key(d: Path):
        try:
            return (0, int(d.name))
        except Exception:
            return (1, d.name)
    return sorted(dirs, key=key)


def _sorted_problem_dirs(root: Path) -> List[Path]:
    """GoogleJam: root contains numbered problem directories (1,2,3,...)."""
    return _sorted_index_dirs(root)


def _scan_googlejam_programs(cfg: ProgramProjectConfig) -> Dict[str, ProgramEntry]:
    """Scan GoogleJam dataset: each .java file is its own ProgramEntry.

    Expected layout:
      dataset_root/
        1/
        2/
        ...

      <problem_dir>/*.java

    We treat each Java file as a standalone 'program project' so the rest of the pipeline
    (compile -> decompile -> ast_dataset -> embed_cache) can run unchanged.

    idx is a stable identifier: "<problem>/<filename_without_ext>".
    anchor is the source file path.
    clones is empty (pairing/labels are handled by pair_dataset_googlejam).
    """
    root = cfg.dataset_root
    problem_dirs = _sorted_problem_dirs(root)
    if cfg.limit_indices is not None:
        problem_dirs = problem_dirs[: cfg.limit_indices]

    out: Dict[str, ProgramEntry] = {}
    for prob_dir in problem_dirs:
        prob = prob_dir.name
        for src in sorted(prob_dir.rglob("*.java")):
            if not src.is_file():
                continue
            # Use relative-ish stable id
            idx = f"{prob}/{src.stem}"
            out[idx] = ProgramEntry(idx=idx, anchor=src, clones=[])
    return out


def scan_programs(cfg: ProgramProjectConfig) -> Dict[str, ProgramEntry]:
    """
    Scan all projects based on config
    """
    root = cfg.dataset_root

    # Special dataset mode: GoogleJam (unlabeled, many implementations per problem).
    # In this mode each .java file is treated as a standalone program.
    if cfg.clone_type == "googlejam":
        return _scan_googlejam_programs(cfg)

    base_dir = root / "base"
    clone_dir = root / cfg.clone_type

    out: Dict[str, ProgramEntry] = {}
    idx_dirs = _sorted_index_dirs(base_dir)
    if cfg.limit_indices is not None:
        idx_dirs = idx_dirs[: cfg.limit_indices]

    for d in idx_dirs:
        idx = d.name
        anchor = d / "main.java"
        if not anchor.exists():
            continue

        clones_dir = clone_dir / idx
        clones: List[Path] = []
        if clones_dir.exists():
            for p in sorted(clones_dir.glob("*.java")):
                # clones are typically 1.java,2.java,3.java
                clones.append(p)

        out[idx] = ProgramEntry(idx=idx, anchor=anchor, clones=clones)

    return out
