"""
Module for dataset gen
"""
from pathlib import Path
from typing import Generator, List, Tuple
import random

def code_clone_dataset_generator(
    dataset_root: str | Path = "data/code-clone-dataset/dataset",
    clone_type: str = "type-1",
) -> Generator[Tuple[str, List[str], List[str]], None, None]:
    """
    Yields (base_code, positive_clone_codes, negative_clone_codes) triples.
    Negative clones come from a different project in the same clone_type.

    :param dataset_root: Path to dataset/ directory
    :param clone_type: One of {"type-1", "type-2", "type-3"}
    """
    dataset_root = Path(dataset_root)

    # Resolve relative paths from project root
    if not dataset_root.is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        dataset_root = project_root / dataset_root

    base_dir = dataset_root / "base"
    clone_dir = dataset_root / clone_type

    if not base_dir.exists():
        raise FileNotFoundError(f"Missing base directory: {base_dir}")
    if not clone_dir.exists():
        raise FileNotFoundError(f"Missing clone directory: {clone_dir}")

    # List all project directories
    project_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])

    if len(project_dirs) < 2:
        raise ValueError("Need at least 2 projects to generate negatives")

    for project_dir in project_dirs:
        project_id = project_dir.name

        base_file = project_dir / "main.java"
        pos_clone_dir = clone_dir / project_id

        if not base_file.exists() or not pos_clone_dir.exists():
            continue

        # Base code
        base_code = base_file.read_text(encoding="utf-8")

        # Positive clones (same project)
        positive_clone_codes: List[str] = [
            f.read_text(encoding="utf-8") for f in sorted(pos_clone_dir.glob("*.java"))
        ]

        # Negative project (different from current)
        negative_project_dir = random.choice(
            [d for d in project_dirs if d != project_dir]
        )
        neg_clone_dir = clone_dir / negative_project_dir.name

        # Negative clones
        negative_clone_codes: List[str] = [
            f.read_text(encoding="utf-8") for f in sorted(neg_clone_dir.glob("*.java"))
        ]

        yield base_code, positive_clone_codes, negative_clone_codes
