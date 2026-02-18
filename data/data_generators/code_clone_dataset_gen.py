"""
Module for dataset gen
"""
from pathlib import Path
from typing import Generator, List, Tuple
import random

# pylint: disable=too-many-locals
# pylint: disable=broad-exception-caught
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

    if not dataset_root.is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        dataset_root = project_root / dataset_root

    base_dir = dataset_root / "base"
    clone_dir = dataset_root / clone_type

    if not base_dir.exists():
        raise FileNotFoundError(f"Missing base directory: {base_dir}")
    if not clone_dir.exists():
        raise FileNotFoundError(f"Missing clone directory: {clone_dir}")

    project_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])

    if len(project_dirs) < 2:
        raise ValueError("Need at least 2 projects to generate negatives")

    for project_dir in project_dirs:
        try:
            project_id = project_dir.name

            base_file = project_dir / "main.java"
            pos_clone_dir = clone_dir / project_id

            if not base_file.exists() or not pos_clone_dir.exists():
                continue

            # ---- Read base code ----
            base_code = base_file.read_text(encoding="utf-8")

            # ---- Read positive clones ----
            positive_clone_codes: List[str] = []
            for f in sorted(pos_clone_dir.glob("*.java")):
                positive_clone_codes.append(
                    f.read_text(encoding="utf-8")
                )

            if not positive_clone_codes:
                continue

            # ---- Pick negative project ----
            negative_project_dir = random.choice(
                [d for d in project_dirs if d != project_dir]
            )
            neg_clone_dir = clone_dir / negative_project_dir.name

            if not neg_clone_dir.exists():
                continue

            # ---- Read negative clones ----
            negative_clone_codes: List[str] = []
            for f in sorted(neg_clone_dir.glob("*.java")):
                negative_clone_codes.append(
                    f.read_text(encoding="utf-8")
                )

            if not negative_clone_codes:
                continue

            yield base_code, positive_clone_codes, negative_clone_codes

        except Exception as e:
            # Skip this sample entirely if anything goes wrong
            print(f"[SKIP] Project {project_dir.name}: {type(e).__name__} - {e}")
            continue
