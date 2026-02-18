"""
Generator for Source Code Plagiarism Dataset

Dataset:
    Name: IR-Plag-Dataset
    Source: https://github.com/oscarkarnalim/sourcecodeplagiarismdataset
    Version/Commit: f07e951 (submodule commit)
    License: Apache License 2.0
"""
from pathlib import Path
import random
from data.data_generators.schema import CodeSample

ORIGINAL    = 'original'
NON_PLAG    = 'non-plagiarized'
PLAG        = 'plagiarized'

_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = (
    _THIS_DIR
    / ".."
    / "sourcecodeplagiarismdataset"
    / "IR-Plag-Dataset"
).resolve()

def _read_java(java_path: Path) -> str:
    """
    Reads a single java file from path
    """
    return java_path.read_text(encoding="utf-8", errors="ignore")


def _find_single_java(folder: Path) -> Path:
    """
    Finds the single file in a folder given as Path

    @param folder: Path: Folder to flatten
    @return Path: File retirved
    """
    java_files = list(folder.glob("*.java"))
    if len(java_files) != 1:
        raise ValueError(f"Expected 1 java file in {folder}, found {len(java_files)}")
    return java_files[0]


def original_non_plagiarized_generator(dataset_root: str | Path = DEFAULT_DATASET_ROOT):
    """
    Generator for retrieving pair of original and non-plagiarized files

    @param dataset_root: str of path
    @return Generator
    """
    dataset_root = Path(dataset_root)

    for outer in dataset_root.iterdir():
        if not outer.is_dir():
            continue

        original_dir = outer / ORIGINAL
        non_plag_dir = outer / NON_PLAG

        if not original_dir.exists() or not non_plag_dir.exists():
            continue

        original_java = _find_single_java(original_dir)
        original_code = _read_java(original_java)

        for np_folder in non_plag_dir.iterdir():
            if not np_folder.is_dir():
                continue

            np_java = _find_single_java(np_folder)
            np_code = _read_java(np_java)

            yield CodeSample(
                code_a=original_code,
                code_b=np_code,
                label=0,
                dataset='SourceCodePlag'
            )


def original_plagiarized_generator(dataset_root: str | Path = DEFAULT_DATASET_ROOT):
    """
    Generator for retrieving pair of original and plagiarized files

    @param dataset_root: str of path
    @return Generator
    """
    dataset_root = Path(dataset_root)

    for outer in dataset_root.iterdir():
        if not outer.is_dir():
            continue

        original_dir = outer / ORIGINAL
        plag_dir = outer / PLAG

        if not original_dir.exists() or not plag_dir.exists():
            continue

        original_java = _find_single_java(original_dir)
        original_code = _read_java(original_java)

        for plag_outer in plag_dir.iterdir():
            if not plag_outer.is_dir():
                continue

            for plag_inner in plag_outer.iterdir():
                if not plag_inner.is_dir():
                    continue

                plag_java = _find_single_java(plag_inner)
                plag_code = _read_java(plag_java)

                yield CodeSample(
                    code_a=original_code,
                    code_b=plag_code,
                    label=1,
                    dataset='SourceCodePlag'
                )


# pylint: disable=too-many-locals
# pylint: disable=broad-exception-caught
def original_plag_triplet_generator(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    seed: int | None = None
):
    """
    Yields (anchor, clone, nonclone) triplets from IR-Plag-Dataset.

    Anchor     = original code
    Clone      = plagiarized version of anchor
    Non-clone  = original code from a different problem
    """
    if seed is not None:
        random.seed(seed)

    dataset_root = Path(dataset_root)

    problem_dirs = [
        d for d in dataset_root.iterdir()
        if d.is_dir() and (d / ORIGINAL).exists()
    ]

    if len(problem_dirs) < 2:
        raise ValueError("Need at least 2 problems to sample non-clones")

    for problem_dir in problem_dirs:
        original_dir = problem_dir / ORIGINAL
        plag_dir = problem_dir / PLAG

        if not plag_dir.exists():
            continue

        try:
            anchor_java = _find_single_java(original_dir)
            anchor_code = _read_java(anchor_java)
        except Exception:
            continue

        # Pick a plagiarized version
        plag_folders = [
            d for d in plag_dir.iterdir()
            if d.is_dir()
        ]

        for plag_outer in plag_folders:
            for plag_inner in plag_outer.iterdir():
                if not plag_inner.is_dir():
                    continue

                try:
                    clone_java = _find_single_java(plag_inner)
                    clone_code = _read_java(clone_java)
                except Exception:
                    continue

                # Sample non-clone from a DIFFERENT problem
                neg_problem = random.choice(
                    [d for d in problem_dirs if d != problem_dir]
                )

                try:
                    neg_original_java = _find_single_java(
                        neg_problem / ORIGINAL
                    )
                    nonclone_code = _read_java(neg_original_java)
                except Exception:
                    continue

                yield {
                    "anchor": anchor_code,
                    "clone": clone_code,
                    "nonclone": nonclone_code,
                    "dataset": "SourceCodePlag",
                    "problem_anchor": problem_dir.name,
                    "problem_nonclone": neg_problem.name,
                }
