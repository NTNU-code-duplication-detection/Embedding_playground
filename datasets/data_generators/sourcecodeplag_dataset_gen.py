from pathlib import Path


def find_single_java(folder: Path) -> Path:
    """
    Finds the single file in a folder given as Path

    @param folder: Path: Folder to flatten
    @return Path: File retirved
    """
    java_files = list(folder.glob("*.java"))
    if len(java_files) != 1:
        raise ValueError(f"Expected 1 java file in {folder}, found {len(java_files)}")
    return java_files[0]


def original_non_plagiarized_generator(dataset_root: str):
    """
    Generator for retrieving pair of original and non-plagiarized files

    @param dataset_root: str of path
    @return Generator
    """
    dataset_root = Path(dataset_root)

    for outer in dataset_root.iterdir():
        if not outer.is_dir():
            continue

        original_dir = outer / "original"
        non_plag_dir = outer / "non-plagiarized"

        if not original_dir.exists() or not non_plag_dir.exists():
            continue

        original_java = find_single_java(original_dir)

        for np_folder in non_plag_dir.iterdir():
            if not np_folder.is_dir():
                continue

            np_java = find_single_java(np_folder)
            yield original_java, np_java


def original_plagiarized_generator(dataset_root: str):
    """
    Generator for retrieving pair of original and plagiarized files

    @param dataset_root: str of path
    @return Generator
    """
    dataset_root = Path(dataset_root)

    for outer in dataset_root.iterdir():
        if not outer.is_dir():
            continue

        original_dir = outer / "original"
        plag_dir = outer / "plagiarized"

        if not original_dir.exists() or not plag_dir.exists():
            continue

        original_java = find_single_java(original_dir)

        for plag_outer in plag_dir.iterdir():
            if not plag_outer.is_dir():
                continue

            for plag_inner in plag_outer.iterdir():
                if not plag_inner.is_dir():
                    continue

                plag_java = find_single_java(plag_inner)
                yield original_java, plag_java
