from pathlib import Path

def samples_generator(dataset_root: str):
    """
    Generator for retrieving file paths from BigCloneBench samples folder.

    @param dataset_root: str of path to BigCloneBench dataset root
    @return Generator of Path objects for each .java file in samples/
    """
    samples_path = Path(dataset_root) / "samples"
    if not samples_path.exists():
        raise ValueError(f"samples folder not found at {samples_path}")

    for java_file in samples_path.glob("*.java"):
        yield java_file


def default_generator(dataset_root: str):
    """
    Generator for retrieving file paths from BigCloneBench default folder.

    @param dataset_root: str of path to BigCloneBench dataset root
    @return Generator of Path objects for each .java file in default/
    """
    default_path = Path(dataset_root) / "default"
    if not default_path.exists():
        raise ValueError(f"default folder not found at {default_path}")

    for java_file in default_path.glob("*.java"):
        yield java_file
