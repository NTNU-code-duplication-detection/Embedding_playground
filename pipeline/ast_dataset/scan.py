"""
Module for scanning
"""
from pathlib import Path
from typing import List

def find_java_files(root: Path) -> List[Path]:
    """
    Scan for all paths in folder
    """
    return sorted([p for p in root.rglob("*.java") if p.is_file()])
