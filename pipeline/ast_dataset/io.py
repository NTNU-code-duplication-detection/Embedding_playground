"""
Module for IO
"""
import json
from pathlib import Path

class JsonlWriter:
    """
    Class for writing jsonl
    """
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")

    def write(self, obj: dict):
        """
        Write jsonl
        """
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def close(self):
        """
        Close path
        """
        self.f.close()
