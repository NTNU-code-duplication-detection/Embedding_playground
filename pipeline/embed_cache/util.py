"""
Module for utils for caching
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterator

def sha1_text(s: str) -> str:
    """
    Hash string
    """
    return hashlib.sha1(s.encode("utf-8", errors="replace")).hexdigest()

def read_jsonl(path: Path) -> Iterator[Dict]:
    """
    Read jsonl stream from path
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def ensure_dir(p: Path) -> None:
    """
    Ensure dir exists
    """
    p.mkdir(parents=True, exist_ok=True)

def write_json(path: Path, obj: Dict) -> None:
    """
    Write json
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
