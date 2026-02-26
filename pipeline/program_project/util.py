"""
Util
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def sha1_text(s: str) -> str:
    """
    sha text
    """
    return hashlib.sha1(s.encode("utf-8", errors="replace")).hexdigest()


def read_text(p: Path) -> str:
    """
    Read text
    """
    return p.read_text(encoding="utf-8", errors="replace")


def write_text(p: Path, s: str) -> None:
    """
    Write text
    """
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def run(cmd: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return p.returncode, p.stdout, p.stderr


_CLASS_RE = re.compile(r"^\s*public\s+class\s+([A-Za-z_]\w*)\b", re.MULTILINE)


def detect_public_class_name(java_source: str) -> Optional[str]:
    """
    Returns the first 'public class X' name, else None.
    This avoids file-name mismatch errors in javac.
    """
    m = _CLASS_RE.search(java_source)
    if not m:
        return None
    return m.group(1)


def java_bin(jdk_home: Path) -> Path:
    """
    Get java from jdk bin
    """
    return jdk_home / "bin" / "java"


def javac_bin(jdk_home: Path) -> Path:
    """
    Get javac from bin
    """
    return jdk_home / "bin" / "javac"


def jar_bin(jdk_home: Path) -> Path:
    """
    Get jar from bin
    """
    return jdk_home / "bin" / "jar"
