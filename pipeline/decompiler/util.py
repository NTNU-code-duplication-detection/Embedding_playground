"""
Utils for decompilation module
"""
from __future__ import annotations
import os
import subprocess
from pathlib import Path
from typing import List, Tuple


def with_jdk_env(jdk_home: str) -> dict:
    """
    Get ENV with JDK home
    """
    env = dict(os.environ)
    if jdk_home:
        env["JAVA_HOME"] = jdk_home
        env["PATH"] = str(Path(jdk_home) / "bin") + os.pathsep + env.get("PATH", "")
    return env


def run(cmd: List[str], cwd: Path, env: dict) -> Tuple[int, str, str]:
    """
    Create subprocess of cmd
    """
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return p.returncode, p.stdout, p.stderr


def tail(text: str, n: int = 4000) -> str:
    """
    Tail logic
    """
    return text if len(text) <= n else text[-n:]


def ensure_dir(p: Path) -> None:
    """
    Ensure path exists
    """
    p.mkdir(parents=True, exist_ok=True)
