"""
Compiler utilities module
"""
from __future__ import annotations
import os
import subprocess
from pathlib import Path
from typing import List, Tuple


def with_jdk_env(jdk_home: str) -> dict:
    """
    Forces subprocesses to use a specific JDK (JDK21).
    """
    env = dict(os.environ)
    if jdk_home:
        env["JAVA_HOME"] = jdk_home
        env["PATH"] = str(Path(jdk_home) / "bin") + os.pathsep + env.get("PATH", "")
    return env


def run(cmd: List[str], cwd: Path, env: dict) -> Tuple[int, str, str]:
    """
    Executes a process and captures stdout/stderr.
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
