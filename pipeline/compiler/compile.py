"""
Module for compilation
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from compiler.config import CompilerConfig
from compiler.detect import detect_build_tool
from compiler.outputs import find_outputs
from compiler.util import run, tail, with_jdk_env


def compile_project(project_dir: Path, out_dir: Path, cfg: CompilerConfig) -> Dict:
    """
    Compile entire project based on
    Project dir
    Output dir
    Compiler config
    """
    env = with_jdk_env(cfg.jdk_home)
    tool = detect_build_tool(project_dir)

    stdout = ""
    stderr = ""
    success = False

    # ---------------- Maven ----------------
    if tool == "maven":
        cmd = ["mvn", "-q", "-DskipTests", "-Dmaven.test.skip=true", "package"]
        rc, stdout, stderr = run(cmd, project_dir, env)
        success = rc == 0

    # ---------------- Gradle ----------------
    elif tool == "gradle":
        if (project_dir / "gradlew").exists():
            gradlew = project_dir / "gradlew"
            gradlew.chmod(gradlew.stat().st_mode | 0o111)
            cmd = [str(gradlew), "-q", "assemble", "-x", "test"]
        else:
            cmd = ["gradle", "-q", "assemble", "-x", "test"]

        rc, stdout, stderr = run(cmd, project_dir, env)
        success = rc == 0

    # ---------------- Javac fallback ----------------
    else:
        sources: List[Path] = list(project_dir.rglob("*.java"))
        if not sources:
            return {
                "tool": "javac",
                "success": False,
                "error": "No Java files found",
            }

        classes_out = out_dir / "javac_classes"
        classes_out.mkdir(parents=True, exist_ok=True)

        cmd = ["javac", "-g", "-d", str(classes_out)] + [str(s) for s in sources]
        rc, stdout, stderr = run(cmd, project_dir, env)
        success = rc == 0

    class_dirs, jars = find_outputs(project_dir)

    # if javac succeeded, include our output directory
    if tool == "javac" and success:
        class_dirs = [str((out_dir / "javac_classes").resolve())]

    return {
        "tool": tool,
        "success": success,
        "stdout_tail": tail(stdout),
        "stderr_tail": tail(stderr),
        "class_dirs": [str(p) for p in class_dirs],
        "jars": [str(p) for p in jars],
    }
