"""
Module for building program
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from program_project.util import (
    detect_public_class_name,
    jar_bin,
    javac_bin,
    read_text,
    run,
    sha1_text,
    write_text,
)

def prepare_workdir(out_dir: Path, source_path: Path) -> Path:
    # Deterministic workdir name from absolute path
    h = sha1_text(str(source_path.resolve()))
    wd = out_dir / f"prog_{h}"
    wd.mkdir(parents=True, exist_ok=True)
    return wd


def stage_source(workdir: Path, source_path: Path) -> Path:
    """
    Copies source into workdir/src/<ClassName>.java (or Main.java fallback).
    Returns staged java file path.
    """
    src_dir = workdir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    text = read_text(source_path)
    cls = detect_public_class_name(text) or "Main"
    staged = src_dir / f"{cls}.java"
    write_text(staged, text)
    return staged


def compile_with_javac(workdir: Path, staged_java: Path, jdk_home: Path) -> Tuple[bool, str, str, Path]:
    """
    Compile java file with javac
    """
    classes_dir = workdir / "build" / "classes"
    classes_dir.mkdir(parents=True, exist_ok=True)

    javac = str(javac_bin(jdk_home))
    cmd = [javac, "-encoding", "UTF-8", "-d", str(classes_dir), str(staged_java)]

    rc, out, err = run(cmd, cwd=workdir)
    return (rc == 0), out, err, classes_dir


def package_jar(workdir: Path, classes_dir: Path, jdk_home: Path) -> Tuple[bool, str, str, Path]:
    jar_dir = workdir / "build"
    jar_dir.mkdir(parents=True, exist_ok=True)
    jar_path = jar_dir / "program.jar"

    jar = str(jar_bin(jdk_home))
    # jar --create --file program.jar -C classes_dir .
    cmd = [jar, "--create", "--file", str(jar_path), "-C", str(classes_dir), "."]

    rc, out, err = run(cmd, cwd=workdir)
    return (rc == 0), out, err, jar_path


def write_compile_manifest(workdir: Path, classes_dir: Path, jar_path: Path, success: bool, stdout: str, stderr: str) -> Path:
    manifest = {
        "tool": "javac",
        "success": bool(success),
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
        "class_dirs": [str(classes_dir)],
        "jars": [str(jar_path)],
    }
    path = workdir / "compile_manifest.json"
    write_text(path, json.dumps(manifest, indent=2))
    return path


def build_program_artifacts(
    *,
    out_dir: Path,
    source_path: Path,
    jdk_home: Path,
    force: bool = False,
) -> Dict[str, str]:
    """
    Returns dict with keys:
      workdir, staged_java, classes_dir, jar_path, compile_manifest
    """
    workdir = prepare_workdir(out_dir, source_path)

    compile_manifest = workdir / "compile_manifest.json"
    jar_path = workdir / "build" / "program.jar"
    classes_dir = workdir / "build" / "classes"

    if not force and compile_manifest.exists() and jar_path.exists() and classes_dir.exists():
        return {
            "workdir": str(workdir),
            "staged_java": str(next((workdir / "src").glob("*.java"), workdir / "src" / "Main.java")),
            "classes_dir": str(classes_dir),
            "jar_path": str(jar_path),
            "compile_manifest": str(compile_manifest),
        }

    staged_java = stage_source(workdir, source_path)
    ok, out1, err1, classes_dir = compile_with_javac(workdir, staged_java, jdk_home)
    if not ok:
        write_compile_manifest(workdir, classes_dir, jar_path, False, out1, err1)
        raise RuntimeError(f"javac failed for {source_path}\n{err1[-2000:]}")

    ok2, out2, err2, jar_path = package_jar(workdir, classes_dir, jdk_home)
    stdout = (out1 or "") + "\n" + (out2 or "")
    stderr = (err1 or "") + "\n" + (err2 or "")

    if not ok2:
        write_compile_manifest(workdir, classes_dir, jar_path, False, stdout, stderr)
        raise RuntimeError(f"jar packaging failed for {source_path}\n{stderr[-2000:]}")

    mf = write_compile_manifest(workdir, classes_dir, jar_path, True, stdout, stderr)
    return {
        "workdir": str(workdir),
        "staged_java": str(staged_java),
        "classes_dir": str(classes_dir),
        "jar_path": str(jar_path),
        "compile_manifest": str(mf),
    }
