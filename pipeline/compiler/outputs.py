"""
Module for outputs
"""
from pathlib import Path
from typing import List, Tuple


def find_outputs(project_dir: Path) -> Tuple[List[Path], List[Path]]:
    """
    Returns:
        class_dirs: directories containing .class files
        jars: built jar artifacts
    """

    class_dirs: List[Path] = []
    jars: List[Path] = []

    # ---------- Maven ----------
    target = project_dir / "target"
    if target.exists():
        classes = target / "classes"
        if classes.is_dir():
            class_dirs.append(classes.resolve())

        for j in target.glob("*.jar"):
            jars.append(j.resolve())

    # ---------- Gradle ----------
    build = project_dir / "build"
    if build.exists():
        main = build / "classes" / "java" / "main"
        if main.is_dir():
            class_dirs.append(main.resolve())

        libs = build / "libs"
        if libs.exists():
            for j in libs.glob("*.jar"):
                jars.append(j.resolve())

    # ---------- Multi-module scan ----------
    for sub in project_dir.rglob("target/classes"):
        if sub.is_dir():
            class_dirs.append(sub.resolve())

    for sub in project_dir.rglob("build/classes/java/main"):
        if sub.is_dir():
            class_dirs.append(sub.resolve())

    for sub in project_dir.rglob("build/libs"):
        if sub.is_dir():
            for j in sub.glob("*.jar"):
                jars.append(j.resolve())

    # deduplicate
    class_dirs = sorted(set(class_dirs))
    jars = sorted(set(jars))

    return class_dirs, jars
