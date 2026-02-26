"""
Project handler detection module
Supports Maven, Gradle and pure JavaC
"""
from pathlib import Path


def detect_build_tool(project_dir: Path) -> str:
    """
    Determines how the project should be compiled.
    """
    if (project_dir / "pom.xml").exists():
        return "maven"

    if (
        (project_dir / "gradlew").exists()
        or (project_dir / "build.gradle").exists()
        or (project_dir / "build.gradle.kts").exists()
    ):
        return "gradle"

    return "javac"
