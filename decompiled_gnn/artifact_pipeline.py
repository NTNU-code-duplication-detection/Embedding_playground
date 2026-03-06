"""Optional compile/decompile pipeline for Java files or projects."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Optional

from cache import CacheManager
from config import CompilationConfig


_CLASS_RE = re.compile(r"^\s*public\s+class\s+([A-Za-z_]\w*)\b", re.MULTILINE)


@dataclass
class CommandResult:
    """Captured command result."""

    returncode: int
    stdout: str
    stderr: str


def _tail(text: str, n: int = 4000) -> str:
    return text if len(text) <= n else text[-n:]


def _with_jdk_env(java_home: Optional[Path]) -> dict[str, str]:
    env = dict(os.environ)
    if java_home:
        env["JAVA_HOME"] = str(java_home)
        env["PATH"] = str(java_home / "bin") + os.pathsep + env.get("PATH", "")
    return env


def _run(cmd: list[str], cwd: Path, env: dict[str, str], timeout_sec: int) -> CommandResult:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_sec,
    )
    return CommandResult(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


def _detect_public_class_name(java_source: str) -> Optional[str]:
    match = _CLASS_RE.search(java_source)
    return match.group(1) if match else None


def _detect_build_tool(project_dir: Path) -> str:
    if (project_dir / "pom.xml").exists():
        return "maven"
    if (project_dir / "build.gradle").exists() or (project_dir / "build.gradle.kts").exists():
        return "gradle"
    return "javac"


def _find_outputs(project_dir: Path) -> tuple[list[str], list[str]]:
    class_dirs: list[Path] = []
    for candidate in [
        project_dir / "target" / "classes",
        project_dir / "build" / "classes",
        project_dir / "build" / "classes" / "java" / "main",
    ]:
        if candidate.exists():
            class_dirs.append(candidate.resolve())

    jars = [
        jar.resolve()
        for jar in project_dir.rglob("*.jar")
        if jar.is_file()
        and "sources" not in jar.name
        and "javadoc" not in jar.name
        and "tests" not in jar.name
    ]

    return [str(path) for path in class_dirs], [str(path) for path in jars]


class CompileDecompilePipeline:
    """Build compilation/decompilation artifacts and cache manifests."""

    def __init__(self, cfg: CompilationConfig, cache: CacheManager):
        self.cfg = cfg
        self.cache = cache

    def process(self, source_path: Path) -> dict[str, Any]:
        """Run optional compile/decompile and return an artifact manifest."""

        source_path = source_path.expanduser().resolve()
        program_dir = self.cache.ensure_program_dirs(source_path)
        manifest_path = self.cache.artifact_manifest_path(source_path)

        if manifest_path.exists() and not self.cfg.force_rebuild:
            return self.cache.read_json(manifest_path)

        manifest: dict[str, Any] = {
            "source_path": str(source_path),
            "program_dir": str(program_dir),
            "compilation_enabled": bool(self.cfg.enabled),
            "compile": None,
            "decompile": None,
            "graph_input_path": str(source_path),
        }

        if not self.cfg.enabled:
            self.cache.write_json(manifest_path, manifest)
            return manifest

        compile_result = self._compile(source_path, program_dir)
        manifest["compile"] = compile_result

        if not compile_result.get("success", False):
            self.cache.write_json(manifest_path, manifest)
            return manifest

        decompile_result = self._decompile(compile_result, program_dir)
        manifest["decompile"] = decompile_result

        decompiled_root = Path(decompile_result.get("out_src_dir", "")) if decompile_result else None
        if decompiled_root and decompiled_root.exists():
            java_candidates = sorted(decompiled_root.rglob("*.java"))
            if source_path.is_file() and java_candidates:
                manifest["graph_input_path"] = str(java_candidates[0])
            else:
                manifest["graph_input_path"] = str(decompiled_root)

        self.cache.write_json(manifest_path, manifest)
        return manifest

    def _compile(self, source_path: Path, program_dir: Path) -> dict[str, Any]:
        env = _with_jdk_env(self.cfg.java_home)
        compiled_dir = program_dir / "compiled"
        compiled_dir.mkdir(parents=True, exist_ok=True)

        if source_path.is_file() and source_path.suffix.lower() == ".java":
            return self._compile_single_file(source_path, compiled_dir, env)
        if source_path.is_dir():
            return self._compile_project(source_path, compiled_dir, env)

        return {
            "tool": "none",
            "success": False,
            "error": f"Unsupported source path: {source_path}",
            "class_dirs": [],
            "jars": [],
        }

    def _compile_single_file(
        self,
        source_path: Path,
        compiled_dir: Path,
        env: dict[str, str],
    ) -> dict[str, Any]:
        staged_dir = compiled_dir / "staged_src"
        classes_dir = compiled_dir / "classes"
        staged_dir.mkdir(parents=True, exist_ok=True)
        classes_dir.mkdir(parents=True, exist_ok=True)

        source_text = source_path.read_text(encoding="utf-8", errors="replace")
        class_name = _detect_public_class_name(source_text) or "Main"
        staged_java = staged_dir / f"{class_name}.java"
        staged_java.write_text(source_text, encoding="utf-8")

        javac_bin = (
            str(self.cfg.java_home / "bin" / "javac")
            if self.cfg.java_home
            else "javac"
        )
        jar_bin = str(self.cfg.java_home / "bin" / "jar") if self.cfg.java_home else "jar"

        javac_cmd = [javac_bin, "-encoding", "UTF-8", "-g", "-d", str(classes_dir), str(staged_java)]
        javac_res = _run(javac_cmd, cwd=compiled_dir, env=env, timeout_sec=self.cfg.compile_timeout_sec)

        jar_path = compiled_dir / "program.jar"
        jar_success = False
        jar_out = ""
        jar_err = ""
        if javac_res.returncode == 0:
            jar_cmd = [jar_bin, "--create", "--file", str(jar_path), "-C", str(classes_dir), "."]
            jar_res = _run(jar_cmd, cwd=compiled_dir, env=env, timeout_sec=self.cfg.compile_timeout_sec)
            jar_success = jar_res.returncode == 0
            jar_out = jar_res.stdout
            jar_err = jar_res.stderr

        success = javac_res.returncode == 0 and jar_success
        return {
            "tool": "javac",
            "success": success,
            "staged_java": str(staged_java),
            "class_dirs": [str(classes_dir.resolve())] if classes_dir.exists() else [],
            "jars": [str(jar_path.resolve())] if jar_path.exists() else [],
            "stdout_tail": _tail(javac_res.stdout + "\n" + jar_out),
            "stderr_tail": _tail(javac_res.stderr + "\n" + jar_err),
        }

    def _compile_project(
        self,
        project_dir: Path,
        compiled_dir: Path,
        env: dict[str, str],
    ) -> dict[str, Any]:
        tool = _detect_build_tool(project_dir)

        stdout = ""
        stderr = ""
        success = False

        if tool == "maven":
            cmd = ["mvn", "-q", "-DskipTests", "-Dmaven.test.skip=true", "package"]
            res = _run(cmd, cwd=project_dir, env=env, timeout_sec=self.cfg.compile_timeout_sec)
            stdout, stderr, success = res.stdout, res.stderr, res.returncode == 0

        elif tool == "gradle":
            if (project_dir / "gradlew").exists():
                gradlew = project_dir / "gradlew"
                gradlew.chmod(gradlew.stat().st_mode | 0o111)
                cmd = [str(gradlew), "-q", "assemble", "-x", "test"]
            else:
                cmd = ["gradle", "-q", "assemble", "-x", "test"]
            res = _run(cmd, cwd=project_dir, env=env, timeout_sec=self.cfg.compile_timeout_sec)
            stdout, stderr, success = res.stdout, res.stderr, res.returncode == 0

        else:
            javac_bin = (
                str(self.cfg.java_home / "bin" / "javac")
                if self.cfg.java_home
                else "javac"
            )
            sources = sorted(project_dir.rglob("*.java"))
            classes_dir = compiled_dir / "javac_classes"
            classes_dir.mkdir(parents=True, exist_ok=True)
            if not sources:
                return {
                    "tool": "javac",
                    "success": False,
                    "error": f"No Java sources under {project_dir}",
                    "class_dirs": [],
                    "jars": [],
                }
            cmd = [javac_bin, "-g", "-d", str(classes_dir)] + [str(src) for src in sources]
            res = _run(cmd, cwd=project_dir, env=env, timeout_sec=self.cfg.compile_timeout_sec)
            stdout, stderr, success = res.stdout, res.stderr, res.returncode == 0

        class_dirs, jars = _find_outputs(project_dir)
        if tool == "javac" and success:
            fallback_classes = compiled_dir / "javac_classes"
            if fallback_classes.exists():
                class_dirs = [str(fallback_classes.resolve())]

        return {
            "tool": tool,
            "success": success,
            "class_dirs": class_dirs,
            "jars": jars,
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
        }

    def _decompile(self, compile_result: dict[str, Any], program_dir: Path) -> dict[str, Any]:
        env = _with_jdk_env(self.cfg.java_home)
        vineflower = self.cfg.vineflower_jar.expanduser().resolve()
        out_src_dir = program_dir / "decompiled"

        if not vineflower.exists():
            return {
                "success": False,
                "error": f"Vineflower jar not found at {vineflower}",
                "out_src_dir": str(out_src_dir),
            }

        jars = [Path(path) for path in compile_result.get("jars", [])]
        class_dirs = [Path(path) for path in compile_result.get("class_dirs", [])]

        if self.cfg.prefer_jars and jars:
            inputs = jars
        elif class_dirs:
            inputs = class_dirs
        else:
            inputs = jars

        if out_src_dir.exists():
            shutil.rmtree(out_src_dir)
        out_src_dir.mkdir(parents=True, exist_ok=True)

        all_stdout = []
        all_stderr = []
        success = True

        for inp in inputs:
            inp = inp.expanduser().resolve()
            if not inp.exists():
                success = False
                all_stderr.append(f"Input not found: {inp}")
                continue

            name = inp.stem if inp.suffix == ".jar" else inp.name
            dest = out_src_dir / name
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True, exist_ok=True)

            cmd = ["java", "-jar", str(vineflower), str(inp), str(dest)]
            res = _run(cmd, cwd=out_src_dir, env=env, timeout_sec=self.cfg.decompile_timeout_sec)
            all_stdout.append(res.stdout)
            all_stderr.append(res.stderr)
            if res.returncode != 0:
                success = False

        return {
            "success": success,
            "inputs": [str(path) for path in inputs],
            "out_src_dir": str(out_src_dir.resolve()),
            "stdout_tail": _tail("\n".join(all_stdout)),
            "stderr_tail": _tail("\n".join(all_stderr)),
        }
