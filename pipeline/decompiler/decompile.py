"""
Decompile module
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from decompiler.config import DecompilerConfig
from decompiler.util import ensure_dir, run, tail, with_jdk_env


def _as_paths(xs: List[str]) -> List[Path]:
    """
    Resolve path given as list
    """
    return [Path(x).expanduser().resolve() for x in xs]


def read_compile_manifest(manifest_path: Path) -> Dict:
    """
    Read content of manifest
    """
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "class_dirs" not in data or "jars" not in data:
        raise ValueError("Manifest missing required keys: class_dirs, jars")
    return data


def select_inputs_from_manifest(manifest: Dict, prefer_jars: bool = True) -> List[Path]:
    """
    Find entrypoint for decompilation
    """
    jars = _as_paths(manifest.get("jars", []))
    class_dirs = _as_paths(manifest.get("class_dirs", []))

    if prefer_jars and jars:
        return jars
    if class_dirs:
        return class_dirs
    return jars


def decompile_artifacts(
    inputs: List[Path],
    out_src_dir: Path,
    cfg: DecompilerConfig,
    extra_vineflower_args: Optional[List[str]] = None,
) -> Dict:
    """
    Decompile each input (jar or class dir) into out_src_dir/<input_name>/...
    Returns a manifest-like dict with logs.
    """
    env = with_jdk_env(cfg.jdk_home)
    vf = Path(cfg.vineflower_jar).expanduser().resolve()

    if not vf.exists():
        return {"success": False, "error": f"Vineflower jar not found: {vf}"}

    ensure_dir(out_src_dir)

    extra_vineflower_args = extra_vineflower_args or []

    all_out: List[str] = []
    all_err: List[str] = []
    ok = True

    # One invocation per artifact to avoid collisions and simplify debugging
    for inp in inputs:
        inp = inp.resolve()
        if not inp.exists():
            ok = False
            all_err.append(f"Input not found: {inp}")
            continue

        name = inp.name
        if name.endswith(".jar"):
            name = inp.stem  # drops .jar
        dest = out_src_dir / name
        ensure_dir(dest)

        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)

        cmd = ["java", "-jar", str(vf)] + extra_vineflower_args + [str(inp), str(dest)]
        rc, out, err = run(cmd, cwd=out_src_dir, env=env)
        all_out.append(out)
        all_err.append(err)

        if rc != 0:
            ok = False

    return {
        "success": ok,
        "inputs": [str(p.resolve()) for p in inputs],
        "out_src_dir": str(out_src_dir.resolve()),
        "stdout_tail": tail("\n".join(all_out)),
        "stderr_tail": tail("\n".join(all_err)),
    }


def decompile_from_manifest(
    compile_manifest_path: Path,
    out_src_dir: Path,
    cfg: DecompilerConfig,
    prefer_jars: bool = True,
    extra_vineflower_args: Optional[List[str]] = None,
) -> Dict:
    """
    Decompiles based on manifest info
    """
    manifest = read_compile_manifest(compile_manifest_path)
    inputs = select_inputs_from_manifest(manifest, prefer_jars=prefer_jars)
    result = decompile_artifacts(
        inputs=inputs,
        out_src_dir=out_src_dir,
        cfg=cfg,
        extra_vineflower_args=extra_vineflower_args,
    )
    result["compile_manifest"] = str(compile_manifest_path.resolve())
    result["prefer_jars"] = prefer_jars
    return result
