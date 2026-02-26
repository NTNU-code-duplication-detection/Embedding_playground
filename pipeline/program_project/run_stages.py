"""
Run stages module
"""
from __future__ import annotations

import json
from pathlib import Path

from program_project.util import run


def run_decompiler(workdir: Path, compile_manifest: Path, out_dir: Path, jdk_home: Path, vineflower_jar: Path) -> Path:
    """
    Calls your existing: python -m decompiler.cli ...
    Returns decompiled source directory path.
    """
    decomp_out = workdir / "decompiled_out"
    decomp_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3", "-m", "decompiler.cli",
        "--manifest", str(compile_manifest),
        "--out", str(decomp_out),
        "--jdk-home", str(jdk_home),
        "--vineflower", str(vineflower_jar),
        "--prefer-jars",
    ]
    rc, out, err = run(cmd, cwd=out_dir)
    if rc != 0:
        raise RuntimeError(f"decompiler.cli failed\n{err[-2000:]}\n{out[-2000:]}")

    # decompiler writes decompile_manifest.json in its --out
    decomp_manifest = decomp_out / "decompile_manifest.json"
    if not decomp_manifest.exists():
        raise RuntimeError("decompiler did not produce decompile_manifest.json")

    j = json.loads(decomp_manifest.read_text(encoding="utf-8"))
    out_src_dir = Path(j["out_src_dir"])
    return out_src_dir


def run_ast_dataset(workdir: Path, decompiled_src_dir: Path, out_dir: Path) -> Path:
    """
    Calls your existing: python -m ast_dataset.cli ...
    Produces methods JSONL somewhere under an output directory.
    Returns the resolved JSONL path.
    """
    # Use an explicit directory to avoid confusion
    ast_out_dir = workdir / "ast_out"
    ast_out_dir.mkdir(parents=True, exist_ok=True)

    # Call AST stage (keep flags as-is; we’ll handle both file/dir outputs)
    cmd = [
        "python3", "-m", "ast_dataset.cli",
        "--src", str(decompiled_src_dir),
        "--out", str(ast_out_dir),   # IMPORTANT: now always a directory
    ]
    rc, out, err = run(cmd, cwd=out_dir)
    if rc != 0:
        raise RuntimeError(f"ast_dataset.cli failed\n{err[-2000:]}\n{out[-2000:]}")

    # Find a jsonl inside ast_out_dir
    candidates = sorted(ast_out_dir.rglob("*.jsonl"))
    if not candidates:
        raise RuntimeError(f"ast_dataset produced no *.jsonl under {ast_out_dir}")

    # Prefer a file literally named methods.jsonl if present
    for c in candidates:
        if c.name == "methods.jsonl":
            return c

    # Else return the first jsonl (deterministic due to sorted())
    return candidates[0]


def run_embed_cache(
    workdir: Path,
    methods_jsonl: Path,
    out_dir: Path,
    node_cache_dir: Path,
    model_name: str,
    device: str,
    batch_size: int,
    max_length: int,
    shard_size: int,
) -> Path:
    """
    Calls your existing: python -m embed_cache.cli ...
    Returns shards directory path.
    """
    embed_out = workdir / "embed_cache"
    embed_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3", "-m", "embed_cache.cli",
        "--in-jsonl", str(methods_jsonl),
        "--out", str(embed_out),
        "--cache", str(node_cache_dir),
        "--model", model_name,
        "--device", device,
        "--batch-size", str(batch_size),
        "--max-length", str(max_length),
        "--shard-size", str(shard_size),
    ]
    rc, out, err = run(cmd, cwd=out_dir)
    if rc != 0:
        raise RuntimeError(f"embed_cache.cli failed\n{err[-2000:]}\n{out[-2000:]}")

    shards_dir = embed_out / "shards"
    if not shards_dir.exists():
        raise RuntimeError("embed_cache did not produce shards/")

    return shards_dir
