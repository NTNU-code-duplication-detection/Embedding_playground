"""
Cli module
"""
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List

from program_project.config import ProgramProjectConfig
from program_project.scan_dataset import scan_programs
from program_project.build_program import build_program_artifacts
from program_project.run_stages import run_decompiler, run_ast_dataset, run_embed_cache
from program_project.util import write_text


def main() -> int:
    """Main entry point"""
    ap = argparse.ArgumentParser("Build program-level artifacts for code-clone dataset")
    ap.add_argument(
        "--dataset-root",
        required=True,
        help=(
            "Synthetic: .../dataset (contains base/, type-1/, type-2/, type-3/). "
            "GoogleJam: root containing numeric problem dirs (1/,2/,...) with .java files."
        ),
    )
    ap.add_argument(
        "--clone-type",
        default="type-3",
        choices=["type-1", "type-2", "type-3", "googlejam"],
        help="Dataset mode: synthetic clone types (type-1/type-2/type-3) or googlejam",
    )
    ap.add_argument("--out", default="./program_artifacts", help="output artifact root")

    # Only required for synthetic pipeline (we compile+decompile). For googlejam we can run source-only.
    ap.add_argument("--jdk-home", default=None)
    ap.add_argument("--vineflower", default=None)

    ap.add_argument("--model", default="microsoft/graphcodebert-base")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--shard-size", type=int, default=2000)

    ap.add_argument("--node-cache", default="./node_cache")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit-indices", type=int, default=None)

    args = ap.parse_args()

    if args.clone_type != "googlejam" and (args.jdk_home is None or args.vineflower is None):
        raise SystemExit("Error: --jdk-home and --vineflower are required unless --clone-type is 'googlejam'")

    cfg = ProgramProjectConfig(
        dataset_root=Path(args.dataset_root).expanduser().resolve(),
        clone_type=args.clone_type,
        out_dir=Path(args.out).expanduser().resolve(),
        jdk_home=(Path(args.jdk_home).expanduser().resolve() if args.jdk_home else None),
        vineflower_jar=(Path(args.vineflower).expanduser().resolve() if args.vineflower else None),
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shard_size=args.shard_size,
        node_cache_dir=Path(args.node_cache).expanduser().resolve(),
        force_rebuild=bool(args.force),
        limit_indices=args.limit_indices,
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.node_cache_dir.mkdir(parents=True, exist_ok=True)

    programs = scan_programs(cfg)

    # Build list of all unique java program files we need artifacts for
    all_files: List[Path] = []
    for e in programs.values():
        all_files.append(e.anchor)
        all_files.extend(e.clones)

    # deterministic order
    all_files = sorted({p.resolve() for p in all_files}, key=lambda p: str(p))

    index: Dict[str, dict] = {}
    failures: List[dict] = []

    # We run from the repo root context of "pipeline", so cwd should be pipeline/
    out_dir = Path(".").resolve()

    for p in all_files:
        key = str(p)
        try:
            if cfg.clone_type == "googlejam":
                # Source-only artifacts (no compilation/decompilation). This avoids dependency issues.
                sha1 = hashlib.sha1(key.encode("utf-8")).hexdigest()
                workdir = cfg.out_dir / f"prog_{sha1}"
                workdir.mkdir(parents=True, exist_ok=True)

                src_single_dir = workdir / "src_single"
                src_single_dir.mkdir(parents=True, exist_ok=True)
                # Normalize file name; class-name detection happens in build_program for synthetic.
                dest_file = src_single_dir / "Main.java"
                shutil.copy2(p, dest_file)

                shards_dir = workdir / "embed_cache" / "shards"
                if cfg.force_rebuild or not shards_dir.exists():
                    methods_jsonl = run_ast_dataset(workdir=workdir, decompiled_src_dir=src_single_dir, out_dir=out_dir)
                    shards_dir = run_embed_cache(
                        workdir=workdir,
                        methods_jsonl=methods_jsonl,
                        out_dir=out_dir,
                        node_cache_dir=cfg.node_cache_dir,
                        model_name=cfg.model_name,
                        device=cfg.device,
                        batch_size=cfg.batch_size,
                        max_length=cfg.max_length,
                        shard_size=cfg.shard_size,
                    )

                index[key] = {
                    "source_path": key,
                    "artifact_dir": str(workdir),
                    "shards_dir": str(shards_dir),
                }
                continue

            # Synthetic dataset path: compile -> jar -> decompile -> ast -> embed
            if cfg.jdk_home is None or cfg.vineflower_jar is None:
                raise RuntimeError("Internal error: jdk_home/vineflower_jar missing for synthetic run")

            art = build_program_artifacts(
                out_dir=cfg.out_dir,
                source_path=p,
                jdk_home=cfg.jdk_home,
                force=cfg.force_rebuild,
            )
            workdir = Path(art["workdir"])

            shards_dir = workdir / "embed_cache" / "shards"
            if cfg.force_rebuild or not shards_dir.exists():
                decomp_src = run_decompiler(
                    workdir=workdir,
                    compile_manifest=Path(art["compile_manifest"]),
                    out_dir=out_dir,
                    jdk_home=cfg.jdk_home,
                    vineflower_jar=cfg.vineflower_jar,
                )
                methods_jsonl = run_ast_dataset(workdir=workdir, decompiled_src_dir=decomp_src, out_dir=out_dir)
                shards_dir = run_embed_cache(
                    workdir=workdir,
                    methods_jsonl=methods_jsonl,
                    out_dir=out_dir,
                    node_cache_dir=cfg.node_cache_dir,
                    model_name=cfg.model_name,
                    device=cfg.device,
                    batch_size=cfg.batch_size,
                    max_length=cfg.max_length,
                    shard_size=cfg.shard_size,
                )

            index[key] = {
                "source_path": key,
                "artifact_dir": str(workdir),
                "shards_dir": str(shards_dir),
            }
        except Exception as ex:
            failures.append({"source_path": key, "error": str(ex)})

    index_path = cfg.out_dir / "program_index.json"
    write_text(index_path, json.dumps({"items": index, "failures": failures}, indent=2))

    print(f"Wrote program index: {index_path}")
    print(f"Programs ok: {len(index)}  failed: {len(failures)}")
    if failures:
        print("First failure:", failures[0])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
