#!/usr/bin/env python3
"""
CLI module for testing
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from decompiler.config import DecompilerConfig
from decompiler.decompile import decompile_artifacts, decompile_from_manifest
from decompiler.util import ensure_dir


def main() -> int:
    """
    Main entry point
    """
    ap = argparse.ArgumentParser("Vineflower decompiler")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="Path to compile_manifest.json from compiler stage.")
    src.add_argument(
        "--input",
        action="append",
        help="Repeatable: artifact path (jar or class dir)."
        )
    ap.add_argument(
        "--out",
        required=True,
        help="Output folder. Will create src_decompiled inside."
        )
    ap.add_argument(
        "--jdk-home",
        required=True,
        help="JAVA_HOME for JDK21."
        )
    ap.add_argument(
        "--vineflower",
        required=True,
        help="Path to vineflower.jar."
        )
    ap.add_argument(
        "--prefer-jars",
        action="store_true",
        help="When using --manifest, prefer jars if available."
        )
    ap.add_argument(
        "--vf-arg",
        action="append",
        default=[],
        help="Repeatable: extra Vineflower arg (advanced)."
        )

    args = ap.parse_args()

    out_root = Path(args.out).expanduser().resolve()
    ensure_dir(out_root)
    out_src = out_root / "src_decompiled"
    ensure_dir(out_src)

    cfg = DecompilerConfig(jdk_home=args.jdk_home, vineflower_jar=args.vineflower)

    if args.manifest:
        res = decompile_from_manifest(
            compile_manifest_path=Path(args.manifest).expanduser().resolve(),
            out_src_dir=out_src,
            cfg=cfg,
            prefer_jars=args.prefer_jars,
            extra_vineflower_args=args.vf_arg,
        )
    else:
        inputs = [Path(p).expanduser().resolve() for p in (args.input or [])]
        res = decompile_artifacts(
            inputs=inputs,
            out_src_dir=out_src,
            cfg=cfg,
            extra_vineflower_args=args.vf_arg,
        )

    (out_root / "decompile_manifest.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("SUCCESS" if res["success"] else "FAILED")
    return 0 if res["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
