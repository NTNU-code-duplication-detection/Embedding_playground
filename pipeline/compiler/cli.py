#!/usr/bin/env python3
"""
CLI module
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from compiler.config import CompilerConfig
from compiler.compile import compile_project


def main() -> int:
    """
    Main entry point
    """
    ap = argparse.ArgumentParser("Java project compiler")
    ap.add_argument("--project", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jdk-home", required=True)
    args = ap.parse_args()

    project = Path(args.project).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    cfg = CompilerConfig(jdk_home=args.jdk_home)

    result = compile_project(project, out, cfg)

    manifest = out / "compile_manifest.json"
    manifest.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("SUCCESS" if result["success"] else "FAILED")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
