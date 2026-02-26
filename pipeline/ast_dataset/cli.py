#!/usr/bin/env python3
"""
CLI module for testing ast parsing
"""
import argparse
from pathlib import Path

from ast_dataset.config import DatasetConfig
from ast_dataset.scan import find_java_files
from ast_dataset.parse_java import make_parser
from ast_dataset.extract_methods import extract_methods
from ast_dataset.build_graph import build_method_chunk_graph
from ast_dataset.io import JsonlWriter


def run(cfg: DatasetConfig):
    """
    Run parser
    """
    parser = make_parser()
    writer = JsonlWriter(cfg.out_root / "methods.jsonl")

    java_files = find_java_files(cfg.src_root)

    for file in java_files:
        try:
            src = file.read_bytes()
            tree = parser.parse(src)

            methods = extract_methods(tree, src, file)

            for m in methods:
                g = build_method_chunk_graph(m["node"], src, undirected=True, add_if_role_edges=True)

                if len(g["nodes"]) == 0:
                    continue  # drop empty methods

                record = {
                    "method_id": m["method_id"],
                    "method_name": m["method_name"],
                    "file": str(file),
                    "nodes": g["nodes"],    # includes code for embedding later
                    "edges": g["edges"],    # src/dst/type with only SEQ/AST/IF_THEN/IF_ELSE
                }

                writer.write(record)

        except Exception as e:
            print("FAILED:", file, e)

    writer.close()


def main():
    """
    Main entry point
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="src_decompiled directory")
    ap.add_argument("--out", required=True, help="output dataset directory")
    args = ap.parse_args()

    cfg = DatasetConfig(Path(args.src), Path(args.out))
    run(cfg)


if __name__ == "__main__":
    main()
