"""Module pipeline/augment_pipeline/cli_preview.py."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import AugmentConfig
from .generators import iter_pairs


def main() -> int:
    ap = argparse.ArgumentParser("Preview self-supervised augmentation pairs")
    ap.add_argument("--root", required=True, help="Root containing Java files (e.g., ../data/gcj_compiled)")
    ap.add_argument("--out", required=True, help="Directory to write augmented .java files")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pos-ratio", type=float, default=0.5)

    ap.add_argument("--limit-buckets", type=int, default=None)
    ap.add_argument("--max-files-per-bucket", type=int, default=None)
    ap.add_argument("--glob", default="*.java")

    ap.add_argument("--no-rename", action="store_true")
    ap.add_argument("--rename-prob", type=float, default=1.0)
    ap.add_argument("--no-ws", action="store_true")
    ap.add_argument("--ws-prob", type=float, default=0.5)

    args = ap.parse_args()

    cfg = AugmentConfig(
        root=Path(args.root).expanduser().resolve(),
        out_dir=Path(args.out).expanduser().resolve(),
        seed=args.seed,
        glob=args.glob,
        limit_buckets=args.limit_buckets,
        max_files_per_bucket=args.max_files_per_bucket,
        rename_identifiers=not args.no_rename,
        rename_prob=args.rename_prob,
        whitespace_noise=not args.no_ws,
        whitespace_prob=args.ws_prob,
    )

    it = iter_pairs(cfg, infinite=True, pos_ratio=args.pos_ratio)
    for _ in range(args.n):
        a, b, y = next(it)
        print(y, a, "->", b)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
