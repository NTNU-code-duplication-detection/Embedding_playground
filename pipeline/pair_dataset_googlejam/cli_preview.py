"""Module pipeline/pair_dataset_googlejam/cli_preview.py."""

# pipeline/pair_dataset_googlejam/cli_preview.py
from __future__ import annotations

import argparse
from pathlib import Path

from pair_dataset_googlejam.generators import GoogleJamConfig, positive_pairs, negative_pairs, interleave


def main() -> int:
    """
    Main entry point
    """
    ap = argparse.ArgumentParser("Preview GoogleJam pairs")
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-buckets", type=int, default=None)
    ap.add_argument("--max-files-per-bucket", type=int, default=None)
    ap.add_argument("--pos-ratio", type=float, default=0.5)
    args = ap.parse_args()

    cfg = GoogleJamConfig(
        root=Path(args.root),
        seed=args.seed,
        limit_buckets=args.limit_buckets,
        max_files_per_bucket=args.max_files_per_bucket,
    )

    pos = positive_pairs(cfg, infinite=True)
    neg = negative_pairs(cfg, infinite=True)
    it = interleave(pos, neg, pos_ratio=args.pos_ratio, seed=args.seed)

    print("Samples:")
    for i in range(args.n):
        a, b, y = next(it)
        print(y, a, "->", b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
