"""
Cli preview module
"""
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pair_dataset.config import PairDatasetConfig
from pair_dataset.scan import scan_dataset, summarize_index
from pair_dataset.generators import positive_pairs, negative_pairs


def main() -> int:
    """
    Main entry point
    """
    ap = argparse.ArgumentParser("Preview clone/non-clone pair generators")
    ap.add_argument("--root", required=True, help=".../data/code-clone-dataset/dataset")
    ap.add_argument("--clone-type", default="type-3", choices=["type-1", "type-2", "type-3"])
    ap.add_argument("--neg-pool", default="same_clone_type", choices=["same_clone_type", "base"])
    ap.add_argument("--limit-indices", type=int, default=None)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = PairDatasetConfig(
        root=Path(args.root).expanduser().resolve(),
        clone_type=args.clone_type,
        negative_pool=args.neg_pool,
        seed=args.seed,
        limit_indices=args.limit_indices,
    )

    entries = scan_dataset(cfg)
    a, b, c = summarize_index(entries)
    print(f"indices={a} indices_with_clones={b} total_clones={c} clone_type={cfg.clone_type} neg_pool={cfg.negative_pool}")

    pos = positive_pairs(cfg, infinite=True)
    neg = negative_pairs(cfg, infinite=True)

    print("\nPOSITIVE samples:")
    for _ in range(args.n):
        x, y, lab = next(pos)
        print(lab, x, "->", y)

    print("\nNEGATIVE samples:")
    for _ in range(args.n):
        x, y, lab = next(neg)
        print(lab, x, "->", y)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
