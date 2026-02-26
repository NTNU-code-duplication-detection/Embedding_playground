"""
01_explore_bcb.py — Explore the BigCloneBench dataset.

Run this first to understand the dataset size, split distribution,
clone type breakdown, and check for missing files.

Usage:
    python scripts/01_explore_bcb.py --bcb_root /path/to/dataset_bigclonebench

On IDUN:
    python scripts/01_explore_bcb.py \
        --bcb_root ~/Multigraph_match_optimized/data/data_source/dataset_bigclonebench
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import os

# Add parent directory to path so we can import chunk_gnn modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from chunk_gnn.data.bcb_loader import BCBLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Explore BigCloneBench dataset")
    parser.add_argument(
        "--bcb_root",
        type=str,
        required=True,
        help="Path to BCB data dir (contains clone_labels.txt and dataset_files/)",
    )
    parser.add_argument(
        "--sample_files",
        type=int,
        default=5,
        help="Number of sample .java files to print (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: save stats to JSON file",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("BigCloneBench Dataset Exploration")
    log.info("=" * 60)
    log.info("BCB root: %s", args.bcb_root)

    # Load labels
    loader = BCBLoader(bcb_root=args.bcb_root)
    pairs = loader.load_labels()
    stats = loader.get_stats()

    # Print overview
    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Total BCB pairs:      {stats.total_pairs:,}")
    print(f"  Clone pairs:        {stats.clone_pairs:,}")
    print(f"  Non-clone pairs:    {stats.non_clone_pairs:,}")
    print(f"  Clone ratio:        {stats.clone_pairs / stats.total_pairs:.1%}")
    print(f"Unique function IDs:  {stats.unique_function_ids:,}")

    # Split distribution
    print("\n" + "-" * 40)
    print("SPLIT DISTRIBUTION")
    print("-" * 40)
    for split in ["train", "test", "val"]:
        count = stats.pairs_by_split.get(split, 0)
        pct = count / stats.total_pairs * 100 if stats.total_pairs > 0 else 0
        print(f"  {split:6s}: {count:>10,} pairs ({pct:.1f}%)")

    # Clone type distribution
    print("\n" + "-" * 40)
    print("CLONE TYPE DISTRIBUTION")
    print("-" * 40)
    for ctype, count in sorted(
        stats.pairs_by_type.items(), key=lambda x: -x[1]
    ):
        pct = count / stats.total_pairs * 100
        print(f"  {ctype:12s}: {count:>10,} ({pct:.2f}%)")

    # Clone type distribution (clones only, excluding Non_Clone)
    print("\n" + "-" * 40)
    print("CLONE TYPE DISTRIBUTION (clones only)")
    print("-" * 40)
    clone_only = {
        k: v for k, v in stats.pairs_by_type.items() if k != "Non_Clone"
    }
    total_clones = sum(clone_only.values())
    for ctype, count in sorted(clone_only.items(), key=lambda x: -x[1]):
        pct = count / total_clones * 100 if total_clones > 0 else 0
        print(f"  {ctype:12s}: {count:>10,} ({pct:.2f}%)")

    # File availability check
    print("\n" + "-" * 40)
    print("FILE AVAILABILITY")
    print("-" * 40)
    available, missing = loader.check_file_availability()
    print(f"  Available files:  {len(available):,}")
    print(f"  Missing files:    {len(missing):,}")
    if missing:
        sample_missing = list(missing)[:10]
        print(f"  Sample missing:   {sample_missing}")

    # Count pairs that would be excluded due to missing files
    excluded_pairs = 0
    for pair in pairs:
        if pair.id1 in missing or pair.id2 in missing:
            excluded_pairs += 1
    print(f"  Pairs with missing files: {excluded_pairs:,}")
    usable_pairs = stats.total_pairs - excluded_pairs
    print(f"  Usable pairs:     {usable_pairs:,}")

    # Sample some .java files
    print("\n" + "-" * 40)
    print(f"SAMPLE .java FILES ({args.sample_files} random)")
    print("-" * 40)
    available_list = list(available)
    random.seed(42)
    sample_ids = random.sample(
        available_list, min(args.sample_files, len(available_list))
    )
    for func_id in sample_ids:
        source = loader.load_function_source(func_id)
        if source:
            lines = source.strip().split("\n")
            print(f"\n  --- {func_id}.java ({len(lines)} lines) ---")
            # Print first 15 lines max
            for line in lines[:15]:
                print(f"  {line}")
            if len(lines) > 15:
                print(f"  ... ({len(lines) - 15} more lines)")

    # Storage estimate
    print("\n" + "-" * 40)
    print("STORAGE ESTIMATE (for pre-computed graphs)")
    print("-" * 40)
    n_funcs = len(available)
    avg_chunks = 10  # rough estimate
    bytes_per_func = avg_chunks * 768 * 2  # float16
    total_bytes = n_funcs * bytes_per_func
    print(f"  Unique functions:        {n_funcs:,}")
    print(f"  Estimated avg chunks:    {avg_chunks}")
    print(f"  Bytes per function:      ~{bytes_per_func:,} (float16)")
    print(f"  Total estimated storage: ~{total_bytes / 1e9:.1f} GB")

    # Save stats to JSON if requested
    if args.output:
        output_data = {
            "total_pairs": stats.total_pairs,
            "clone_pairs": stats.clone_pairs,
            "non_clone_pairs": stats.non_clone_pairs,
            "unique_function_ids": stats.unique_function_ids,
            "pairs_by_split": stats.pairs_by_split,
            "pairs_by_type": stats.pairs_by_type,
            "available_files": len(available),
            "missing_files": len(missing),
            "excluded_pairs": excluded_pairs,
            "usable_pairs": usable_pairs,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nStats saved to {args.output}")

    print("\n" + "=" * 60)
    print("Exploration complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
