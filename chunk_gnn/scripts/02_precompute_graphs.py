"""
02_precompute_graphs.py — Pre-compute chunk graphs for all BCB functions.

For each unique function in BigCloneBench:
  1. Load .java source
  2. Chunk with tree-sitter
  3. Embed chunks with UniXcoder
  4. Build graph (Data object with edges)
  5. Save to cache_dir/{func_id}.pt

Supports resuming: skips function IDs that already have a .pt file in cache.

Usage:
    python scripts/02_precompute_graphs.py \
        --bcb_root ~/Multigraph_match_optimized/data/data_source/dataset_bigclonebench \
        --cache_dir ~/chunk_gnn_cache \
        --device cuda

On IDUN via SLURM:
    sbatch slurm/precompute.slurm
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from chunk_gnn.data.bcb_loader import BCBLoader
from chunk_gnn.data.chunker import TreeSitterChunker
from chunk_gnn.data.embedder import ChunkEmbedder
from chunk_gnn.data.graph_builder import ChunkGraphBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute chunk graphs for all BCB functions"
    )
    parser.add_argument(
        "--bcb_root", type=str, required=True,
        help="Path to BCB data dir (contains clone_labels.txt + dataset_files/)",
    )
    parser.add_argument(
        "--cache_dir", type=str, required=True,
        help="Output directory for pre-computed .pt graph files",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for UniXcoder inference (default: cuda if available)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for UniXcoder embedding (default: 32)",
    )
    parser.add_argument(
        "--max_chunks", type=int, default=50,
        help="Max chunks per function (default: 50)",
    )
    parser.add_argument(
        "--log_interval", type=int, default=500,
        help="Print progress every N functions (default: 500)",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Load BCB metadata
    # -----------------------------------------------------------------------
    log.info("Step 1: Loading BCB labels...")
    loader = BCBLoader(bcb_root=args.bcb_root)
    func_ids = loader.get_unique_function_ids()
    available, missing = loader.check_file_availability()

    log.info(
        "  Unique functions: %d, Available: %d, Missing: %d",
        len(func_ids), len(available), len(missing),
    )

    # Check which functions are already cached (for resume support)
    # Both .pt (real graphs) and .empty (unparseable markers) count as cached
    already_cached = set()
    for pt_file in cache_dir.glob("*.pt"):
        already_cached.add(pt_file.stem)
    for empty_file in cache_dir.glob("*.empty"):
        already_cached.add(empty_file.stem)

    to_process = available - already_cached
    log.info(
        "  Already cached: %d, To process: %d",
        len(already_cached), len(to_process),
    )

    if not to_process:
        log.info("All functions already cached. Nothing to do.")
        _save_metadata(cache_dir, available, missing, already_cached, {})
        return

    # -----------------------------------------------------------------------
    # Step 2: Initialize components
    # -----------------------------------------------------------------------
    log.info("Step 2: Initializing chunker, embedder, and graph builder...")
    chunker = TreeSitterChunker()
    embedder = ChunkEmbedder(
        device=args.device,
        batch_size=args.batch_size,
    )
    graph_builder = ChunkGraphBuilder(add_self_loops=True)

    # -----------------------------------------------------------------------
    # Step 3: Process each function
    # -----------------------------------------------------------------------
    log.info("Step 3: Processing %d functions...", len(to_process))
    start_time = time.time()

    stats = {
        "processed": 0,
        "success": 0,
        "empty_chunks": 0,
        "read_failed": 0,
        "chunk_counts": [],
        "errors": [],
    }

    func_list = sorted(to_process)  # Deterministic order

    for i, func_id in enumerate(func_list):
        try:
            # Load source
            source = loader.load_function_source(func_id)
            if source is None:
                stats["read_failed"] += 1
                continue

            # Chunk
            chunks = chunker.chunk_function(source, max_chunks=args.max_chunks)
            if chunks is None:
                stats["empty_chunks"] += 1
                # Save a marker file so we don't retry on resume
                _save_empty_marker(cache_dir, func_id)
                continue

            # Embed
            embeddings = embedder.embed_chunks(chunks)

            # Build graph
            graph_data = graph_builder.build_graph(chunks, embeddings)

            # Save to disk
            save_path = cache_dir / f"{func_id}.pt"
            torch.save(graph_data, save_path)

            stats["success"] += 1
            stats["chunk_counts"].append(len(chunks))

        except (ValueError, RuntimeError, OSError) as e:
            log.error("Error processing %s: %s", func_id, e)
            stats["errors"].append({"func_id": func_id, "error": str(e)})

        stats["processed"] += 1

        # Progress logging
        if (i + 1) % args.log_interval == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(func_list) - i - 1) / rate if rate > 0 else 0
            log.info(
                "  Progress: %d/%d (%.1f%%) | "
                "Success: %d | Empty: %d | Errors: %d | "
                "Rate: %.1f func/s | ETA: %.0f min",
                i + 1,
                len(func_list),
                (i + 1) / len(func_list) * 100,
                stats["success"],
                stats["empty_chunks"],
                len(stats["errors"]),
                rate,
                eta / 60,
            )

    elapsed = time.time() - start_time

    # -----------------------------------------------------------------------
    # Step 4: Summary
    # -----------------------------------------------------------------------
    log.info("")
    log.info("=" * 60)
    log.info("PRECOMPUTE COMPLETE")
    log.info("=" * 60)
    log.info("  Total processed:    %d", stats["processed"])
    log.info("  Successful graphs:  %d", stats["success"])
    log.info("  Empty (no chunks):  %d", stats["empty_chunks"])
    log.info("  Read failures:      %d", stats["read_failed"])
    log.info("  Errors:             %d", len(stats["errors"]))
    log.info("  Time:               %.1f min", elapsed / 60)
    if stats["chunk_counts"]:
        avg_chunks = sum(stats["chunk_counts"]) / len(stats["chunk_counts"])
        log.info(
            "  Chunk count: min=%d, max=%d, avg=%.1f",
            min(stats["chunk_counts"]),
            max(stats["chunk_counts"]),
            avg_chunks,
        )

    # Check disk usage
    total_bytes = sum(
        f.stat().st_size for f in cache_dir.glob("*.pt")
    )
    log.info("  Cache disk usage:   %.1f MB", total_bytes / 1e6)

    if stats["errors"]:
        log.warning("  Errors encountered:")
        for err in stats["errors"][:10]:
            log.warning("    %s: %s", err["func_id"], err["error"])
        if len(stats["errors"]) > 10:
            log.warning("    ... and %d more", len(stats["errors"]) - 10)

    # Save metadata
    _save_metadata(
        cache_dir, available, missing,
        already_cached | {fid for fid in func_list if (cache_dir / f"{fid}.pt").exists()},
        stats,
    )


def _save_empty_marker(cache_dir: Path, func_id: str) -> None:
    """Save a small marker for functions that produced no chunks.

    Uses .empty extension (not .pt) so these are never loaded as real
    graphs by BCBPairDataset, which only globs *.pt files.
    """
    marker_path = cache_dir / f"{func_id}.empty"
    marker_path.touch()


def _save_metadata(
    cache_dir: Path,
    available: set,
    missing: set,
    cached: set,
    stats: dict,
) -> None:
    """Save metadata about the precompute run."""
    metadata = {
        "available_functions": len(available),
        "missing_functions": len(missing),
        "cached_graphs": len(cached),
        "stats": {
            k: v for k, v in stats.items()
            if k != "chunk_counts"  # Too large for JSON
        },
    }
    if stats.get("chunk_counts"):
        metadata["chunk_count_stats"] = {
            "min": min(stats["chunk_counts"]),
            "max": max(stats["chunk_counts"]),
            "avg": sum(stats["chunk_counts"]) / len(stats["chunk_counts"]),
            "total": len(stats["chunk_counts"]),
        }
    with open(cache_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    log.info("  Metadata saved to %s/metadata.json", cache_dir)


if __name__ == "__main__":
    main()
