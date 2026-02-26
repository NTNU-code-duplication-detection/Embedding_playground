#!/usr/bin/env python3
"""
Module for CLI testing of cache embeddig module
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from embed_cache.config import EmbedCacheConfig
from embed_cache.util import ensure_dir, read_jsonl, sha1_text, write_json
from embed_cache.cache import DiskEmbeddingCache
from embed_cache.embedder import Embedder
from embed_cache.build_tensors import edges_to_tensors

def process(cfg: EmbedCacheConfig) -> None:
    """
    Process a embeddin config
    """
    ensure_dir(cfg.out_dir)
    ensure_dir(cfg.cache_dir)

    cache = DiskEmbeddingCache(cfg.cache_dir)
    emb = Embedder(model_name=cfg.model_name, device=cfg.device, max_length=cfg.max_length)

    shard_dir = cfg.out_dir / "shards"
    ensure_dir(shard_dir)

    shard: List[Dict] = []
    shard_idx = 0
    seen_methods = 0
    cache_hits = 0
    cache_misses = 0

    # A small batching buffer for uncached node texts
    pending_keys: List[str] = []
    pending_texts: List[str] = []
    pending_targets: List[tuple] = []  # (method_tmp, node_pos, key)

    def flush_pending():
        nonlocal cache_misses
        if not pending_texts:
            return

        vecs = emb.embed_texts(pending_texts)  # [B, D] on CPU
        for i, key in enumerate(pending_keys):
            cache.put(key, vecs[i])
        cache_misses += len(pending_texts)

        pending_keys.clear()
        pending_texts.clear()
        pending_targets.clear()

    # We build each method record into a temporary dict, then finalize with tensors
    for rec in tqdm(read_jsonl(cfg.in_jsonl), desc="Embedding methods"):
        nodes = rec.get("nodes", [])
        edges = rec.get("edges", [])

        if not nodes:
            continue

        # Prepare per-node embeddings (fill from cache or pending)
        node_vecs: List[torch.Tensor] = [None] * len(nodes)  # type: ignore

        for i, n in enumerate(nodes):
            code = n.get("code", "")
            key = sha1_text(code)

            v = cache.get(key)
            if v is not None:
                node_vecs[i] = v
                cache_hits += 1
                continue

            # Not cached: batch it
            pending_keys.append(key)
            pending_texts.append(code)
            # store pointer so we can fill node_vecs after flush via cache.get
            pending_targets.append((rec["method_id"], i, key))

            if len(pending_texts) >= cfg.batch_size:
                flush_pending()
                # load back into node_vecs from cache (cheap, local disk)
                for _, idx, k in pending_targets:
                    node_vecs[idx] = cache.get(k)  # type: ignore
                pending_targets.clear()

        # Flush remaining pending for this method if any
        if pending_texts:
            flush_pending()
            for _, idx, k in pending_targets:
                node_vecs[idx] = cache.get(k)  # type: ignore
            pending_targets.clear()

        # Stack node embeddings
        # Safety: if any None slipped through, drop method
        if any(v is None for v in node_vecs):
            continue

        x = torch.stack(node_vecs, dim=0)  # [N, D]

        edge_index, edge_type = edges_to_tensors(edges)

        item = {
            "method_id": rec["method_id"],
            "method_name": rec.get("method_name"),
            "file": rec.get("file"),
            "x": x,  # [N, D]
            "edge_index": edge_index,  # [2, E]
            "edge_type": edge_type,    # [E]
            "num_nodes": int(x.shape[0]),
            "num_edges": int(edge_index.shape[1]),
        }

        shard.append(item)
        seen_methods += 1

        if len(shard) >= cfg.shard_size:
            out_path = shard_dir / f"methods_{shard_idx:05d}.pt"
            torch.save(shard, out_path)
            shard_idx += 1
            shard.clear()

    # Final shard
    if shard:
        out_path = shard_dir / f"methods_{shard_idx:05d}.pt"
        torch.save(shard, out_path)

    meta = {
        "in_jsonl": str(cfg.in_jsonl),
        "model_name": cfg.model_name,
        "device": cfg.device,
        "max_length": cfg.max_length,
        "batch_size": cfg.batch_size,
        "shard_size": cfg.shard_size,
        "num_methods": seen_methods,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "edge_type_to_id": {
            "SEQ": 0, "AST": 1, "IF_THEN": 2, "IF_ELSE": 3
        },
        "shards_dir": str((cfg.out_dir / "shards").resolve()),
    }
    write_json(cfg.out_dir / "embed_manifest.json", meta)

def main() -> int:
    """
    Main entrypoint
    """
    ap = argparse.ArgumentParser("Embed + cache ast_dataset graphs")
    ap.add_argument("--in-jsonl", required=True, help="Path to ast_dataset/methods.jsonl")
    ap.add_argument("--out", required=True, help="Output directory for embed_cache stage")
    ap.add_argument("--cache", required=True, help="Cache directory for node embeddings")
    ap.add_argument("--model", required=True, help="HF model name or local path")
    ap.add_argument("--device", default="cpu", help="cpu|mps|cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--shard-size", type=int, default=2000)

    args = ap.parse_args()

    cfg = EmbedCacheConfig(
        in_jsonl=Path(args.in_jsonl).expanduser().resolve(),
        out_dir=Path(args.out).expanduser().resolve(),
        cache_dir=Path(args.cache).expanduser().resolve(),
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shard_size=args.shard_size,
    )

    process(cfg)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
