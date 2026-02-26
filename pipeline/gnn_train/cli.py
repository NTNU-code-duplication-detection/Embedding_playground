#!/usr/bin/env python3
"""
Module for CLI testing
"""
from __future__ import annotations

import argparse
from pathlib import Path
import torch

from gnn_train.config import TrainConfig
from gnn_train.shard_index import build_index, save_index, load_index, ShardStore
from gnn_train.data import default_pair_generator
from gnn_train.train import train_loop, set_seed


def main() -> int:
    """
    Main entry point
    """
    ap = argparse.ArgumentParser("Train GNN on embedded method graphs")
    ap.add_argument("--shards", required=True, help="embed_cache/shards directory")
    ap.add_argument(
        "--index", 
        default="./gnn_index.json", 
        help="index json path (created if missing)"
        )
    ap.add_argument("--device", default="cpu", help="cpu|mps|cuda")

    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-pairs", type=int, default=64)

    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--in-dim", type=int, default=768)

    args = ap.parse_args()

    cfg = TrainConfig(
        shards_dir=Path(args.shards).expanduser().resolve(),
        index_path=Path(args.index).expanduser().resolve(),
        device=args.device,
        in_dim=args.in_dim,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.wd,
        steps=args.steps,
        batch_pairs=args.batch_pairs,
        seed=args.seed,
    )

    set_seed(cfg.seed)

    if not cfg.index_path.exists():
        idx = build_index(cfg.shards_dir)
        save_index(idx, cfg.index_path)
        print(f"Saved index with {len(idx)} methods to {cfg.index_path}")
    else:
        idx = load_index(cfg.index_path)
        print(f"Loaded index with {len(idx)} methods from {cfg.index_path}")

    store = ShardStore(cfg.shards_dir, max_cached=4)

    # Placeholder pair generator.
    # Replace with your generator that yields (method_id_a, method_id_b, label).
    method_ids = list(idx.keys())
    pair_iter = default_pair_generator(method_ids, seed=cfg.seed)

    enc, clf = train_loop(
        pair_iter=pair_iter,
        index=idx,
        store=store,
        device=cfg.device,
        in_dim=cfg.in_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        steps=cfg.steps,
        batch_pairs=cfg.batch_pairs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        dropout=cfg.dropout,
        log_every=cfg.log_every,
        num_edge_types=4,
    )

    out_dir = Path("./gnn_models").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    enc_path = out_dir / "method_encoder.pt"
    clf_path = out_dir / "pair_classifier.pt"

    torch.save(enc.state_dict(), enc_path)
    torch.save(clf.state_dict(), clf_path)

    print(f"Saved encoder to {enc_path}")
    print(f"Saved classifier to {clf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
