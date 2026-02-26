"""
04_evaluate.py — Evaluate a trained chunk-GNN checkpoint.

Standalone evaluation for re-running metrics on a saved model without
retraining. Useful for evaluating best_model.pt on test set.

Usage:
    python scripts/04_evaluate.py \
        --config configs/bcb_mvp.json \
        --bcb_root ~/Multigraph_match_optimized/data/data_source/dataset_bigclonebench \
        --cache_dir ~/chunk_gnn_cache \
        --checkpoint ~/chunk_gnn_out/run_XXXXXX/checkpoints/best_model.pt \
        --split test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from chunk_gnn.data.bcb_loader import BCBLoader
from chunk_gnn.model.siamese import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate chunk-GNN checkpoint")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--bcb_root", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Load data
    loader = BCBLoader(bcb_root=args.bcb_root)
    pairs = loader.get_split(args.split)
    log.info("Loaded %d %s pairs", len(pairs), args.split)

    # Create dataloader for the requested split only
    from chunk_gnn.data.pair_dataset import BCBPairDataset, collate_pairs
    from torch.utils.data import DataLoader

    dataset = BCBPairDataset(
        pairs, args.cache_dir,
        label_positive=config.get("training", {}).get("label_positive", 1.0),
        label_negative=config.get("training", {}).get("label_negative", -1.0),
    )
    dataloader = DataLoader(
        dataset, batch_size=config.get("training", {}).get("batch_size", 32),
        shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_pairs, pin_memory=True,
    )

    # Build model and load checkpoint
    model = build_model(config, device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    log.info(
        "Loaded checkpoint from epoch %d (best F1=%.4f)",
        checkpoint.get("epoch", -1), checkpoint.get("best_f1", 0),
    )

    # Evaluate
    evaluator = Evaluator(
        threshold_steps=config.get("evaluation", {}).get("threshold_steps", 200),
    )
    results = evaluator.evaluate(model, dataloader, device, args.split)

    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS ({args.split} set)")
    print(f"{'='*60}")
    print(f"Samples:      {results.num_samples:,}")
    print()
    print("At threshold=0 (MagNET-comparable):")
    print(f"  Precision:  {results.precision_t0:.4f}")
    print(f"  Recall:     {results.recall_t0:.4f}")
    print(f"  F1:         {results.f1_t0:.4f}")
    print()
    print(f"At optimal threshold ({results.optimal_threshold:.3f}):")
    print(f"  Precision:  {results.precision_opt:.4f}")
    print(f"  Recall:     {results.recall_opt:.4f}")
    print(f"  F1:         {results.f1_opt:.4f}")
    print()
    print("Per-type recall:")
    for ctype, recall in sorted(results.per_type_recall.items()):
        print(f"  {ctype:12s}: {recall:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
