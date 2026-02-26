"""
03_train.py — Train the siamese chunk-level GNN on BigCloneBench.

Requires pre-computed graphs (from 02_precompute_graphs.py).

Usage:
    python scripts/03_train.py \
        --config configs/bcb_mvp.json \
        --bcb_root ~/Multigraph_match_optimized/data/data_source/dataset_bigclonebench \
        --cache_dir ~/chunk_gnn_cache \
        --output_dir ~/chunk_gnn_out

On IDUN via SLURM:
    sbatch slurm/train.slurm
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from chunk_gnn.data.bcb_loader import BCBLoader
from chunk_gnn.data.pair_dataset import create_dataloaders
from chunk_gnn.model.siamese import build_model
from chunk_gnn.train.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    log.info("Random seed set to %d", seed)


def main():
    parser = argparse.ArgumentParser(description="Train chunk-level GNN on BCB")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to config JSON file",
    )
    parser.add_argument(
        "--bcb_root", type=str, required=True,
        help="Path to BCB data dir",
    )
    parser.add_argument(
        "--cache_dir", type=str, required=True,
        help="Directory with pre-computed .pt graph files",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (default: cuda if available)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader workers (default: 4)",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    train_cfg = config.get("training", {})

    # Setup
    set_seed(train_cfg.get("seed", 42))

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))
        log.info("GPU Memory: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    # Create output directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save config to output dir for reproducibility
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    log.info("Loading BCB labels...")
    loader = BCBLoader(bcb_root=args.bcb_root)

    train_pairs = loader.get_split("train")
    val_pairs = loader.get_split("val")
    test_pairs = loader.get_split("test")

    log.info(
        "Pairs: train=%d, val=%d, test=%d",
        len(train_pairs), len(val_pairs), len(test_pairs),
    )

    # -----------------------------------------------------------------------
    # Create DataLoaders
    # -----------------------------------------------------------------------
    log.info("Creating DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        cache_dir=args.cache_dir,
        batch_size=train_cfg.get("batch_size", 32),
        num_workers=args.num_workers,
        label_positive=train_cfg.get("label_positive", 1.0),
        label_negative=train_cfg.get("label_negative", -1.0),
    )

    # -----------------------------------------------------------------------
    # Build model
    # -----------------------------------------------------------------------
    log.info("Building model...")
    model = build_model(config, device)

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        output_dir=output_dir,
    )

    results = trainer.train()

    # Print final summary
    log.info("")
    log.info("=" * 60)
    log.info("FINAL RESULTS")
    log.info("=" * 60)
    log.info("Best epoch: %d", results["best_epoch"])
    log.info("Best val F1: %.4f", results["best_val_f1"])

    # Print the best test results
    for epoch_data in results["per_epoch"]:
        if epoch_data["epoch"] == results["best_epoch"] and "test" in epoch_data:
            test = epoch_data["test"]
            log.info(
                "Test at best epoch: P=%.4f R=%.4f F1=%.4f (t=%.3f)",
                test["precision_opt"], test["recall_opt"],
                test["f1_opt"], test["threshold"],
            )
            log.info(
                "Test at t=0 (MagNET-comparable): F1=%.4f",
                test["f1_t0"],
            )
            if test.get("classifier_f1"):
                log.info(
                    "Classifier head: P=%.4f R=%.4f F1=%.4f AUROC=%.4f",
                    test["classifier_precision"], test["classifier_recall"],
                    test["classifier_f1"], test["classifier_auroc"],
                )
            if test.get("per_type_recall"):
                log.info("Per-type recall:")
                for ctype, recall in sorted(test["per_type_recall"].items()):
                    log.info("  %s: %.4f", ctype, recall)


if __name__ == "__main__":
    main()
