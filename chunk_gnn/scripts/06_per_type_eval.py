"""
06_per_type_eval.py — Per-clone-type evaluation of trained chunk-GNN.

Loads the granular clone_labels_typed.txt (with T1, T2, VST3, ST3, MT3,
WT3_T4 labels) and evaluates a saved checkpoint to produce per-type recall
comparable to MagNET's Table III.

Usage (from dataset_loader/):
    python chunk_gnn/scripts/06_per_type_eval.py

Hardcoded for local evaluation of Run 7 (best model). Edit paths below
if running on IDUN or with a different checkpoint.
"""

from __future__ import annotations

import json
import logging
import os
import sys

import numpy as np
import torch
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from chunk_gnn.data.bcb_loader import BCBLoader
from chunk_gnn.data.pair_dataset import BCBPairDataset, collate_pairs
from chunk_gnn.model.siamese import build_model

from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---- Configuration (edit these for your setup) ----
BCB_ROOT = os.path.expanduser("~/datasets/bigclonebench")
CACHE_DIR = os.path.expanduser("~/datasets/chunk_gnn_cache")
CHECKPOINT = os.path.expanduser(
    "~/datasets/chunk_gnn_out/run_20260222_153049/checkpoints/best_model.pt"
)
CONFIG_FILE = os.path.join(
    os.path.dirname(__file__), "..", "configs", "bcb_classifier.json"
)
LABELS_FILE = "clone_labels_typed.txt"  # Granular types
# ---------------------------------------------------

# MagNET paper results (Table III, BigCloneBench)
MAGNET_RECALL = {
    "T1": 1.000,
    "T2": 1.000,
    "VST3": None,  # Not reported separately in MagNET Table III
    "ST3": 1.000,
    "MT3": 1.000,
    "WT3_T4": 0.970,  # Reported as "T4" in paper
}
MAGNET_OVERALL = {"P": 0.960, "R": 0.970, "F1": 0.965}


def main() -> None:
    device = torch.device("cpu")  # Local eval on MacBook

    # Load config
    with open(CONFIG_FILE, encoding="utf-8") as f:
        config = json.load(f)

    # Load data with granular type labels
    loader = BCBLoader(bcb_root=BCB_ROOT, labels_file=LABELS_FILE)
    test_pairs = loader.get_split("test")
    log.info("Loaded %d test pairs with granular types", len(test_pairs))

    # Log type distribution in test set
    type_counts: dict[str, int] = {}
    for p in test_pairs:
        type_counts[p.clone_type] = type_counts.get(p.clone_type, 0) + 1
    log.info("Test set type distribution:")
    for ctype in sorted(type_counts.keys()):
        log.info("  %s: %d", ctype, type_counts[ctype])

    # Create dataloader
    dataset = BCBPairDataset(
        test_pairs, CACHE_DIR,
        label_positive=config.get("training", {}).get("label_positive", 1.0),
        label_negative=config.get("training", {}).get("label_negative", -1.0),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.get("training", {}).get("batch_size", 32),
        shuffle=False,
        num_workers=0,  # Single-threaded for stability on Mac
        collate_fn=collate_pairs,
    )

    # Build model and load checkpoint
    model = build_model(config, device)
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    log.info(
        "Loaded checkpoint from epoch %d (best val F1=%.4f)",
        checkpoint.get("epoch", -1),
        checkpoint.get("best_f1", 0),
    )

    # Evaluate
    model.eval()
    all_logits: list[float] = []
    all_sims: list[float] = []
    all_labels: list[int] = []
    all_types: list[str] = []

    with torch.no_grad():
        for batch_idx, (batch1, batch2, labels, clone_types) in enumerate(dataloader):
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)

            # Cosine similarity (for comparison)
            sims = model.similarity(batch1, batch2)
            all_sims.extend(sims.cpu().tolist())

            # Classifier logits
            if model.use_classifier:
                logits = model.classify(batch1, batch2)
                all_logits.extend(logits.cpu().tolist())

            all_labels.extend([1 if l > 0 else 0 for l in labels.tolist()])
            all_types.extend(clone_types)

            if (batch_idx + 1) % 500 == 0:
                log.info("  Processed %d/%d batches", batch_idx + 1, len(dataloader))

    labels_arr = np.array(all_labels)
    types_arr = np.array(all_types)

    # Classifier metrics
    logits_arr = np.array(all_logits)
    probs_arr = 1.0 / (1.0 + np.exp(-logits_arr))
    preds_cls = (probs_arr > 0.5).astype(int)

    overall_p = precision_score(labels_arr, preds_cls, zero_division=0)
    overall_r = recall_score(labels_arr, preds_cls, zero_division=0)
    overall_f1 = f1_score(labels_arr, preds_cls, zero_division=0)
    overall_auroc = roc_auc_score(labels_arr, probs_arr)

    # Per-type recall (classifier)
    per_type_recall: dict[str, float] = {}
    per_type_counts: dict[str, int] = {}
    per_type_precision: dict[str, float] = {}

    for ctype in sorted(set(all_types)):
        if ctype == "Non_Clone":
            continue
        mask = types_arr == ctype
        type_labels = labels_arr[mask]
        type_preds = preds_cls[mask]

        # Only count actual clone pairs for recall
        clone_mask = type_labels == 1
        if clone_mask.sum() == 0:
            continue

        tp = ((type_preds == 1) & (type_labels == 1)).sum()
        fn = ((type_preds == 0) & (type_labels == 1)).sum()
        fp = ((type_preds == 1) & (type_labels == 0)).sum()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        per_type_recall[ctype] = float(recall)
        per_type_precision[ctype] = float(precision)
        per_type_counts[ctype] = int(clone_mask.sum())

    # Print results
    print()
    print("=" * 72)
    print("  PER-TYPE EVALUATION — Chunk-GNN (Run 7) vs MagNET")
    print("=" * 72)
    print()
    print(f"  Test samples: {len(all_labels):,}")
    print(f"  Clone pairs:  {labels_arr.sum():,.0f}")
    print(f"  Non-clone:    {(labels_arr == 0).sum():,.0f}")
    print()

    # Overall metrics comparison
    print("  OVERALL METRICS")
    print("  " + "-" * 50)
    print(f"  {'Metric':<12} {'Chunk-GNN':>12} {'MagNET (paper)':>16} {'Delta':>10}")
    delta_p = overall_p - MAGNET_OVERALL['P']
    delta_r = overall_r - MAGNET_OVERALL['R']
    delta_f1 = overall_f1 - MAGNET_OVERALL['F1']
    print(f"  {'Precision':<12} {overall_p:>12.4f} {MAGNET_OVERALL['P']:>16.3f} {delta_p:>+10.4f}")
    print(f"  {'Recall':<12} {overall_r:>12.4f} {MAGNET_OVERALL['R']:>16.3f} {delta_r:>+10.4f}")
    print(f"  {'F1':<12} {overall_f1:>12.4f} {MAGNET_OVERALL['F1']:>16.3f} {delta_f1:>+10.4f}")
    print(f"  {'AUROC':<12} {overall_auroc:>12.4f} {'—':>16}")
    print()

    # Per-type recall comparison
    print("  PER-TYPE RECALL (clone pairs only)")
    print("  " + "-" * 68)
    print(f"  {'Type':<10} {'Count':>8} {'Chunk-GNN':>12} {'MagNET':>10} {'Delta':>10} {'Note'}")
    print(f"  {'----':<10} {'-----':>8} {'---------':>12} {'------':>10} {'-----':>10}")

    for ctype in ["T1", "T2", "VST3", "ST3", "MT3", "WT3_T4"]:
        if ctype not in per_type_recall:
            continue
        our_recall = per_type_recall[ctype]
        count = per_type_counts[ctype]
        magnet_recall = MAGNET_RECALL.get(ctype)

        if magnet_recall is not None:
            delta = f"{our_recall - magnet_recall:+.4f}"
            magnet_str = f"{magnet_recall:.3f}"
        else:
            delta = "—"
            magnet_str = "—"

        # Note for small sample sizes
        note = ""
        if count < 100:
            note = f"(small n={count})"

        print(f"  {ctype:<10} {count:>8,} {our_recall:>12.4f} {magnet_str:>10} {delta:>10} {note}")

    print()
    print("  " + "-" * 68)
    overall_delta = overall_r - MAGNET_OVERALL['R']
    print(
        f"  {'OVERALL':<10} {int(labels_arr.sum()):>8,} "
        f"{overall_r:>12.4f} {MAGNET_OVERALL['R']:>10.3f} {overall_delta:>+10.4f}"
    )
    print()

    # Non-clone precision (specificity)
    non_clone_mask = types_arr == "Non_Clone"
    if non_clone_mask.sum() > 0:
        nc_labels = labels_arr[non_clone_mask]
        nc_preds = preds_cls[non_clone_mask]
        tn = ((nc_preds == 0) & (nc_labels == 0)).sum()
        fp = ((nc_preds == 1) & (nc_labels == 0)).sum()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        print(f"  Non-clone specificity (TN rate): {specificity:.4f} ({tn}/{tn+fp})")
        print()

    print("=" * 72)

    # Save results to JSON
    output_json = os.path.join(
        os.path.dirname(CHECKPOINT), "..", "per_type_results.json"
    )
    results = {
        "overall": {
            "precision": float(overall_p),
            "recall": float(overall_r),
            "f1": float(overall_f1),
            "auroc": float(overall_auroc),
        },
        "per_type_recall": per_type_recall,
        "per_type_precision": per_type_precision,
        "per_type_counts": per_type_counts,
        "magnet_comparison": MAGNET_RECALL,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_json}")


if __name__ == "__main__":
    main()
