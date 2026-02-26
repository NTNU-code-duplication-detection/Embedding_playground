"""
Evaluation for the siamese chunk-level GNN.

Computes precision, recall, F1 with threshold sweeping, and per-clone-type
recall to match MagNET's evaluation methodology.

Reports metrics for two comparison approaches:
  1. Cosine similarity (always computed for cross-run comparability):
     - F1 at threshold=0 (MagNET-compatible)
     - F1 at optimal threshold (best possible via sweep)
  2. Classifier head (when available):
     - F1 at sigmoid > 0.5 decision boundary
     - AUROC for overall discriminative power
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from chunk_gnn.model.siamese import SiameseChunkGNN

log = logging.getLogger(__name__)


@dataclass
class EvalResults:
    """Evaluation results container."""

    # At threshold=0 (MagNET-compatible cosine similarity)
    precision_t0: float = 0.0
    recall_t0: float = 0.0
    f1_t0: float = 0.0

    # At optimal cosine similarity threshold
    precision_opt: float = 0.0
    recall_opt: float = 0.0
    f1_opt: float = 0.0
    optimal_threshold: float = 0.0

    # Classifier head metrics (only populated when classifier is used)
    classifier_f1: float = 0.0
    classifier_precision: float = 0.0
    classifier_recall: float = 0.0
    classifier_auroc: float = 0.0

    # Per-clone-type recall (at optimal threshold or classifier boundary)
    per_type_recall: dict[str, float] = field(default_factory=dict)

    # Raw data for further analysis
    num_samples: int = 0


class Evaluator:
    """Evaluates the siamese GNN on clone detection."""

    def __init__(
        self,
        threshold_steps: int = 200,
        threshold_range: tuple[float, float] = (-1.0, 1.0),
    ):
        self.threshold_steps = threshold_steps
        self.threshold_range = threshold_range

    @torch.no_grad()
    def evaluate(
        self,
        model: SiameseChunkGNN,
        dataloader: DataLoader,
        device: torch.device,
        split_name: str = "test",
    ) -> EvalResults:
        """Run evaluation on a dataset split.

        Args:
            model: The siamese model (will be set to eval mode)
            dataloader: DataLoader for the split
            device: torch device
            split_name: Name for logging ("test", "val")

        Returns:
            EvalResults with all metrics
        """
        model.eval()

        all_sims: list[float] = []
        all_logits: list[float] = []
        all_labels: list[int] = []
        all_types: list[str] = []

        for batch1, batch2, labels, clone_types in dataloader:
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)

            # Always compute cosine similarity (for cross-run comparison)
            sims = model.similarity(batch1, batch2)
            all_sims.extend(sims.cpu().tolist())

            # Compute classifier logits if available
            if model.use_classifier:
                logits = model.classify(batch1, batch2)
                all_logits.extend(logits.cpu().tolist())

            # Convert +1/-1 labels to binary 0/1 for sklearn
            all_labels.extend([1 if l > 0 else 0 for l in labels.tolist()])
            all_types.extend(clone_types)

        results = EvalResults(num_samples=len(all_sims))

        if not all_sims:
            log.warning("No samples to evaluate")
            return results

        sims_arr = np.array(all_sims)
        labels_arr = np.array(all_labels)

        # Metrics at threshold=0 (MagNET-compatible: sign of cosine sim)
        preds_t0 = (sims_arr > 0).astype(int)
        results.precision_t0 = precision_score(labels_arr, preds_t0, zero_division=0)
        results.recall_t0 = recall_score(labels_arr, preds_t0, zero_division=0)
        results.f1_t0 = f1_score(labels_arr, preds_t0, zero_division=0)

        # Threshold sweep for optimal F1
        best_f1 = 0.0
        best_threshold = 0.0
        thresholds = np.linspace(
            self.threshold_range[0],
            self.threshold_range[1],
            self.threshold_steps,
        )

        for t in thresholds:
            preds = (sims_arr > t).astype(int)
            f1 = f1_score(labels_arr, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(t)

        # Final metrics at optimal threshold
        preds_opt = (sims_arr > best_threshold).astype(int)
        results.precision_opt = precision_score(labels_arr, preds_opt, zero_division=0)
        results.recall_opt = recall_score(labels_arr, preds_opt, zero_division=0)
        results.f1_opt = best_f1
        results.optimal_threshold = best_threshold

        log.info(
            "[%s] cosine t=0: P=%.4f R=%.4f F1=%.4f | "
            "optimal t=%.3f: P=%.4f R=%.4f F1=%.4f | n=%d",
            split_name,
            results.precision_t0, results.recall_t0, results.f1_t0,
            results.optimal_threshold,
            results.precision_opt, results.recall_opt, results.f1_opt,
            results.num_samples,
        )

        # Classifier head metrics (if available)
        if all_logits:
            logits_arr = np.array(all_logits)
            probs_arr = 1.0 / (1.0 + np.exp(-logits_arr))  # Sigmoid
            preds_cls = (probs_arr > 0.5).astype(int)

            results.classifier_precision = precision_score(
                labels_arr, preds_cls, zero_division=0
            )
            results.classifier_recall = recall_score(
                labels_arr, preds_cls, zero_division=0
            )
            results.classifier_f1 = f1_score(
                labels_arr, preds_cls, zero_division=0
            )

            # AUROC (handle edge case of single-class splits)
            if len(np.unique(labels_arr)) > 1:
                results.classifier_auroc = roc_auc_score(labels_arr, probs_arr)
            else:
                results.classifier_auroc = 0.0

            log.info(
                "[%s] classifier: P=%.4f R=%.4f F1=%.4f AUROC=%.4f",
                split_name,
                results.classifier_precision,
                results.classifier_recall,
                results.classifier_f1,
                results.classifier_auroc,
            )

            # Use classifier predictions for per-type recall
            results.per_type_recall = self._per_type_recall(
                probs_arr, labels_arr, all_types, 0.5
            )
        else:
            # Use cosine optimal threshold for per-type recall
            results.per_type_recall = self._per_type_recall(
                sims_arr, labels_arr, all_types, best_threshold
            )

        for ctype, recall in sorted(results.per_type_recall.items()):
            log.info("  [%s] %s recall: %.4f", split_name, ctype, recall)

        return results

    def _per_type_recall(
        self,
        sims: np.ndarray,
        labels: np.ndarray,
        types: list[str],
        threshold: float,
    ) -> dict[str, float]:
        """Compute recall per clone type (matching MagNET's Table III)."""
        types_arr = np.array(types)
        preds = (sims > threshold).astype(int)

        results = {}
        for ctype in sorted(set(types)):
            if ctype == "Non_Clone":
                continue  # Skip non-clones for per-type recall

            mask = types_arr == ctype
            if mask.sum() == 0:
                continue

            type_labels = labels[mask]
            type_preds = preds[mask]

            # Only compute recall for actual clones of this type
            clone_mask = type_labels == 1
            if clone_mask.sum() == 0:
                continue

            recall = recall_score(
                type_labels[clone_mask],
                type_preds[clone_mask],
                zero_division=0,
            )
            results[ctype] = float(recall)

        return results
