"""Evaluation utilities for classification, cosine similarity, and uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


@dataclass
class ThresholdMetrics:
    """Metrics computed at a specific threshold."""

    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float


def metrics_at_threshold(y_true: Iterable[int], y_score: Iterable[float], threshold: float = 0.5) -> ThresholdMetrics:
    """Compute accuracy/precision/recall/F1 at one decision threshold."""

    y_true_arr = np.asarray(list(y_true), dtype=np.int64)
    y_score_arr = np.asarray(list(y_score), dtype=np.float32)
    y_pred = (y_score_arr >= threshold).astype(np.int64)

    return ThresholdMetrics(
        threshold=float(threshold),
        accuracy=float(accuracy_score(y_true_arr, y_pred)),
        precision=float(precision_score(y_true_arr, y_pred, zero_division=0)),
        recall=float(recall_score(y_true_arr, y_pred, zero_division=0)),
        f1=float(f1_score(y_true_arr, y_pred, zero_division=0)),
    )


def best_f1_threshold(y_true: Iterable[int], y_score: Iterable[float], num_thresholds: int = 201) -> ThresholdMetrics:
    """Grid-search threshold maximizing F1."""

    y_true_arr = np.asarray(list(y_true), dtype=np.int64)
    y_score_arr = np.asarray(list(y_score), dtype=np.float32)

    best = None
    for thr in np.linspace(0.0, 1.0, num=num_thresholds):
        metric = metrics_at_threshold(y_true_arr, y_score_arr, threshold=float(thr))
        if best is None or metric.f1 > best.f1:
            best = metric

    assert best is not None
    return best


def binary_metrics(y_true: Iterable[int], y_score: Iterable[float], threshold: float = 0.5) -> dict[str, float]:
    """Compute AUC and threshold-based metrics for binary clone detection."""

    y_true_arr = np.asarray(list(y_true), dtype=np.int64)
    y_score_arr = np.asarray(list(y_score), dtype=np.float32)

    auc = float("nan")
    if len(np.unique(y_true_arr)) > 1:
        auc = float(roc_auc_score(y_true_arr, y_score_arr))

    at_thr = metrics_at_threshold(y_true_arr, y_score_arr, threshold=threshold)
    best = best_f1_threshold(y_true_arr, y_score_arr)

    return {
        "auc": auc,
        "accuracy": at_thr.accuracy,
        "precision": at_thr.precision,
        "recall": at_thr.recall,
        "f1": at_thr.f1,
        "threshold": at_thr.threshold,
        "best_f1": best.f1,
        "best_f1_threshold": best.threshold,
        "best_precision": best.precision,
        "best_recall": best.recall,
        "best_accuracy": best.accuracy,
    }


def cosine_scores(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> np.ndarray:
    """Compute cosine similarities for aligned embedding arrays."""

    a = embeddings_a / np.linalg.norm(embeddings_a, axis=1, keepdims=True).clip(min=1e-12)
    b = embeddings_b / np.linalg.norm(embeddings_b, axis=1, keepdims=True).clip(min=1e-12)
    return (a * b).sum(axis=1)


def cosine_similarity_metrics(
    y_true: Iterable[int],
    cosine_sim: Iterable[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Treat cosine sim in [-1,1] as score mapped to [0,1] for metric computation."""

    cosine_arr = np.asarray(list(cosine_sim), dtype=np.float32)
    mapped = (cosine_arr + 1.0) / 2.0
    return binary_metrics(y_true=y_true, y_score=mapped, threshold=(threshold + 1.0) / 2.0)


def uncertainty_summary(prob_matrix: np.ndarray) -> dict[str, float]:
    """Aggregate MC-dropout uncertainty statistics.

    `prob_matrix` shape: [num_examples, num_mc_passes].
    """

    mean_prob = prob_matrix.mean(axis=1)
    std_prob = prob_matrix.std(axis=1)
    entropy = -(mean_prob * np.log(mean_prob + 1e-12) + (1.0 - mean_prob) * np.log(1.0 - mean_prob + 1e-12))

    return {
        "mean_pred_std": float(np.mean(std_prob)),
        "median_pred_std": float(np.median(std_prob)),
        "mean_entropy": float(np.mean(entropy)),
        "max_pred_std": float(np.max(std_prob)),
    }
