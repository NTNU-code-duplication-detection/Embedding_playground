"""Plot helpers for training and evaluation metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import det_curve


def _steps(history: list[dict[str, float]]) -> list[float]:
    return [row.get("step", idx + 1) for idx, row in enumerate(history)]


def plot_loss_over_steps(history: list[dict[str, float]], save_path: Path | None = None) -> None:
    """Plot training and validation loss progression."""

    if not history:
        return

    steps = _steps(history)
    train_loss = [row.get("train_loss", float("nan")) for row in history]
    ema_loss = [row.get("ema_loss", float("nan")) for row in history]
    val_loss = [row.get("loss", float("nan")) for row in history]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, train_loss, label="train_loss")
    plt.plot(steps, ema_loss, label="ema_loss")
    plt.plot(steps, val_loss, label="val_loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss Over Steps")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path is not None:
        save_path = save_path.expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_metric_trends(
    history: list[dict[str, float]],
    metrics: tuple[str, ...] = ("auc", "f1", "recall", "accuracy"),
    save_path: Path | None = None,
) -> None:
    """Plot key classification metrics over evaluation steps."""

    if not history:
        return

    steps = _steps(history)
    plt.figure(figsize=(10, 5))
    for metric in metrics:
        values = [row.get(metric, float("nan")) for row in history]
        plt.plot(steps, values, label=metric)

    plt.xlabel("Step")
    plt.ylabel("Metric")
    plt.title("AUC / F1 / Recall / Accuracy Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path is not None:
        save_path = save_path.expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_final_metrics(metrics: dict[str, Any], save_path: Path | None = None) -> None:
    """Plot final evaluation metrics as a bar chart."""

    keys = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "best_f1",
    ]
    values = [float(metrics.get(key, float("nan"))) for key in keys]

    plt.figure(figsize=(9, 5))
    plt.bar(keys, values)
    plt.ylim(0.0, 1.0)
    plt.title("Final Test Metrics")
    plt.ylabel("Score")
    plt.xticks(rotation=20)
    plt.grid(axis="y", alpha=0.3)

    if save_path is not None:
        save_path = save_path.expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_score_distributions(
    y_true: list[int],
    y_score: list[float],
    title: str = "Score Distributions (Clone vs Non-Clone)",
    bins: int = 50,
    save_path: Path | None = None,
) -> None:
    """Plot score histograms for target and non-target classes."""

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_score_arr = np.asarray(y_score, dtype=np.float32)

    pos = y_score_arr[y_true_arr == 1]
    neg = y_score_arr[y_true_arr == 0]

    plt.figure(figsize=(10, 5))
    plt.hist(neg, bins=bins, alpha=0.5, density=True, label="Non-clone (H0)")
    plt.hist(pos, bins=bins, alpha=0.5, density=True, label="Clone (H1)")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path is not None:
        save_path = save_path.expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_det_curve(
    y_true: list[int],
    y_score: list[float],
    title: str = "DET Curve",
    save_path: Path | None = None,
) -> None:
    """Plot Detection Error Tradeoff curve (FPR vs FNR)."""

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_score_arr = np.asarray(y_score, dtype=np.float32)
    fpr, fnr, _ = det_curve(y_true_arr, y_score_arr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, fnr, label="DET")
    plt.plot([0, 1], [0, 1], "--", color="gray", alpha=0.7, label="FPR=FNR")
    plt.xlabel("False Positive Rate")
    plt.ylabel("False Negative Rate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path is not None:
        save_path = save_path.expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
