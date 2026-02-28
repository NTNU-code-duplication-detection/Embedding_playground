"""Plot helpers for training and evaluation metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


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
