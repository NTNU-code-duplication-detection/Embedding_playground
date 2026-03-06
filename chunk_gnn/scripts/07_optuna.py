"""
07_optuna.py — Optuna hyperparameter optimization for GATConv on BigCloneBench.

Uses the same training pipeline as 03_train.py but wraps it with Optuna
to systematically search over learning rate, hidden dim, dropout, etc.

The objective is to maximize validation classifier F1. Each trial trains
for up to 30 epochs with early stopping (patience=10). Optuna prunes
unpromising trials using MedianPruner after 3 epochs.

Usage:
    python scripts/07_optuna.py \
        --bcb_root ~/Multigraph_match_optimized/data/data_source/dataset_bigclonebench \
        --cache_dir ~/chunk_gnn_cache \
        --output_dir ~/chunk_gnn_out/optuna_gat \
        --n_trials 50 \
        --device cuda

On IDUN:
    sbatch slurm/optuna.slurm
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
import optuna
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(trial: optuna.Trial) -> dict:
    """Build a config dict with Optuna-sampled hyperparameters.

    Search space based on Exp2 (GATConv) as the baseline, exploring
    variations that could improve WT3/T4 recall and overall F1.
    """
    lr = trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    gnn_dropout = trial.suggest_float("gnn_dropout", 0.05, 0.3)
    classifier_dropout = trial.suggest_float("classifier_dropout", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    return {
        "data": {
            "bcb_root": "data/data_source/dataset_bigclonebench",
            "clone_labels": "clone_labels_typed.txt",
            "dataset_files": "dataset_files",
            "skip_functions": ["37044", "4892654", "6966398", "7550876"],
        },
        "embedding": {
            "model_name": "microsoft/unixcoder-base",
            "embedding_dim": 768,
            "max_tokens": 512,
            "batch_size": 32,
        },
        "graph": {
            "edge_types": ["sequential", "parent_child"],
            "self_loops": True,
        },
        "model": {
            "gnn_hidden_dim": hidden_dim,
            "gnn_output_dim": 128,
            "gnn_layers": 2,
            "gnn_type": "GATConv",
            "num_heads": num_heads,
            "pooling": "global_mean_pool",
            "dropout": gnn_dropout,
            "input_projection": True,
            "residual": False,
            "l2_normalize": False,
            "similarity": "mlp_classifier",
            "classifier_hidden_dim": hidden_dim,
            "classifier_dropout": classifier_dropout,
        },
        "training": {
            "epochs": 30,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "loss": "bce_logits",
            "label_positive": 1.0,
            "label_negative": -1.0,
            "save_epoch_interval": 999,  # Don't save periodic checkpoints
            "eval_epoch_interval": 1,
            "log_batch_interval": 500,  # Less verbose during search
            "seed": 42,
            "scheduler": "reduce_on_plateau",
            "scheduler_patience": 5,
            "scheduler_factor": 0.5,
            "scheduler_min_lr": 1e-6,
            "early_stop_patience": 10,
        },
        "evaluation": {
            "threshold_sweep": True,
            "threshold_range": [-1.0, 1.0],
            "threshold_steps": 200,
        },
    }


def objective(
    trial: optuna.Trial,
    args: argparse.Namespace,
    loader: BCBLoader,
) -> float:
    """Optuna objective: train one trial and return best val classifier F1."""

    config = build_config(trial)
    train_cfg = config["training"]

    set_seed(train_cfg["seed"])

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Log trial params
    log.info(
        "Trial %d: lr=%.5f, hidden=%d, heads=%d, gnn_drop=%.2f, "
        "cls_drop=%.2f, wd=%.5f, bs=%d",
        trial.number,
        trial.params["learning_rate"],
        trial.params["hidden_dim"],
        trial.params["num_heads"],
        trial.params["gnn_dropout"],
        trial.params["classifier_dropout"],
        trial.params["weight_decay"],
        trial.params["batch_size"],
    )

    # Create dataloaders (batch_size may vary per trial)
    train_pairs = loader.get_split("train")
    val_pairs = loader.get_split("val")
    test_pairs = loader.get_split("test")

    train_loader, val_loader, test_loader = create_dataloaders(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        cache_dir=args.cache_dir,
        batch_size=train_cfg["batch_size"],
        num_workers=args.num_workers,
        label_positive=train_cfg["label_positive"],
        label_negative=train_cfg["label_negative"],
    )

    # Build model
    model = build_model(config, device)

    # Create output dir for this trial
    trial_dir = os.path.join(args.output_dir, f"trial_{trial.number:04d}")
    os.makedirs(trial_dir, exist_ok=True)

    # Save config
    with open(os.path.join(trial_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        output_dir=trial_dir,
    )

    results = trainer.train()

    best_val_f1 = results["best_val_f1"]
    log.info("Trial %d complete: best val F1 = %.4f", trial.number, best_val_f1)

    return best_val_f1


def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for GATConv chunk-GNN"
    )
    parser.add_argument("--bcb_root", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--n_trials", type=int, default=50,
        help="Number of Optuna trials (default: 50)",
    )
    parser.add_argument(
        "--study_name", type=str, default="gatconv_hpo",
        help="Optuna study name (default: gatconv_hpo)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load BCB data once (shared across all trials)
    log.info("Loading BCB data (shared across trials)...")
    # Use a dummy config for data loading — actual params come from trials
    data_cfg = {
        "clone_labels": "clone_labels_typed.txt",
        "dataset_files": "dataset_files",
        "skip_functions": ["37044", "4892654", "6966398", "7550876"],
    }
    loader = BCBLoader(bcb_root=args.bcb_root, config=data_cfg, labels_file="clone_labels_typed.txt")
    log.info("BCB data loaded.")

    # Create Optuna study
    # Use SQLite storage so the study survives crashes and can be resumed
    db_path = os.path.join(args.output_dir, "optuna_study.db")
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,  # Resume if study already exists
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    log.info("Starting Optuna search: %d trials", args.n_trials)
    log.info("Study DB: %s", db_path)

    study.optimize(
        lambda trial: objective(trial, args, loader),
        n_trials=args.n_trials,
    )

    # Print results
    log.info("")
    log.info("=" * 60)
    log.info("OPTUNA SEARCH COMPLETE")
    log.info("=" * 60)
    log.info("Best trial: %d", study.best_trial.number)
    log.info("Best val F1: %.4f", study.best_value)
    log.info("Best params:")
    for key, value in study.best_params.items():
        log.info("  %s: %s", key, value)

    # Save summary
    summary = {
        "best_trial": study.best_trial.number,
        "best_val_f1": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": t.state.name,
            }
            for t in study.trials
        ],
    }
    summary_path = os.path.join(args.output_dir, "optuna_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
