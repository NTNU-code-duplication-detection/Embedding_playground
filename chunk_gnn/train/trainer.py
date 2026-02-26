"""
Training loop for the siamese chunk-level GNN.

Features:
  - MSE loss on cosine similarity (same as MagNET)
  - Per-epoch evaluation on val and test sets
  - Checkpoint saving (best model + periodic)
  - Gradient norm monitoring (detect exploding/vanishing gradients)
  - NaN/Inf loss detection (halt immediately)
  - Stale training detection (warn if loss hasn't improved in N epochs)
  - Detailed logging to both console and file
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from chunk_gnn.model.siamese import SiameseChunkGNN
from chunk_gnn.train.evaluator import Evaluator
from chunk_gnn.train.losses import CosineContrastiveLoss

log = logging.getLogger(__name__)


class Trainer:
    """Trains the SiameseChunkGNN on BCB clone pairs."""

    def __init__(
        self,
        model: SiameseChunkGNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: dict,
        device: torch.device,
        output_dir: str,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = Path(output_dir)

        # Training config
        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 20)
        self.lr = train_cfg.get("learning_rate", 0.0005)
        self.weight_decay = train_cfg.get("weight_decay", 1e-4)
        self.save_interval = train_cfg.get("save_epoch_interval", 1)
        self.eval_interval = train_cfg.get("eval_epoch_interval", 1)
        self.log_batch_interval = train_cfg.get("log_batch_interval", 100)

        # Evaluation config
        eval_cfg = config.get("evaluation", {})
        self.evaluator = Evaluator(
            threshold_steps=eval_cfg.get("threshold_steps", 200),
        )

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Loss function (selectable via config)
        loss_type = train_cfg.get("loss", "mse_cosine")
        # Strip [INFO] annotation if present
        self.loss_type = loss_type.replace(" [INFO]", "").strip()

        # Check if model uses classifier head — override loss to BCE
        self.use_classifier = model.use_classifier
        if self.use_classifier:
            self.loss_type = "bce_logits"
            self.criterion = nn.BCEWithLogitsLoss()
            log.info("Loss: BCEWithLogitsLoss (classifier head mode)")
        elif self.loss_type == "contrastive":
            self.criterion = CosineContrastiveLoss(
                margin_pos=train_cfg.get("margin_pos", 0.25),
                margin_neg=train_cfg.get("margin_neg", -0.25),
            )
            log.info("Loss: %s", self.criterion)
        elif self.loss_type == "cosine_embedding":
            margin = train_cfg.get("margin", 0.5)
            self.criterion = nn.CosineEmbeddingLoss(margin=margin)
            log.info("Loss: CosineEmbeddingLoss(margin=%.2f)", margin)
        else:
            self.criterion = nn.MSELoss()
            log.info("Loss: MSELoss (on cosine similarity)")

        # LR scheduler (optional, configured via training.scheduler)
        sched_cfg = train_cfg.get("scheduler", None)
        if sched_cfg == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",  # Maximize val F1
                factor=train_cfg.get("scheduler_factor", 0.5),
                patience=train_cfg.get("scheduler_patience", 3),
                min_lr=train_cfg.get("scheduler_min_lr", 1e-6),
            )
        else:
            self.scheduler = None

        # Tracking
        self.best_f1 = 0.0
        self.best_epoch = -1
        self.loss_history: list[float] = []
        self.stale_epochs = 0
        self.stale_threshold = 5  # Warn after 5 epochs without improvement

        # Early stopping on val F1 plateau
        self.early_stop_patience = train_cfg.get("early_stop_patience", 0)
        self.epochs_without_improvement = 0

        # Create output directories
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # File logging
        fh = logging.FileHandler(self.log_dir / "training.log")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(fh)

    def train(self) -> dict:
        """Main training loop. Returns final results dict."""
        log.info("=" * 60)
        log.info("Starting training: %d epochs, lr=%.6f, batch_size=%d",
                 self.epochs, self.lr, self.train_loader.batch_size)
        log.info("Train batches: %d, Val batches: %d, Test batches: %d",
                 len(self.train_loader), len(self.val_loader), len(self.test_loader))
        log.info("Output: %s", self.output_dir)
        log.info("=" * 60)

        all_results = []

        for epoch in range(self.epochs):
            epoch_start = time.time()

            # Train one epoch
            train_loss, grad_norm = self._train_epoch(epoch)
            self.loss_history.append(train_loss)

            # Check for NaN/Inf
            if not torch.isfinite(torch.tensor(train_loss)):
                log.error("NaN/Inf loss at epoch %d! Halting training.", epoch)
                break

            # Check for stale training
            self._check_stale(train_loss)

            epoch_time = time.time() - epoch_start

            # Evaluate
            val_results = None
            test_results = None

            if epoch % self.eval_interval == 0:
                val_results = self.evaluator.evaluate(
                    self.model, self.val_loader, self.device, "val"
                )
                test_results = self.evaluator.evaluate(
                    self.model, self.test_loader, self.device, "test"
                )

                # Track best model (classifier F1 when available, else val F1@opt)
                if self.use_classifier:
                    primary_metric = val_results.classifier_f1
                else:
                    primary_metric = val_results.f1_opt

                if primary_metric > self.best_f1:
                    self.best_f1 = primary_metric
                    self.best_epoch = epoch
                    self._save_checkpoint(epoch, is_best=True)
                    self.epochs_without_improvement = 0
                    log.info(
                        "New best model! Val F1=%.4f at epoch %d",
                        self.best_f1, epoch,
                    )
                else:
                    self.epochs_without_improvement += 1

            # Early stopping check
            if (self.early_stop_patience > 0
                    and self.epochs_without_improvement >= self.early_stop_patience):
                log.info(
                    "Early stopping: val F1 has not improved for %d epochs "
                    "(best=%.4f at epoch %d)",
                    self.early_stop_patience, self.best_f1, self.best_epoch,
                )
                break

            # Step LR scheduler based on val F1
            if self.scheduler is not None and val_results is not None:
                self.scheduler.step(val_results.f1_opt)

            # Save periodic checkpoint
            if epoch % self.save_interval == 0:
                self._save_checkpoint(epoch, is_best=False)

            # Log epoch summary
            current_lr = self.optimizer.param_groups[0]["lr"]
            log.info(
                "Epoch %d/%d: loss=%.6f, grad_norm=%.4f, lr=%.2e, time=%.1fs",
                epoch, self.epochs - 1, train_loss, grad_norm, current_lr, epoch_time,
            )

            # Collect results for this epoch
            epoch_results = {
                "epoch": epoch,
                "train_loss": train_loss,
                "grad_norm": grad_norm,
                "epoch_time": epoch_time,
            }
            if val_results:
                epoch_results["val"] = {
                    "f1_t0": val_results.f1_t0,
                    "f1_opt": val_results.f1_opt,
                    "precision_opt": val_results.precision_opt,
                    "recall_opt": val_results.recall_opt,
                    "threshold": val_results.optimal_threshold,
                }
                if self.use_classifier:
                    epoch_results["val"]["classifier_f1"] = val_results.classifier_f1
                    epoch_results["val"]["classifier_precision"] = val_results.classifier_precision
                    epoch_results["val"]["classifier_recall"] = val_results.classifier_recall
                    epoch_results["val"]["classifier_auroc"] = val_results.classifier_auroc
            if test_results:
                epoch_results["test"] = {
                    "f1_t0": test_results.f1_t0,
                    "f1_opt": test_results.f1_opt,
                    "precision_opt": test_results.precision_opt,
                    "recall_opt": test_results.recall_opt,
                    "threshold": test_results.optimal_threshold,
                    "per_type_recall": test_results.per_type_recall,
                }
                if self.use_classifier:
                    test_cls = epoch_results["test"]
                    test_cls["classifier_f1"] = test_results.classifier_f1
                    test_cls["classifier_precision"] = test_results.classifier_precision
                    test_cls["classifier_recall"] = test_results.classifier_recall
                    test_cls["classifier_auroc"] = test_results.classifier_auroc
            all_results.append(epoch_results)

            # Restore training mode after evaluation
            self.model.train()

        # Final evaluation if last epoch wasn't evaluated
        if self.epochs > 0 and (self.epochs - 1) % self.eval_interval != 0:
            log.info("Final evaluation (last epoch not covered by interval)...")
            val_results = self.evaluator.evaluate(
                self.model, self.val_loader, self.device, "val"
            )
            test_results = self.evaluator.evaluate(
                self.model, self.test_loader, self.device, "test"
            )
            self._save_checkpoint(self.epochs - 1, is_best=False)

            if self.use_classifier:
                final_metric = val_results.classifier_f1
            else:
                final_metric = val_results.f1_opt

            if final_metric > self.best_f1:
                self.best_f1 = final_metric
                self.best_epoch = self.epochs - 1
                self._save_checkpoint(self.epochs - 1, is_best=True)

            # Append final eval results so they appear in results.json
            final_results = {
                "epoch": self.epochs - 1,
                "train_loss": self.loss_history[-1] if self.loss_history else None,
                "val": {
                    "f1_t0": val_results.f1_t0,
                    "f1_opt": val_results.f1_opt,
                    "precision_opt": val_results.precision_opt,
                    "recall_opt": val_results.recall_opt,
                    "threshold": val_results.optimal_threshold,
                },
                "test": {
                    "f1_t0": test_results.f1_t0,
                    "f1_opt": test_results.f1_opt,
                    "precision_opt": test_results.precision_opt,
                    "recall_opt": test_results.recall_opt,
                    "threshold": test_results.optimal_threshold,
                    "per_type_recall": test_results.per_type_recall,
                },
            }
            if self.use_classifier:
                final_results["val"]["classifier_f1"] = val_results.classifier_f1
                final_results["val"]["classifier_precision"] = val_results.classifier_precision
                final_results["val"]["classifier_recall"] = val_results.classifier_recall
                final_results["val"]["classifier_auroc"] = val_results.classifier_auroc
                final_results["test"]["classifier_f1"] = test_results.classifier_f1
                final_results["test"]["classifier_precision"] = test_results.classifier_precision
                final_results["test"]["classifier_recall"] = test_results.classifier_recall
                final_results["test"]["classifier_auroc"] = test_results.classifier_auroc
            all_results.append(final_results)

        # Save all results to JSON
        summary = {
            "best_epoch": self.best_epoch,
            "best_val_f1": self.best_f1,
            "epochs_trained": len(self.loss_history),
            "per_epoch": all_results,
        }
        results_path = self.output_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log.info("Results saved to %s", results_path)

        log.info("=" * 60)
        log.info(
            "Training complete. Best val F1=%.4f at epoch %d",
            self.best_f1, self.best_epoch,
        )
        log.info("=" * 60)

        return summary

    def _train_epoch(self, epoch: int) -> tuple[float, float]:
        """Train for one epoch. Returns (avg_loss, avg_grad_norm)."""
        self.model.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        num_batches = 0

        for batch_idx, (batch1, batch2, labels, _) in enumerate(self.train_loader):
            batch1 = batch1.to(self.device)
            batch2 = batch2.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            emb1, emb2 = self.model(batch1, batch2)

            if self.use_classifier:
                # Classification head: predict logits, BCEWithLogitsLoss
                # Labels must be 0/1 for BCE (convert from +1/-1)
                binary_labels = (labels > 0).float()
                logits = self.model.classifier(emb1, emb2)
                loss = self.criterion(logits, binary_labels)
            elif self.loss_type == "cosine_embedding":
                # CosineEmbeddingLoss takes raw embeddings + labels {+1, -1}
                loss = self.criterion(emb1, emb2, labels)
            else:
                # MSE and contrastive both operate on cosine similarity
                cosine_sim = F.cosine_similarity(emb1, emb2)
                loss = self.criterion(cosine_sim, labels)

            loss.backward()

            # Monitor gradient norms
            grad_norm = self._compute_grad_norm()
            total_grad_norm += grad_norm

            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Periodic batch logging
            if (batch_idx + 1) % self.log_batch_interval == 0:
                log.info(
                    "  Epoch %d, Batch %d/%d, Loss: %.6f, Grad: %.4f",
                    epoch, batch_idx + 1, len(self.train_loader),
                    loss.item(), grad_norm,
                )

        avg_loss = total_loss / max(num_batches, 1)
        avg_grad_norm = total_grad_norm / max(num_batches, 1)

        return avg_loss, avg_grad_norm

    def _compute_grad_norm(self) -> float:
        """Compute total gradient norm across all parameters."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def _check_stale(self, current_loss: float) -> None:
        """Warn if training loss hasn't improved recently."""
        if len(self.loss_history) < 2:
            return

        min_recent = min(self.loss_history[-self.stale_threshold:])
        if current_loss >= min_recent:
            self.stale_epochs += 1
        else:
            self.stale_epochs = 0

        if self.stale_epochs >= self.stale_threshold:
            log.warning(
                "Training may be stale: loss hasn't improved in %d epochs "
                "(current: %.6f, best recent: %.6f)",
                self.stale_epochs, current_loss, min_recent,
            )

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_f1": self.best_f1,
            "best_epoch": self.best_epoch,
            "loss_history": self.loss_history,
        }

        # Save periodic checkpoint
        path = self.ckpt_dir / f"model_epoch_{epoch}.pt"
        torch.save(checkpoint, path)

        # Save best model separately
        if is_best:
            best_path = self.ckpt_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
