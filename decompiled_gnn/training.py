"""Training and evaluation loops for program-level clone detection."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import ModelConfig, TrainingConfig, UncertaintyConfig
from metrics import binary_metrics, cosine_similarity_metrics, uncertainty_summary
from model import ProgramCloneModel
from program_store import ProgramStore

Pair = tuple[str, str, int]


def set_seed(seed: int) -> None:
    """Set random seeds for deterministic experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def contrastive_cosine_loss(
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Contrastive loss over cosine similarity."""

    cosine = F.cosine_similarity(emb_a, emb_b, dim=-1)
    pos = labels * (1.0 - cosine).pow(2)
    neg = (1.0 - labels) * F.relu(cosine - margin).pow(2)
    return (pos + neg).mean()


def collect_valid_pairs(
    pair_iter: Iterator[Pair],
    store: ProgramStore,
    target_pairs: int,
    max_tries: int | None = None,
) -> list[Pair]:
    """Collect a fixed number of pairs that exist in the program store."""

    if max_tries is None:
        max_tries = max(target_pairs * 5, 1000)

    pairs: list[Pair] = []
    tries = 0
    while len(pairs) < target_pairs and tries < max_tries:
        tries += 1
        a, b, label = next(pair_iter)
        if store.load_program_methods(a) is None or store.load_program_methods(b) is None:
            continue
        pairs.append((a, b, int(label)))
    return pairs


def pair_scores(
    model: ProgramCloneModel,
    methods_a: list[dict],
    methods_b: list[dict],
    device: str,
    loss_type: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return `(logit, cosine_sim, score_01)` where score_01 is used for metrics."""

    out = model.forward_pair(methods_a=methods_a, methods_b=methods_b, device=device)
    cosine = ProgramCloneModel.cosine_similarity(out.emb_a, out.emb_b)

    if loss_type == "bce":
        score = torch.sigmoid(out.logit)
    elif loss_type == "contrastive":
        score = (cosine + 1.0) / 2.0
    else:
        raise ValueError(f"Unsupported loss_type={loss_type}")

    return out.logit, cosine, score


def evaluate_pairs(
    model: ProgramCloneModel,
    store: ProgramStore,
    pairs: list[Pair],
    device: str,
    model_cfg: ModelConfig,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate a fixed pair list and compute metrics."""

    model.eval()
    losses: list[torch.Tensor] = []
    y_true: list[int] = []
    y_score: list[float] = []

    with torch.no_grad():
        for a, b, label in pairs:
            methods_a = store.load_program_methods(a)
            methods_b = store.load_program_methods(b)
            if methods_a is None or methods_b is None:
                continue

            logit, cosine, score = pair_scores(
                model=model,
                methods_a=methods_a,
                methods_b=methods_b,
                device=device,
                loss_type=model_cfg.loss_type,
            )

            target = torch.tensor(float(label), device=device)
            if model_cfg.loss_type == "bce":
                loss = F.binary_cross_entropy_with_logits(logit, target)
            else:
                contrastive = (
                    target * (1.0 - cosine).pow(2)
                    + (1.0 - target) * F.relu(cosine - model_cfg.contrastive_margin).pow(2)
                )
                if model_cfg.contrastive_head_weight > 0.0:
                    bce = F.binary_cross_entropy_with_logits(logit, target)
                    loss = contrastive + model_cfg.contrastive_head_weight * bce
                else:
                    loss = contrastive

            losses.append(loss.detach().cpu())
            y_true.append(int(label))
            y_score.append(float(score.item()))

    if not y_true:
        return {
            "loss": float("nan"),
            "used": 0,
            "pos": 0,
            "neg": 0,
            "auc": float("nan"),
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "threshold": threshold,
            "best_f1": float("nan"),
            "best_f1_threshold": float("nan"),
            "best_precision": float("nan"),
            "best_recall": float("nan"),
            "best_accuracy": float("nan"),
        }

    metric_values = binary_metrics(y_true=y_true, y_score=y_score, threshold=threshold)
    return {
        "loss": float(torch.stack(losses).mean().item()) if losses else float("nan"),
        "used": len(y_true),
        "pos": int(sum(y_true)),
        "neg": int(len(y_true) - sum(y_true)),
        **metric_values,
    }


def evaluate_stream(
    model: ProgramCloneModel,
    store: ProgramStore,
    pair_iter: Iterator[Pair],
    device: str,
    model_cfg: ModelConfig,
    num_pairs: int,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate by first collecting valid pairs from an iterator."""

    pairs = collect_valid_pairs(pair_iter=pair_iter, store=store, target_pairs=num_pairs)
    return evaluate_pairs(
        model=model,
        store=store,
        pairs=pairs,
        device=device,
        model_cfg=model_cfg,
        threshold=threshold,
    )


def train_loop(
    model: ProgramCloneModel,
    store: ProgramStore,
    train_iter: Iterator[Pair],
    val_iter: Iterator[Pair],
    device: str,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    seed: int,
) -> tuple[ProgramCloneModel, list[dict[str, float]]]:
    """Main training loop with validation-driven LR scheduling."""

    set_seed(seed)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=train_cfg.scheduler_factor,
        patience=train_cfg.scheduler_patience,
    )

    history: list[dict[str, float]] = []
    ema_loss = None
    best_metric_value = float("-inf")
    best_metric_step = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    metric_name = train_cfg.early_stopping_metric

    for step in tqdm(range(1, train_cfg.steps + 1), desc="Training(decompiled_gnn)"):
        model.train()

        batch_pairs = collect_valid_pairs(
            pair_iter=train_iter,
            store=store,
            target_pairs=train_cfg.batch_size,
            max_tries=train_cfg.batch_size * 20,
        )
        if not batch_pairs:
            continue

        logits: list[torch.Tensor] = []
        emb_a_batch: list[torch.Tensor] = []
        emb_b_batch: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []

        for a, b, label in batch_pairs:
            methods_a = store.load_program_methods(a)
            methods_b = store.load_program_methods(b)
            if methods_a is None or methods_b is None:
                continue

            out = model.forward_pair(methods_a=methods_a, methods_b=methods_b, device=device)
            logits.append(out.logit)
            emb_a_batch.append(out.emb_a)
            emb_b_batch.append(out.emb_b)
            labels.append(torch.tensor(float(label), device=device))

        if not labels:
            continue

        logits_t = torch.stack(logits)
        emb_a_t = torch.stack(emb_a_batch)
        emb_b_t = torch.stack(emb_b_batch)
        labels_t = torch.stack(labels)

        if model_cfg.loss_type == "bce":
            loss = F.binary_cross_entropy_with_logits(logits_t, labels_t)
        elif model_cfg.loss_type == "contrastive":
            contrastive = contrastive_cosine_loss(
                emb_a=emb_a_t,
                emb_b=emb_b_t,
                labels=labels_t,
                margin=model_cfg.contrastive_margin,
            )
            if model_cfg.contrastive_head_weight > 0.0:
                bce = F.binary_cross_entropy_with_logits(logits_t, labels_t)
                loss = contrastive + model_cfg.contrastive_head_weight * bce
            else:
                loss = contrastive
        else:
            raise ValueError(f"Unsupported loss_type={model_cfg.loss_type}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        ema_loss = loss_value if ema_loss is None else 0.95 * ema_loss + 0.05 * loss_value

        if step % train_cfg.log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"step={step} train_loss={loss_value:.4f} ema={ema_loss:.4f} lr={lr:.6f}")

        if step % train_cfg.eval_every == 0:
            val_metrics = evaluate_stream(
                model=model,
                store=store,
                pair_iter=val_iter,
                device=device,
                model_cfg=model_cfg,
                num_pairs=train_cfg.val_pairs,
                threshold=0.5,
            )
            scheduler.step(val_metrics["loss"])

            metric_value = float(val_metrics.get(metric_name, float("nan")))
            metric_is_valid = bool(np.isfinite(metric_value))
            improved = False
            if metric_is_valid:
                if metric_value > (best_metric_value + train_cfg.early_stopping_min_delta):
                    improved = True
                    best_metric_value = metric_value
                    best_metric_step = step
                    if train_cfg.early_stopping_restore_best:
                        best_state_dict = {
                            key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()
                        }

            if improved:
                print(
                    "[EARLY_STOP] "
                    f"improved {metric_name} to {best_metric_value:.4f} at step={step}"
                )

            steps_since_improvement = (
                step - best_metric_step if best_metric_step > 0 else step
            )
            early_stop_triggered = (
                train_cfg.early_stopping_enabled
                and best_metric_step > 0
                and steps_since_improvement >= train_cfg.early_stopping_patience_steps
            )

            row = {
                "step": float(step),
                "train_loss": float(loss_value),
                "ema_loss": float(ema_loss),
                **{k: float(v) for k, v in val_metrics.items()},
                "lr": float(optimizer.param_groups[0]["lr"]),
                "es_metric_value": float(metric_value) if metric_is_valid else float("nan"),
                "es_best_metric_value": (
                    float(best_metric_value) if np.isfinite(best_metric_value) else float("nan")
                ),
                "es_best_metric_step": float(best_metric_step),
                "es_steps_since_improvement": float(steps_since_improvement),
                "es_improved": float(improved),
                "es_triggered": float(early_stop_triggered),
            }
            history.append(row)

            print(
                "[VAL] "
                f"step={step} used={int(val_metrics['used'])} "
                f"loss={val_metrics['loss']:.4f} auc={val_metrics['auc']:.4f} "
                f"acc={val_metrics['accuracy']:.4f} f1={val_metrics['f1']:.4f} "
                f"recall={val_metrics['recall']:.4f}"
            )

            if early_stop_triggered:
                print(
                    "[EARLY_STOP] "
                    f"stopping at step={step} (no {metric_name} improvement > "
                    f"{train_cfg.early_stopping_min_delta} for "
                    f"{train_cfg.early_stopping_patience_steps} steps)"
                )
                break

    if (
        train_cfg.early_stopping_restore_best
        and best_state_dict is not None
    ):
        model.load_state_dict(best_state_dict)
        print(
            "[EARLY_STOP] "
            f"restored best weights from step={best_metric_step} "
            f"with {metric_name}={best_metric_value:.4f}"
        )

    return model, history


def mc_dropout_eval(
    model: ProgramCloneModel,
    store: ProgramStore,
    pairs: list[Pair],
    device: str,
    model_cfg: ModelConfig,
    uncertainty_cfg: UncertaintyConfig,
) -> dict[str, float]:
    """Run MC-dropout over fixed pairs and summarize predictive uncertainty."""

    if not uncertainty_cfg.enabled or uncertainty_cfg.mc_dropout_passes <= 1:
        return {}

    valid_pairs: list[Pair] = []
    for a, b, label in pairs:
        if store.load_program_methods(a) is None or store.load_program_methods(b) is None:
            continue
        valid_pairs.append((a, b, label))

    if not valid_pairs:
        return {}

    probs_per_pass: list[list[float]] = []
    model.train()  # keep dropout active

    for _ in range(uncertainty_cfg.mc_dropout_passes):
        pass_probs: list[float] = []
        with torch.no_grad():
            for a, b, _ in valid_pairs:
                methods_a = store.load_program_methods(a)
                methods_b = store.load_program_methods(b)
                if methods_a is None or methods_b is None:
                    continue

                logit, cosine, score = pair_scores(
                    model=model,
                    methods_a=methods_a,
                    methods_b=methods_b,
                    device=device,
                    loss_type=model_cfg.loss_type,
                )
                if model_cfg.loss_type == "bce":
                    pass_probs.append(float(torch.sigmoid(logit).item()))
                else:
                    pass_probs.append(float(score.item()))

        probs_per_pass.append(pass_probs)

    prob_matrix = np.asarray(probs_per_pass, dtype=np.float32).T
    return uncertainty_summary(prob_matrix)


def cosine_eval_pairs(
    model: ProgramCloneModel,
    store: ProgramStore,
    pairs: list[Pair],
    device: str,
    threshold: float = 0.0,
) -> dict[str, float]:
    """Evaluate using cosine similarity directly instead of classifier logits."""

    model.eval()
    cosine_scores: list[float] = []
    y_true: list[int] = []

    with torch.no_grad():
        for a, b, label in pairs:
            methods_a = store.load_program_methods(a)
            methods_b = store.load_program_methods(b)
            if methods_a is None or methods_b is None:
                continue

            out = model.forward_pair(methods_a=methods_a, methods_b=methods_b, device=device)
            cosine = ProgramCloneModel.cosine_similarity(out.emb_a, out.emb_b)
            cosine_scores.append(float(cosine.item()))
            y_true.append(int(label))

    if not y_true:
        return {
            "used": 0.0,
            "pos": 0.0,
            "neg": 0.0,
            "auc": float("nan"),
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "threshold": threshold,
            "best_f1": float("nan"),
            "best_f1_threshold": float("nan"),
            "best_precision": float("nan"),
            "best_recall": float("nan"),
            "best_accuracy": float("nan"),
        }

    metrics = cosine_similarity_metrics(y_true=y_true, cosine_sim=cosine_scores, threshold=threshold)
    return {
        "used": float(len(y_true)),
        "pos": float(sum(y_true)),
        "neg": float(len(y_true) - sum(y_true)),
        **{k: float(v) for k, v in metrics.items()},
    }


def save_training_artifacts(
    model: ProgramCloneModel,
    history: list[dict[str, float]],
    out_dir: str,
) -> dict[str, str]:
    """Persist trained weights and history."""

    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / "program_clone_model.pt"
    history_path = out / "training_history.json"

    torch.save(model.state_dict(), model_path)

    import json

    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"model_path": str(model_path), "history_path": str(history_path)}
