"""
Loss functions for siamese chunk-level GNN training.

Provides alternatives to MSE on cosine similarity:
  - CosineContrastiveLoss: Hinge-margin loss centered on t=0 decision boundary.
    Stops gradient for pairs that already satisfy their margin, preventing the
    directional collapse seen with MSE (which always pushes toward ±1).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CosineContrastiveLoss(nn.Module):
    """Margin-based contrastive loss operating on cosine similarity.

    For clone pairs (label > 0): penalizes if similarity < margin_pos
    For non-clone pairs (label <= 0): penalizes if similarity > margin_neg

    Once a pair satisfies its margin, it contributes zero loss and gradient.
    This focuses learning on the decision boundary rather than pushing
    all clones to +1 and all non-clones to -1 (the MSE failure mode).

    Args:
        margin_pos: Clone pairs must have similarity above this (default: 0.25).
        margin_neg: Non-clone pairs must have similarity below this (default: -0.25).
    """

    def __init__(self, margin_pos: float = 0.25, margin_neg: float = -0.25):
        super().__init__()
        if margin_neg >= margin_pos:
            raise ValueError(
                f"margin_neg ({margin_neg}) must be less than "
                f"margin_pos ({margin_pos})"
            )
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg

    def forward(
        self, cosine_sim: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss.

        Args:
            cosine_sim: Cosine similarities, shape (batch_size,), range [-1, 1].
            labels: Shape (batch_size,). Values +1 (clone) or -1 (non-clone).

        Returns:
            Scalar loss value.
        """
        is_clone = (labels > 0).float()

        # Clone pairs: penalize when similarity is BELOW margin_pos
        pos_loss = is_clone * torch.clamp(self.margin_pos - cosine_sim, min=0.0)

        # Non-clone pairs: penalize when similarity is ABOVE margin_neg
        neg_loss = (1.0 - is_clone) * torch.clamp(
            cosine_sim - self.margin_neg, min=0.0
        )

        return (pos_loss + neg_loss).mean()

    def __repr__(self) -> str:
        return (
            f"CosineContrastiveLoss(margin_pos={self.margin_pos}, "
            f"margin_neg={self.margin_neg})"
        )
