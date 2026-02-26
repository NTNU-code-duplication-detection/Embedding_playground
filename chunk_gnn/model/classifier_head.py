"""
Classification head for pair-wise code clone prediction.

Replaces the rigid cosine-similarity-plus-threshold approach with a learned
MLP that takes two graph embeddings and predicts clone/non-clone.

Design: concatenate [h1, h2, h1-h2, h1*h2] -> 2-layer MLP -> binary logit.
This is the standard approach from sentence-pair classification (InferSent,
Sentence-BERT) adapted for our graph embeddings.

Why this helps: cosine similarity treats all 128 dimensions equally and can
only express a linear decision boundary. The MLP can weight dimensions
differently and learn non-linear interactions between embedding features.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """MLP classification head for pair-wise clone detection.

    Takes two embeddings, combines them into a rich feature vector,
    and produces a binary clone/not-clone logit.

    Args:
        embed_dim: Dimension of each input embedding (default: 128).
        hidden_dim: Hidden layer size (default: 256).
        dropout: Dropout rate in hidden layers (default: 0.3).
    """

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Input: [h1, h2, h1-h2, h1*h2] = 4 * embed_dim
        input_dim = 4 * embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, h1: torch.Tensor, h2: torch.Tensor
    ) -> torch.Tensor:
        """Combine two embeddings and classify as clone/non-clone.

        Args:
            h1: First graph embeddings, shape (batch_size, embed_dim).
            h2: Second graph embeddings, shape (batch_size, embed_dim).

        Returns:
            Raw logits, shape (batch_size,). Apply sigmoid for probabilities.
        """
        # Build pair representation with multiple interaction features
        combined = torch.cat([
            h1,
            h2,
            h1 - h2,       # Element-wise difference
            h1 * h2,       # Element-wise product (feature co-activation)
        ], dim=-1)

        # MLP outputs raw logit (no sigmoid -- use BCEWithLogitsLoss)
        logit = self.mlp(combined).squeeze(-1)
        return logit
