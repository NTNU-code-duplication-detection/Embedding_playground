"""
Siamese wrapper for the chunk-level GNN.

Processes two code graphs through the same shared encoder and produces
similarity scores. The encoder weights are shared (siamese architecture)
so both graphs are embedded in the same space.

Supports two comparison modes (selectable via config):
  - "cosine": Cosine similarity + threshold (original approach)
  - "mlp_classifier": Learned MLP classification head (BCEWithLogitsLoss)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from chunk_gnn.model.chunk_gnn import ChunkGNNEncoder
from chunk_gnn.model.classifier_head import ClassifierHead


class SiameseChunkGNN(nn.Module):
    """Siamese network: two graphs through shared GNN encoder.

    The encoder is always shared (siamese property preserved). Only the
    comparison mechanism differs between cosine and classifier modes.
    """

    def __init__(
        self,
        encoder: ChunkGNNEncoder,
        classifier: ClassifierHead | None = None,
    ):
        super().__init__()
        self.encoder = encoder          # Shared weights for both graphs
        self.classifier = classifier    # None = cosine mode, else classifier mode

    @property
    def use_classifier(self) -> bool:
        """Whether this model uses a learned classifier head."""
        return self.classifier is not None

    def forward(
        self, batch1: Batch, batch2: Batch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode two batched graphs into embeddings.

        Args:
            batch1: Batched graphs for the first code in each pair
            batch2: Batched graphs for the second code in each pair

        Returns:
            (emb1, emb2) — both shape (batch_size, out_dim)
        """
        emb1 = self.encoder(batch1.x, batch1.edge_index, batch1.batch)
        emb2 = self.encoder(batch2.x, batch2.edge_index, batch2.batch)
        return emb1, emb2

    def similarity(
        self, batch1: Batch, batch2: Batch
    ) -> torch.Tensor:
        """Compute cosine similarity between paired graph embeddings.

        Always uses cosine similarity regardless of mode, because this is
        needed for evaluation compatibility with previous runs.

        Returns:
            Tensor of shape (batch_size,) with values in [-1, 1]
        """
        emb1, emb2 = self.forward(batch1, batch2)
        return F.cosine_similarity(emb1, emb2)

    def classify(
        self, batch1: Batch, batch2: Batch
    ) -> torch.Tensor:
        """Predict clone/non-clone via the classification head.

        Only available when classifier is not None.

        Returns:
            Raw logits, shape (batch_size,). Apply sigmoid for probabilities.
        """
        if self.classifier is None:
            raise RuntimeError(
                "classify() requires a classifier head. "
                "Set similarity mode to 'mlp_classifier' in config."
            )
        emb1, emb2 = self.forward(batch1, batch2)
        return self.classifier(emb1, emb2)


def build_model(config: dict, device: torch.device) -> SiameseChunkGNN:
    """Build a SiameseChunkGNN from config dict.

    Expected config keys (under 'model'):
        gnn_hidden_dim: int (default 256)
        gnn_output_dim: int (default 128)
        gnn_layers: int (default 2)
        dropout: float (default 0.1)
        gnn_type: str -- "GCNConv" or "GATConv" (default "GCNConv")
        num_heads: int (default 4, only for GATConv)
        similarity: str -- "cosine" or "mlp_classifier" (default "cosine")
        classifier_hidden_dim: int (default 256, only for mlp_classifier)
        classifier_dropout: float (default 0.3, only for mlp_classifier)

    Expected config keys (under 'embedding'):
        embedding_dim: int (default 768)
    """
    model_cfg = config.get("model", {})
    emb_cfg = config.get("embedding", {})

    out_dim = model_cfg.get("gnn_output_dim", 128)

    encoder = ChunkGNNEncoder(
        in_dim=emb_cfg.get("embedding_dim", 768),
        hidden_dim=model_cfg.get("gnn_hidden_dim", 256),
        out_dim=out_dim,
        num_layers=model_cfg.get("gnn_layers", 2),
        dropout=model_cfg.get("dropout", 0.1),
        input_projection=model_cfg.get("input_projection", False),
        residual=model_cfg.get("residual", False),
        l2_normalize=model_cfg.get("l2_normalize", False),
        gnn_type=model_cfg.get("gnn_type", "GCNConv"),
        num_heads=model_cfg.get("num_heads", 4),
    )

    # Build classifier head if configured
    similarity_mode = model_cfg.get("similarity", "cosine")
    classifier = None

    if similarity_mode == "mlp_classifier":
        classifier = ClassifierHead(
            embed_dim=out_dim,
            hidden_dim=model_cfg.get("classifier_hidden_dim", 256),
            dropout=model_cfg.get("classifier_dropout", 0.3),
        )

    model = SiameseChunkGNN(encoder, classifier=classifier).to(device)

    # Log parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    encoder_params = sum(p.numel() for p in encoder.parameters())
    classifier_params = sum(p.numel() for p in classifier.parameters()) if classifier else 0

    print(
        f"SiameseChunkGNN: {total_params:,} total params, "
        f"{trainable_params:,} trainable "
        f"(encoder: {encoder_params:,}, classifier: {classifier_params:,})"
    )

    return model
