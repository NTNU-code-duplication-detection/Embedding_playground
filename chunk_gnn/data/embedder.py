"""
UniXcoder chunk embedder.

Embeds code chunks using the microsoft/unixcoder-base model. Uses batched
inference for efficiency and caches the model to avoid reloading.

The embedder is designed for the pre-compute step (Phase 3): we embed all
chunks once and save the results to disk, so this code does NOT run during
training.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
from transformers import AutoModel, AutoTokenizer

from chunk_gnn.data.chunker import Chunk

log = logging.getLogger(__name__)


class ChunkEmbedder:
    """Embeds code chunks using UniXcoder.

    Produces a 768-dim vector per chunk via mean pooling across tokens.
    """

    def __init__(
        self,
        model_name: str = "microsoft/unixcoder-base",
        device: str | None = None,
        batch_size: int = 32,
        max_tokens: int = 512,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_tokens = max_tokens

        log.info("Loading %s on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Embedding dimension (768 for unixcoder-base)
        self.embedding_dim = self.model.config.hidden_size
        log.info(
            "Embedder ready: dim=%d, batch_size=%d, max_tokens=%d",
            self.embedding_dim,
            self.batch_size,
            self.max_tokens,
        )

    @torch.no_grad()
    def embed_chunks(self, chunks: Sequence[Chunk]) -> torch.Tensor:
        """Embed a list of chunks into a (num_chunks, 768) tensor.

        Uses batched inference and mean pooling across tokens per chunk.
        Returns float32 tensor (caller can convert to float16 for storage).
        """
        if not chunks:
            return torch.empty(0, self.embedding_dim)

        texts = [c.text for c in chunks]
        all_embeddings = []

        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]
            batch_emb = self._embed_batch(batch_texts)
            all_embeddings.append(batch_emb)

        return torch.cat(all_embeddings, dim=0)

    @torch.no_grad()
    def _embed_batch(self, texts: list[str]) -> torch.Tensor:
        """Embed a batch of text strings. Returns (batch_size, 768)."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**encoded)

        # Mean pool across token dimension, respecting attention mask
        # outputs.last_hidden_state: (batch, seq_len, 768)
        # attention_mask: (batch, seq_len)
        token_embeddings = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1)  # (batch, seq_len, 1)

        # Zero out padding tokens, then average over non-padding tokens
        masked = token_embeddings * attention_mask
        summed = masked.sum(dim=1)  # (batch, 768)
        counts = attention_mask.sum(dim=1).clamp(min=1)  # (batch, 1)
        mean_pooled = summed / counts  # (batch, 768)

        return mean_pooled.cpu()

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        """Convenience: embed raw text strings (not Chunk objects).

        Useful for testing without constructing Chunk objects.
        """
        all_embeddings = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            all_embeddings.append(self._embed_batch(batch))
        return torch.cat(all_embeddings, dim=0)
