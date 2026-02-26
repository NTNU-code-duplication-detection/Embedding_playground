"""
Module for embedding
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

@dataclass
class Embedder:
    model_name: str
    device: str = "cpu"
    max_length: int = 256

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        cfg = AutoConfig.from_pretrained(self.model_name)
        # Many RoBERTa-based checkpoints (GraphCodeBERT) don’t need a pooler.
        # If pooler weights are missing, we don't want a randomly init pooler.
        if getattr(cfg, "model_type", None) in ("roberta", "bert"):
            self.model = AutoModel.from_pretrained(self.model_name, add_pooling_layer=False)
        else:
            self.model = AutoModel.from_pretrained(self.model_name)

        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Returns: [B, D] float32 on CPU
        Pooling: mean pooling over last_hidden_state with attention_mask
        """
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        out = self.model(**enc)
        last = out.last_hidden_state  # [B, T, D]
        mask = enc.get("attention_mask")  # [B, T]

        if mask is None:
            pooled = last.mean(dim=1)
        else:
            mask_f = mask.unsqueeze(-1).to(last.dtype)  # [B, T, 1]
            summed = (last * mask_f).sum(dim=1)         # [B, D]
            denom = mask_f.sum(dim=1).clamp(min=1.0)    # [B, 1]
            pooled = summed / denom

        return pooled.detach().to("cpu").float()
