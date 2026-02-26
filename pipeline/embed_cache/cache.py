"""
Module for caching embeddings
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import torch

from embed_cache.util import ensure_dir

class DiskEmbeddingCache:
    """
    Simple content-addressed cache:
      cache_dir/<sha1>.pt  contains a single 1D float tensor [D]
    """
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        ensure_dir(self.cache_dir)

    def path_for(self, key: str) -> Path:
        """
        Get path for key
        """
        return self.cache_dir / f"{key}.pt"

    def get(self, key: str) -> Optional[torch.Tensor]:
        """
        Get embeddings from key
        """
        p = self.path_for(key)
        if not p.exists():
            return None
        try:
            t = torch.load(p, map_location="cpu")
            if isinstance(t, torch.Tensor) and t.dim() == 1:
                return t
        except Exception:
            return None
        return None

    def put(self, key: str, vec: torch.Tensor) -> None:
        """
        Save embedding at key
        """
        p = self.path_for(key)
        # Always save on CPU for portability
        torch.save(vec.detach().to("cpu"), p)
