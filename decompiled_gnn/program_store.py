"""Program embedding shard access helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class ProgramStore:
    """Load per-program method tensor records with a small in-memory cache."""

    def __init__(self, program_index_path: Path, max_cached: int = 64):
        self.program_index_path = program_index_path.expanduser().resolve()
        self.max_cached = max(1, int(max_cached))

        raw = json.loads(self.program_index_path.read_text(encoding="utf-8"))
        self.items = {
            str(Path(source).expanduser().resolve()): data
            for source, data in raw.get("items", {}).items()
        }
        self.cache: dict[str, list[dict[str, Any]]] = {}
        self.cache_order: list[str] = []

    def _evict_if_needed(self) -> None:
        while len(self.cache_order) > self.max_cached:
            oldest = self.cache_order.pop(0)
            self.cache.pop(oldest, None)

    def load_program_methods(self, source_path: str) -> list[dict[str, Any]] | None:
        """Load method tensor records for one source path."""

        key = str(Path(source_path).expanduser().resolve())
        if key in self.cache:
            return self.cache[key]

        item = self.items.get(key)
        if item is None:
            return None

        shard_path_str = item.get("embedding_pt") or item.get("embed_shard")
        if not shard_path_str:
            return None

        shard_path = Path(shard_path_str).expanduser().resolve()
        if not shard_path.exists():
            return None

        records = torch.load(shard_path, map_location="cpu")
        if not isinstance(records, list):
            return None

        self.cache[key] = records
        self.cache_order.append(key)
        self._evict_if_needed()
        return records
