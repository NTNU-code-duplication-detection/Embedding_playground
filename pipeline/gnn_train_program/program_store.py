"""Module pipeline/gnn_train_program/program_store.py."""

import json
from pathlib import Path
from typing import Any, Dict
import torch


class ProgramStore:
    def __init__(self, index_path: Path, max_cached: int = 64):
        self.index_path = index_path
        self.max_cached = max_cached

        with open(index_path, "r") as f:
            raw = json.load(f)

        # Normalize keys in items
        norm_items = {}
        for k, v in raw["items"].items():
            try:
                nk = Path(k).expanduser().resolve().as_posix()
            except Exception:
                nk = k
            norm_items[nk] = v
        self.items = norm_items

        self.cache: Dict[str, Any] = {}

    @staticmethod
    def _norm_key(p: str) -> str:
        try:
            return Path(p).expanduser().resolve().as_posix()
        except Exception:
            return str(p)

    def load_program_methods(self, source_path: str):
        # normalize lookup key (important!)
        key = str(Path(source_path).expanduser().resolve())

        item = self.items.get(key)
        if item is None:
            return None

        # NEW: support your index schema
        shards_dir = item.get("shards_dir")
        if shards_dir:
            shards_path = Path(shards_dir)
        else:
            # fallback if shards_dir is missing
            shards_path = Path(item["artifact_dir"]) / "embed_cache" / "shards"

        if not shards_path.exists():
            return None

        shard_files = sorted(shards_path.glob("*.pt"))
        if not shard_files:
            return None

        # Load all method-records for this program
        records = []
        for sf in shard_files:
            part = torch.load(sf, map_location="cpu")
            # each shard is typically a List[dict] (records)
            if isinstance(part, list):
                records.extend(part)
            else:
                records.append(part)

        return records
