"""Module pipeline/gnn_train/shard_index.py."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch


@dataclass(frozen=True)
class MethodLoc:
    """
    Method locations
    """
    shard: str
    idx: int


def build_index(shards_dir: Path) -> Dict[str, MethodLoc]:
    """
    Builds index of shards connected to program
    """
    index: Dict[str, MethodLoc] = {}

    shard_files = sorted(shards_dir.glob("methods_*.pt"))
    for sf in shard_files:
        data = torch.load(sf, map_location="cpu")
        for i, item in enumerate(data):
            mid = item.get("method_id")
            if mid is None:
                continue
            # If duplicates exist, keep first occurrence deterministically
            if mid not in index:
                index[mid] = MethodLoc(shard=sf.name, idx=int(i))

    return index


def save_index(index: Dict[str, MethodLoc], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serial = {k: {"shard": v.shard, "idx": v.idx} for k, v in index.items()}
    path.write_text(json.dumps(serial, indent=2), encoding="utf-8")


def load_index(path: Path) -> Dict[str, MethodLoc]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: MethodLoc(shard=v["shard"], idx=int(v["idx"])) for k, v in raw.items()}


class ShardStore:
    """
    Lazy shard loader with in-memory caching of the last few shards.
    """
    def __init__(self, shards_dir: Path, max_cached: int = 4):
        self.shards_dir = shards_dir
        self.max_cached = max_cached
        self._cache: Dict[str, List[dict]] = {}
        self._lru: List[str] = []

    def _touch(self, name: str):
        if name in self._lru:
            self._lru.remove(name)
        self._lru.append(name)
        while len(self._lru) > self.max_cached:
            evict = self._lru.pop(0)
            self._cache.pop(evict, None)

    def load_shard(self, shard_name: str) -> List[dict]:
        """
        Loads shard into torch
        """
        if shard_name in self._cache:
            self._touch(shard_name)
            return self._cache[shard_name]

        path = self.shards_dir / shard_name
        data = torch.load(path, map_location="cpu")
        self._cache[shard_name] = data
        self._touch(shard_name)
        return data

    def get_method(self, loc: MethodLoc) -> dict:
        """
        Gets entire method
        """
        shard = self.load_shard(loc.shard)
        return shard[loc.idx]
