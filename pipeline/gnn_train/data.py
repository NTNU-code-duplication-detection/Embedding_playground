"""
Module for data interface
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import torch

from gnn_train.shard_index import ShardStore, MethodLoc

Pair = Tuple[str, str, int]  # (method_id_a, method_id_b, label)


@dataclass
class PairBatch:
    """Tensor bundle for a pairwise training batch."""
    xa: torch.Tensor
    eia: torch.Tensor
    eta: torch.Tensor
    xb: torch.Tensor
    eib: torch.Tensor
    etb: torch.Tensor
    y: torch.Tensor


def default_pair_generator(method_ids: list[str], seed: int = 0) -> Iterator[Pair]:
    """
    Sanity-check generator: positives are (id,id), negatives random different.
    Replace with your generator later.
    """
    rng = random.Random(seed)
    while True:
        a = rng.choice(method_ids)
        if rng.random() < 0.5:
            yield (a, a, 1)
        else:
            b = rng.choice(method_ids)
            while b == a:
                b = rng.choice(method_ids)
            yield (a, b, 0)


def fetch_graph(store: ShardStore, index: dict[str, MethodLoc], method_id: str):
    """
    Fetches entire graph consisting of nodes
    """
    loc = index.get(method_id)
    if loc is None:
        return None
    item = store.get_method(loc)
    return item


def make_batch_from_pairs(
    pairs: list[Pair],
    store: ShardStore,
    index: dict[str, MethodLoc],
    device: str,
) -> Optional[PairBatch]:
    """
    Packs pairs into a batch.
    This version does NOT merge graphs into a single big graph (to keep code simple).
    It returns lists-of-graphs as padded is hard, so we process pair-by-pair in the trainer.
    For now we return None and let trainer do per-pair forward.
    """
    return None
