"""
Module for generator
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from pair_dataset.config import PairDatasetConfig
from pair_dataset.scan import scan_dataset


Pair = Tuple[str, str, int]  # (path_a, path_b, label)


def positive_pairs(
    cfg: PairDatasetConfig,
    *,
    shuffle_indices: bool = True,
    infinite: bool = True,
) -> Iterator[Pair]:
    """
    Yields (anchor_path, clone_path, 1) for clones in cfg.clone_type folder
    with the same index as the anchor.

    If infinite=True, cycles forever.
    """
    entries = scan_dataset(cfg)
    indices = [k for k, e in entries.items() if len(e.clones) > 0]

    rng = random.Random(cfg.seed)
    if shuffle_indices:
        rng.shuffle(indices)

    def one_epoch() -> List[Pair]:
        pairs: List[Pair] = []
        for idx in indices:
            e = entries[idx]
            for c in e.clones:
                pairs.append((str(e.anchor), str(c), 1))
        if shuffle_indices:
            rng.shuffle(pairs)
        return pairs

    if not infinite:
        yield from one_epoch()
        return

    while True:
        yield from one_epoch()


def negative_pairs(
    cfg: PairDatasetConfig,
    *,
    shuffle_indices: bool = True,
    infinite: bool = True,
) -> Iterator[Pair]:
    """
    Yields (anchor_path, nonclone_path, 0) where nonclone is sampled from a DIFFERENT index.

    negative_pool:
      - "same_clone_type": sample from cfg.root/<clone_type>/<other_idx>/*.java
      - "base": sample from cfg.root/base/<other_idx>/<anchor_filename> (other anchors)
    """
    entries = scan_dataset(cfg)

    anchors = [e for e in entries.values() if e.anchor.exists()]
    if len(anchors) < 2:
        return

    # Build candidate pools by idx
    clone_candidates_by_idx: Dict[str, List[Path]] = {}
    for idx, e in entries.items():
        if cfg.negative_pool == "base":
            clone_candidates_by_idx[idx] = [e.anchor]
        else:
            # same_clone_type: use that index's clone files; if empty, fallback to its anchor
            if e.clones:
                clone_candidates_by_idx[idx] = list(e.clones)
            else:
                clone_candidates_by_idx[idx] = [e.anchor]

    idx_list = list(entries.keys())
    rng = random.Random(cfg.seed)

    if shuffle_indices:
        rng.shuffle(idx_list)

    # Stream
    def sample_nonclone_idx(not_idx: str) -> str:
        # ensure different index
        j = rng.choice(idx_list)
        while j == not_idx:
            j = rng.choice(idx_list)
        return j

    if not infinite:
        # one pass: one negative per anchor by default
        for e in anchors:
            j = sample_nonclone_idx(e.idx)
            candidate = rng.choice(clone_candidates_by_idx[j])
            yield (str(e.anchor), str(candidate), 0)
        return

    while True:
        # continuously sample negatives
        e = rng.choice(anchors)
        j = sample_nonclone_idx(e.idx)
        candidate = rng.choice(clone_candidates_by_idx[j])
        yield (str(e.anchor), str(candidate), 0)
