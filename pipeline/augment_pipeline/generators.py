"""Module pipeline/augment_pipeline/generators.py."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from .augmenters import apply_augmentations
from .cache import write_augmented
from .config import AugmentConfig
from .discovery import discover_java_files_by_bucket


Pair = Tuple[str, str, int]


def _flatten(files_by_bucket: Dict[str, List[Path]]) -> List[Path]:
    out: List[Path] = []
    for _, fs in files_by_bucket.items():
        out.extend(fs)
    return out


def _pick_two_distinct(rng: random.Random, items: List[Path]) -> Tuple[Path, Path]:
    a = rng.choice(items)
    b = rng.choice(items)
    while b == a:
        b = rng.choice(items)
    return a, b


def build_positive_pair(cfg: AugmentConfig, rng: random.Random) -> Pair:
    """
    Returns (orig_path, augmented_path, 1).
    """
    files_by_bucket = discover_java_files_by_bucket(
        root=cfg.root,
        glob=cfg.glob,
        limit_buckets=cfg.limit_buckets,
        max_files_per_bucket=cfg.max_files_per_bucket,
    )
    all_files = _flatten(files_by_bucket)
    if not all_files:
        raise RuntimeError(f"No java files found under {cfg.root}")

    src = rng.choice(all_files)
    raw = src.read_bytes()

    # aug_key controls cache identity; include seed and a random nonce for diversity
    nonce = rng.randint(0, 1_000_000_000)
    aug_key = f"seed={cfg.seed}|nonce={nonce}|rename={cfg.rename_identifiers}|ws={cfg.whitespace_noise}"

    aug_bytes, _stats = apply_augmentations(
        raw,
        rng=rng,
        do_rename=cfg.rename_identifiers,
        rename_prob=cfg.rename_prob,
        do_ws=cfg.whitespace_noise,
        ws_prob=cfg.whitespace_prob,
    )

    dst = write_augmented(
        out_dir=cfg.out_dir,
        source_path=src,
        aug_key=aug_key,
        content=aug_bytes,
        cache=cfg.cache_augmented,
    )

    return (str(src.resolve()), str(dst.resolve()), 1)


def build_negative_pair(cfg: AugmentConfig, rng: random.Random) -> Pair:
    """
    Returns (path_a, path_b, 0) where a != b.
    """
    files_by_bucket = discover_java_files_by_bucket(
        root=cfg.root,
        glob=cfg.glob,
        limit_buckets=cfg.limit_buckets,
        max_files_per_bucket=cfg.max_files_per_bucket,
    )
    all_files = _flatten(files_by_bucket)
    if len(all_files) < 2:
        raise RuntimeError("Need at least 2 java files for negatives")

    a, b = _pick_two_distinct(rng, all_files)
    return (str(a.resolve()), str(b.resolve()), 0)


def iter_pairs(
    cfg: AugmentConfig,
    *,
    infinite: bool = True,
    pos_ratio: float = 0.5,
    seed_offset: int = 0,
) -> Iterator[Pair]:
    """
    Mix positives (self-aug) and negatives (random other program).
    """
    rng = random.Random(cfg.seed + seed_offset)
    while True:
        if rng.random() < pos_ratio:
            yield build_positive_pair(cfg, rng)
        else:
            yield build_negative_pair(cfg, rng)
        if not infinite:
            return
