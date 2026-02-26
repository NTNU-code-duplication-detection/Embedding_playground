"""pipeline/pair_dataset_googlejam/generators.py
Supports Google Code Jam-style buckets under data/gcj_compiled/<bucket>/.../*.java"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Optional
import random

Pair = Tuple[str, str, int]


@dataclass(frozen=True)
class GoogleJamConfig:
    root: Path                  # gcj_compiled (expects root/<bucket 1..N>/**/**/**/<file>.java)
    seed: int = 0
    limit_buckets: Optional[int] = None      # e.g. 10
    max_files_per_bucket: Optional[int] = None  # e.g. 200
    glob: str = "*.java"  # recursive via rglob; .class files are ignored


def _list_buckets(cfg: GoogleJamConfig) -> List[Path]:
    root = cfg.root.expanduser().resolve()
    buckets = [p for p in root.iterdir() if p.is_dir()]

    # buckets are named "1", "2", ... so numeric sort is nice if possible
    def key(p: Path):
        try:
            return (0, int(p.name))
        except Exception:
            return (1, p.name)

    buckets.sort(key=key)

    if cfg.limit_buckets is not None:
        buckets = buckets[: cfg.limit_buckets]
    return buckets


def _index_files(cfg: GoogleJamConfig) -> Dict[str, List[Path]]:
    rng = random.Random(cfg.seed)
    buckets = _list_buckets(cfg)

    out: Dict[str, List[Path]] = {}
    for b in buckets:
        # New dataset layout is nested (typically ~3 dirs deep). Use recursive search.
        files = sorted(p for p in b.rglob(cfg.glob) if p.is_file())
        # Safety: keep only real Java source files (ignore hidden/temp files)
        files = [p for p in files if p.suffix.lower() == ".java" and not p.name.startswith(".")]
        # optional downsample per bucket to cap cost
        if cfg.max_files_per_bucket is not None and len(files) > cfg.max_files_per_bucket:
            files = rng.sample(files, cfg.max_files_per_bucket)
            files.sort()
        if len(files) >= 2:
            out[b.name] = files
    return out


def positive_pairs(cfg: GoogleJamConfig, *, infinite: bool = True) -> Iterator[Pair]:
    """
    Yields (a, b, 1) where a and b are different solutions from same bucket.
    """
    rng = random.Random(cfg.seed)
    by_bucket = _index_files(cfg)
    bucket_ids = list(by_bucket.keys())
    if not bucket_ids:
        raise ValueError("No buckets with >=2 java files found")

    while True:
        bid = rng.choice(bucket_ids)
        files = by_bucket[bid]
        a, b = rng.sample(files, 2)
        yield (str(a), str(b), 1)
        if not infinite:
            return


def negative_pairs(cfg: GoogleJamConfig, *, infinite: bool = True) -> Iterator[Pair]:
    """
    Yields (a, b, 0) where a is from bucket i and b is from bucket j != i.
    """
    rng = random.Random(cfg.seed)
    by_bucket = _index_files(cfg)
    bucket_ids = list(by_bucket.keys())
    if len(bucket_ids) < 2:
        raise ValueError("Need >=2 buckets with >=2 java files to make negatives")

    while True:
        bi, bj = rng.sample(bucket_ids, 2)
        a = rng.choice(by_bucket[bi])
        b = rng.choice(by_bucket[bj])
        yield (str(a), str(b), 0)
        if not infinite:
            return


def interleave(pos_iter: Iterator[Pair], neg_iter: Iterator[Pair], *, pos_ratio: float = 0.5, seed: int = 0) -> Iterator[Pair]:
    """
    Mix positives and negatives at requested ratio.
    """
    rng = random.Random(seed)
    while True:
        if rng.random() < pos_ratio:
            yield next(pos_iter)
        else:
            yield next(neg_iter)
