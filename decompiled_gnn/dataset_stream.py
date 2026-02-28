"""Pairwise dataset stream generation with deterministic train/val/test splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, Iterator, List, Literal, Tuple

from config import DatasetConfig

Pair = Tuple[str, str, int]
SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class IndexEntry:
    """Dataset index entry."""

    idx: str
    anchor: Path
    clones: List[Path]


class PairDatasetStream:
    """Create singleton pair streams for train/val/test from a fixed split."""

    def __init__(self, cfg: DatasetConfig, seed: int = 0):
        self.cfg = cfg
        self.seed = seed
        self.entries = self._scan_dataset(cfg)
        self.split_indices = self._create_splits()

    @staticmethod
    def _sorted_index_dirs(parent: Path) -> List[Path]:
        if not parent.exists():
            return []
        dirs = [d for d in parent.iterdir() if d.is_dir()]

        def key(item: Path) -> tuple[int, str]:
            try:
                return (0, f"{int(item.name):08d}")
            except ValueError:
                return (1, item.name)

        return sorted(dirs, key=key)

    @classmethod
    def _scan_dataset(cls, cfg: DatasetConfig) -> Dict[str, IndexEntry]:
        root = cfg.dataset_root.expanduser().resolve()
        base_dir = root / "base"

        out: Dict[str, IndexEntry] = {}
        idx_dirs = cls._sorted_index_dirs(base_dir)
        if cfg.limit_indices is not None:
            idx_dirs = idx_dirs[: cfg.limit_indices]

        for idx_dir in idx_dirs:
            idx = idx_dir.name
            anchor = idx_dir / cfg.anchor_filename
            if not anchor.exists():
                continue

            clone_files: List[Path] = []
            for clone_type in cfg.clone_types:
                clone_dir = root / clone_type / idx
                if not clone_dir.exists():
                    continue
                clone_files.extend(
                    p
                    for p in sorted(clone_dir.glob("*.java"))
                    if p.is_file() and p.name != cfg.anchor_filename
                )

            out[idx] = IndexEntry(idx=idx, anchor=anchor.resolve(), clones=clone_files)

        return out

    def _resolve_split_counts(self, total: int) -> tuple[int, int, int]:
        train_size = self.cfg.train_size
        val_size = self.cfg.val_size
        test_size = self.cfg.test_size

        float_mode = all(value <= 1.0 for value in (train_size, val_size, test_size))
        if float_mode:
            ratio_sum = train_size + val_size + test_size
            if ratio_sum <= 0:
                raise ValueError("train_size + val_size + test_size must be > 0")
            train_n = int(round(total * (train_size / ratio_sum)))
            val_n = int(round(total * (val_size / ratio_sum)))
            test_n = total - train_n - val_n
            return train_n, val_n, test_n

        train_n = int(train_size)
        val_n = int(val_size)
        test_n = int(test_size)
        if train_n + val_n + test_n > total:
            raise ValueError("Requested split counts exceed available indices")
        if train_n + val_n + test_n < total:
            train_n += total - (train_n + val_n + test_n)
        return train_n, val_n, test_n

    def _create_splits(self) -> dict[str, list[str]]:
        idxs = list(self.entries.keys())
        rng = random.Random(self.seed)
        rng.shuffle(idxs)

        train_n, val_n, test_n = self._resolve_split_counts(len(idxs))
        train = idxs[:train_n]
        val = idxs[train_n : train_n + val_n]
        test = idxs[train_n + val_n : train_n + val_n + test_n]

        return {"train": train, "val": val, "test": test}

    def split_summary(self) -> dict[str, int]:
        """Return number of indices in each split."""

        return {split: len(idxs) for split, idxs in self.split_indices.items()}

    def all_sources(self) -> list[Path]:
        """Return all unique source files referenced by this dataset."""

        uniq = {entry.anchor.resolve() for entry in self.entries.values()}
        for entry in self.entries.values():
            uniq.update(clone.resolve() for clone in entry.clones)
        return sorted(uniq)

    def _entries_for_split(self, split: SplitName) -> list[IndexEntry]:
        idx_set = set(self.split_indices[split])
        return [entry for entry in self.entries.values() if entry.idx in idx_set]

    def _positive_stream(self, split: SplitName, rng: random.Random) -> Iterator[Pair]:
        candidates = [entry for entry in self._entries_for_split(split) if entry.clones]
        if not candidates:
            raise RuntimeError(f"No positive pairs available for split '{split}'")

        while True:
            entry = rng.choice(candidates)
            clone_path = rng.choice(entry.clones)
            yield (str(entry.anchor), str(clone_path), 1)

    def _negative_stream(self, split: SplitName, rng: random.Random) -> Iterator[Pair]:
        entries = self._entries_for_split(split)
        if len(entries) < 2:
            raise RuntimeError(f"Need at least two indices to sample negatives for split '{split}'")

        by_idx = {entry.idx: entry for entry in entries}
        idxs = list(by_idx.keys())
        anchors = [entry for entry in entries if entry.anchor.exists()]

        if self.cfg.negative_pool == "base":
            pool = {idx: [by_idx[idx].anchor] for idx in idxs}
        else:
            pool = {
                idx: by_idx[idx].clones if by_idx[idx].clones else [by_idx[idx].anchor]
                for idx in idxs
            }

        while True:
            entry = rng.choice(anchors)
            other_idx = rng.choice(idxs)
            while other_idx == entry.idx:
                other_idx = rng.choice(idxs)
            negative = rng.choice(pool[other_idx])
            yield (str(entry.anchor), str(negative), 0)

    def stream(self, split: SplitName, infinite: bool = True, seed_offset: int = 0) -> Iterator[Pair]:
        """Yield singleton pair samples `(a, b, label)` for the requested split."""

        rng = random.Random(self.seed + seed_offset + {"train": 0, "val": 1000, "test": 2000}[split])
        pos_it = self._positive_stream(split, random.Random(rng.randint(0, 10**9)))
        neg_it = self._negative_stream(split, random.Random(rng.randint(0, 10**9)))

        emitted = 0
        while True:
            use_pos = rng.random() < self.cfg.positive_ratio
            pair = next(pos_it) if use_pos else next(neg_it)
            yield pair
            emitted += 1
            if not infinite and emitted >= len(self.split_indices[split]):
                break

    def sample_one(self, split: SplitName) -> Pair:
        """Return a single pair sample for convenience."""

        return next(self.stream(split=split, infinite=True))

    def pair_count_hint(self, split: SplitName) -> int:
        """Conservative size hint for one finite pass of the split."""

        return max(1, len(self.split_indices[split]))
