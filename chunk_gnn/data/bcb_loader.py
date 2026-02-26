"""
BigCloneBench data loader.

Reads the same raw data files that MagNET uses (clone_labels.txt + individual
.java files), but processes them through our own chunk-level GNN pipeline
instead of MagNET's token-level graph construction.

Using the same data and splits ensures fair comparison with MagNET's results.

Format of clone_labels.txt (no header, comma-delimited):
    col 0: code_file_1       (numeric ID, e.g. "10005623")
    col 1: code_file_2       (numeric ID)
    col 2: clone_label        (0=non-clone, 1=clone)
    col 3: split_label        (0=train, 1=test, 2=val)
    col 4: dataset_label      (0=BCB, 1=GCJ)
    col 5: clone_type         (T1, T2, ST3, MT3, VST3, T4, Non_Clone)
    col 6: similarity_score   (float)
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# Files that MagNET hardcodes as forbidden (javalang parse failures).
# We skip the same ones for a fair comparison.
# Source: Multigraph_match_optimized/data/sast/java_2/ast_api.py:11
FORBIDDEN_FILES = {"37044", "4892654", "6966398", "7550876"}

SPLIT_MAP = {0: "train", 1: "test", 2: "val"}
CLONE_TYPES = {"T1", "T2", "ST3", "MT3", "VST3", "T4", "Non_Clone"}


@dataclass
class BCBPair:
    """A single pair from clone_labels.txt."""

    id1: str
    id2: str
    clone_label: int  # 0=non-clone, 1=clone
    split: str  # "train", "test", or "val"
    clone_type: str  # T1, T2, ST3, MT3, VST3, T4, Non_Clone
    similarity_score: float


@dataclass
class BCBStats:
    """Statistics about the loaded BCB dataset."""

    total_pairs: int = 0
    pairs_by_split: dict[str, int] = field(default_factory=dict)
    pairs_by_type: dict[str, int] = field(default_factory=dict)
    clone_pairs: int = 0
    non_clone_pairs: int = 0
    unique_function_ids: int = 0
    missing_files: int = 0
    forbidden_filtered: int = 0
    skipped_non_bcb: int = 0


class BCBLoader:
    """Loads BigCloneBench clone_labels.txt and individual .java files.

    Uses the same data format and filtering as MagNET for fair comparison.
    """

    def __init__(
        self,
        bcb_root: str,
        config: dict | None = None,
        labels_file: str | None = None,
    ):
        """
        Args:
            bcb_root: Path to the BCB data directory containing
                      clone_labels.txt and dataset_files/
            config: Optional config dict with 'skip_functions' list
            labels_file: Override the labels filename (default: "clone_labels.txt").
                         Use "clone_labels_typed.txt" for granular per-type labels.
        """
        self.bcb_root = Path(bcb_root)
        self.labels_path = self.bcb_root / (labels_file or "clone_labels.txt")
        self.files_dir = self.bcb_root / "dataset_files"
        self.config = config or {}

        # Combine hardcoded forbidden files with any config-specified ones
        extra_skip = set(self.config.get("skip_functions", []))
        self.skip_ids = FORBIDDEN_FILES | extra_skip

        self._pairs: list[BCBPair] = []
        self._unique_ids: set[str] = set()
        self._loaded = False

    def load_labels(self) -> list[BCBPair]:
        """Parse clone_labels.txt and return all BCB pairs.

        Filters:
          - Only dataset_label == 0 (BCB, not GCJ)
          - Skip pairs containing forbidden file IDs
        """
        if self._loaded:
            return self._pairs

        if not self.labels_path.exists():
            raise FileNotFoundError(
                f"clone_labels.txt not found at {self.labels_path}"
            )

        pairs = []
        stats = BCBStats()

        with open(self.labels_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row_num, row in enumerate(reader):
                if len(row) != 7:
                    log.warning(
                        "Skipping malformed row %d: expected 7 cols, got %d",
                        row_num,
                        len(row),
                    )
                    continue

                id1, id2 = row[0].strip(), row[1].strip()
                clone_label = int(row[2])
                split_label = int(row[3])
                dataset_label = int(row[4])
                clone_type = row[5].strip()
                similarity_score = float(row[6])

                # Filter: BCB only (not GCJ)
                if dataset_label != 0:
                    stats.skipped_non_bcb += 1
                    continue

                # Filter: skip forbidden files
                if id1 in self.skip_ids or id2 in self.skip_ids:
                    stats.forbidden_filtered += 1
                    continue

                split = SPLIT_MAP.get(split_label)
                if split is None:
                    log.warning(
                        "Unknown split_label %d at row %d, skipping",
                        split_label,
                        row_num,
                    )
                    continue

                pair = BCBPair(
                    id1=id1,
                    id2=id2,
                    clone_label=clone_label,
                    split=split,
                    clone_type=clone_type,
                    similarity_score=similarity_score,
                )
                pairs.append(pair)
                self._unique_ids.add(id1)
                self._unique_ids.add(id2)

        self._pairs = pairs
        self._loaded = True

        log.info(
            "Loaded %d BCB pairs (%d unique functions, %d forbidden-filtered, "
            "%d non-BCB skipped)",
            len(pairs),
            len(self._unique_ids),
            stats.forbidden_filtered,
            stats.skipped_non_bcb,
        )
        return pairs

    def get_unique_function_ids(self) -> set[str]:
        """Return set of all unique function IDs referenced in BCB pairs."""
        if not self._loaded:
            self.load_labels()
        return self._unique_ids.copy()

    def load_function_source(self, func_id: str) -> str | None:
        """Read a single .java file and return its source code.

        Returns None if the file doesn't exist or can't be read.
        """
        file_path = self.files_dir / f"{func_id}.java"
        try:
            return file_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except UnicodeDecodeError:
            # Some BCB files may have non-UTF8 encoding
            try:
                return file_path.read_text(encoding="latin-1")
            except (UnicodeDecodeError, OSError):
                log.warning("Failed to read %s with any encoding", file_path)
                return None

    def get_split(self, split: str) -> list[BCBPair]:
        """Return all pairs for a given split ('train', 'test', or 'val')."""
        if not self._loaded:
            self.load_labels()
        return [p for p in self._pairs if p.split == split]

    def get_stats(self) -> BCBStats:
        """Compute and return dataset statistics."""
        if not self._loaded:
            self.load_labels()

        stats = BCBStats(
            total_pairs=len(self._pairs),
            unique_function_ids=len(self._unique_ids),
        )

        for pair in self._pairs:
            # By split
            stats.pairs_by_split[pair.split] = (
                stats.pairs_by_split.get(pair.split, 0) + 1
            )
            # By clone type
            stats.pairs_by_type[pair.clone_type] = (
                stats.pairs_by_type.get(pair.clone_type, 0) + 1
            )
            # Clone vs non-clone
            if pair.clone_label == 1:
                stats.clone_pairs += 1
            else:
                stats.non_clone_pairs += 1

        # Check for missing files
        missing = 0
        for func_id in self._unique_ids:
            file_path = self.files_dir / f"{func_id}.java"
            if not file_path.exists():
                missing += 1
        stats.missing_files = missing

        return stats

    def check_file_availability(self) -> tuple[set[str], set[str]]:
        """Check which function files exist and which are missing.

        Returns:
            (available_ids, missing_ids)
        """
        if not self._loaded:
            self.load_labels()

        available = set()
        missing = set()
        for func_id in self._unique_ids:
            file_path = self.files_dir / f"{func_id}.java"
            if file_path.exists():
                available.add(func_id)
            else:
                missing.add(func_id)

        return available, missing
