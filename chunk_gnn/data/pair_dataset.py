"""
Pair dataset for siamese training on BigCloneBench.

Loads pairs of pre-computed chunk graphs from the cache directory and
provides them as (graph1, graph2, label, clone_type) tuples for training.

Uses torch_geometric's Batch.from_data_list() for efficient batched
GNN processing — this is the key difference from MagNET's slow
per-sample iteration.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import torch

# Suppress PyTorch >=2.6 FutureWarning about weights_only (fires per torch.load call)
warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

from chunk_gnn.data.bcb_loader import BCBPair

log = logging.getLogger(__name__)


class BCBPairDataset(Dataset):
    """Loads pairs of pre-computed graph objects for siamese training.

    Each sample is a (graph1, graph2, label, clone_type) tuple where
    the graphs are torch_geometric Data objects loaded from .pt files.
    """

    def __init__(
        self,
        pairs: list[BCBPair],
        cache_dir: str,
        label_positive: float = 1.0,
        label_negative: float = -1.0,
    ):
        """
        Args:
            pairs: List of BCBPair objects (from BCBLoader.get_split())
            cache_dir: Directory containing {func_id}.pt files
            label_positive: Label for clone pairs (default: +1.0)
            label_negative: Label for non-clone pairs (default: -1.0)
        """
        self.cache_dir = Path(cache_dir)
        self.label_positive = label_positive
        self.label_negative = label_negative

        # Filter out pairs where either graph is missing from cache
        self.pairs = self._filter_available_pairs(pairs)
        log.info(
            "BCBPairDataset: %d pairs (filtered from %d, cache_dir=%s)",
            len(self.pairs), len(pairs), cache_dir,
        )

    def _filter_available_pairs(
        self, pairs: list[BCBPair]
    ) -> list[BCBPair]:
        """Keep only pairs where both graphs exist in cache."""
        # Build set of available IDs (scan cache once)
        available = set()
        for pt_file in self.cache_dir.glob("*.pt"):
            available.add(pt_file.stem)

        filtered = []
        skipped = 0
        for pair in pairs:
            if pair.id1 in available and pair.id2 in available:
                filtered.append(pair)
            else:
                skipped += 1

        if skipped > 0:
            log.warning(
                "Filtered out %d pairs (missing graphs in cache)", skipped
            )
        return filtered

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[Data, Data, float, str]:
        pair = self.pairs[idx]

        graph1 = self._load_graph(pair.id1)
        graph2 = self._load_graph(pair.id2)

        label = (
            self.label_positive if pair.clone_label == 1
            else self.label_negative
        )

        return graph1, graph2, label, pair.clone_type

    def _load_graph(self, func_id: str) -> Data:
        """Load a pre-computed graph from cache and upcast to float32."""
        path = self.cache_dir / f"{func_id}.pt"
        data = torch.load(path, weights_only=False, map_location="cpu")

        # Upcast from float16 (storage) to float32 (training)
        if data.x.dtype == torch.float16:
            data.x = data.x.float()

        return data


def collate_pairs(
    batch: list[tuple[Data, Data, float, str]],
) -> tuple[Batch, Batch, torch.Tensor, list[str]]:
    """Custom collate function for batching pairs of variable-size graphs.

    Uses torch_geometric's Batch.from_data_list() which:
    - Concatenates node features from all graphs into one big tensor
    - Adjusts edge_index offsets so edges point to correct nodes
    - Creates a `batch` vector mapping each node to its graph index

    This enables processing all graphs in the batch with a SINGLE
    GCNConv call instead of looping per sample (MagNET's bottleneck).
    """
    graphs1, graphs2, labels, clone_types = zip(*batch)

    batch1 = Batch.from_data_list(list(graphs1))
    batch2 = Batch.from_data_list(list(graphs2))
    labels = torch.tensor(labels, dtype=torch.float)

    return batch1, batch2, labels, list(clone_types)


def create_dataloaders(
    train_pairs: list[BCBPair],
    val_pairs: list[BCBPair],
    test_pairs: list[BCBPair],
    cache_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    label_positive: float = 1.0,
    label_negative: float = -1.0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders with proper batching.

    Args:
        train_pairs, val_pairs, test_pairs: From BCBLoader.get_split()
        cache_dir: Directory with pre-computed .pt graphs
        batch_size: Batch size for training
        num_workers: Parallel data loading workers (uses multiple CPU cores)
        label_positive: Label for clones (default: +1.0)
        label_negative: Label for non-clones (default: -1.0)

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset = BCBPairDataset(
        train_pairs, cache_dir, label_positive, label_negative,
    )
    val_dataset = BCBPairDataset(
        val_pairs, cache_dir, label_positive, label_negative,
    )
    test_dataset = BCBPairDataset(
        test_pairs, cache_dir, label_positive, label_negative,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_pairs,
        pin_memory=True,       # Faster CPU->GPU transfer
        drop_last=False,
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_pairs,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_pairs,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    log.info(
        "DataLoaders created: train=%d batches, val=%d batches, test=%d batches "
        "(batch_size=%d, workers=%d)",
        len(train_loader), len(val_loader), len(test_loader),
        batch_size, num_workers,
    )

    return train_loader, val_loader, test_loader
