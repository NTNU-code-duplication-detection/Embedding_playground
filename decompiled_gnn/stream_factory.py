"""Factory to construct dataset streams from config."""

from __future__ import annotations

from bigclonebench_stream import BigCloneBenchPairStream
from config import PipelineConfig
from dataset_stream import PairDatasetStream


def create_dataset_stream(cfg: PipelineConfig):
    """Return the configured dataset stream implementation."""

    kind = cfg.dataset.dataset_kind
    if kind == "custom":
        return PairDatasetStream(cfg.dataset, seed=cfg.general.seed)
    if kind == "bigclonebench":
        return BigCloneBenchPairStream(
            cfg.dataset,
            seed=cfg.general.seed,
            cache_root=cfg.cache.root_dir,
        )
    raise ValueError(f"Unsupported dataset_kind={kind}")
