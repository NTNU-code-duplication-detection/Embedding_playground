"""
Config module for embeddings
"""
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class EmbedCacheConfig:
    """
    Config class contianing everything in a project
    """
    in_jsonl: Path               # ast_dataset/methods.jsonl
    out_dir: Path                # embed_cache output directory
    cache_dir: Path              # where per-node embeddings are cached
    model_name: str              # e.g. "microsoft/unixcoder-base" or "microsoft/graphcodebert-base"
    device: str                  # "cpu" or "mps" or "cuda"
    batch_size: int = 32
    max_length: int = 256
    shard_size: int = 2000       # number of methods per shard .pt
