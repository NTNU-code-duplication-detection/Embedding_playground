"""
Config module for gnn train
"""
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class TrainConfig:
    """
    Config class
    """
    shards_dir: Path                 # embed_cache/shards
    index_path: Path                 # gnn_train/index.json
    device: str = "cpu"              # cpu|mps|cuda

    # model
    in_dim: int = 768
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1

    # training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_pairs: int = 64
    steps: int = 2000
    log_every: int = 50
    seed: int = 0
