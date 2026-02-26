"""
Config module
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

CloneType = Literal["type-1", "type-2", "type-3"]

@dataclass(frozen=True)
class ProgramProjectConfig:
    """
    Config
    """
    dataset_root: Path                 # .../data/code-clone-dataset/dataset
    clone_type: CloneType = "type-3"
    out_dir: Path = Path("./program_artifacts")

    jdk_home: Path = Path("/Library/Java/JavaVirtualMachines/temurin-21.jdk/Contents/Home")
    vineflower_jar: Path = Path("./vineflower.jar")

    # embedding
    model_name: str = "microsoft/graphcodebert-base"
    device: str = "cpu"
    batch_size: int = 32
    max_length: int = 256
    shard_size: int = 2000

    # caching
    node_cache_dir: Path = Path("./node_cache")  # shared cache across programs

    # behavior
    keep_workdirs: bool = True   # keep program work folders for debugging
    force_rebuild: bool = False  # rebuild even if artifacts already exist
    limit_indices: Optional[int] = None
