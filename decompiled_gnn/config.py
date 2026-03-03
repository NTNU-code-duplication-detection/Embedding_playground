"""Centralized configuration for the decompiled GNN pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import json
from pathlib import Path
from typing import Any, Literal, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class GeneralConfig:
    """Global runtime settings."""

    seed: int = 42
    device: str = "cpu"


@dataclass
class DatasetConfig:
    """Dataset scanning, pairing, and split settings."""

    dataset_kind: Literal["custom", "bigclonebench"] = "custom"
    dataset_root: Path = REPO_ROOT / "data" / "code-clone-dataset" / "dataset"
    clone_types: list[str] = field(default_factory=lambda: ["type-3", "type-4"])
    anchor_filename: str = "main.java"
    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15
    limit_indices: Optional[int] = None
    negative_pool: Literal["base", "same_clone_type"] = "same_clone_type"
    positive_ratio: float = 0.5

    # BigCloneBench-specific settings (used when dataset_kind == "bigclonebench")
    bigclonebench_root: Path = Path("~/datasets/bigclonebench")
    bigclonebench_labels_path: Path = Path("~/datasets/bigclonebench/clone_labels_typed.txt")
    bigclonebench_db_path: Path = Path("~/datasets/bigclonebench/bigclonebenchdb/bcb.h2.db")
    bigclonebench_h2_jar: Path = Path("~/datasets/bigclonebench/libs/h2-1.3.176.jar")
    bigclonebench_source_root: Optional[Path] = Path("~/datasets/bigclonebench/ijadataset/bcb_reduced")
    bigclonebench_limit_pairs: Optional[int] = 200000
    bigclonebench_db_batch_size: int = 500


@dataclass
class CompilationConfig:
    """Optional compile + decompile stage."""

    enabled: bool = True
    java_home: Optional[Path] = None
    vineflower_jar: Path = REPO_ROOT / "pipeline" / "decompiler" / "vineflower-1.11.2.jar"
    prefer_jars: bool = True
    force_rebuild: bool = False
    compile_timeout_sec: int = 600
    decompile_timeout_sec: int = 600


@dataclass
class ASTConfig:
    """AST graph extraction settings."""

    enabled_edge_types: list[str] = field(
        default_factory=lambda: ["AST", "SEQ", "IF_THEN", "IF_ELSE", "DATA_FLOW"]
    )
    undirected_edges: bool = True
    seq_top_level_only: bool = True
    include_file_fallback_method: bool = True


@dataclass
class EmbeddingConfig:
    """Node embedding model and batching settings."""

    model_name: str = "microsoft/graphcodebert-base"
    max_length: int = 256
    batch_size: int = 32
    embedding_device: Optional[str] = None


@dataclass
class CacheConfig:
    """Persistent cache locations."""

    root_dir: Path = REPO_ROOT / "decompiled_gnn" / "cache"
    program_index_path: Path = REPO_ROOT / "decompiled_gnn" / "cache" / "program_index.json"


@dataclass
class ModelConfig:
    """Model architecture and objective configuration."""

    in_dim: int = 768
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    edge_type_dim: int = 32
    loss_type: Literal["bce", "contrastive"] = "bce"
    contrastive_margin: float = 0.5
    contrastive_head_weight: float = 0.0


@dataclass
class TrainingConfig:
    """Training loop hyperparameters."""

    steps: int = 2000
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    log_every: int = 50
    eval_every: int = 200
    val_pairs: int = 200
    test_pairs: int = 1000
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    early_stopping_enabled: bool = True
    early_stopping_metric: str = "f1"
    early_stopping_patience_steps: int = 200
    early_stopping_min_delta: float = 0.005
    early_stopping_restore_best: bool = True


@dataclass
class UncertaintyConfig:
    """MC-dropout uncertainty settings."""

    enabled: bool = True
    mc_dropout_passes: int = 20


@dataclass
class PlotConfig:
    """Plot persistence settings."""

    output_dir: Path = REPO_ROOT / "decompiled_gnn" / "plots"
    save_figures: bool = False


@dataclass
class PipelineConfig:
    """Top-level configuration object used across all modules."""

    general: GeneralConfig = field(default_factory=GeneralConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    compilation: CompilationConfig = field(default_factory=CompilationConfig)
    ast: ASTConfig = field(default_factory=ASTConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)


def default_pipeline_config() -> PipelineConfig:
    """Return a fresh default config instance."""

    return PipelineConfig()


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def to_dict(cfg: PipelineConfig) -> dict[str, Any]:
    """Serialize config dataclasses to a JSON-friendly dict."""

    return _jsonable(cfg)


def save_config(cfg: PipelineConfig, path: Path) -> None:
    """Persist config as formatted JSON."""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(cfg), indent=2), encoding="utf-8")


def _path_or_none(value: Any) -> Optional[Path]:
    if value in (None, ""):
        return None
    return Path(value).expanduser().resolve()


def from_dict(data: dict[str, Any]) -> PipelineConfig:
    """Construct PipelineConfig from a loaded JSON dict."""

    general_data = data.get("general", {})
    dataset_data = data.get("dataset", {})
    compilation_data = data.get("compilation", {})
    ast_data = data.get("ast", {})
    embedding_data = data.get("embedding", {})
    cache_data = data.get("cache", {})
    model_data = data.get("model", {})
    training_data = data.get("training", {})
    uncertainty_data = data.get("uncertainty", {})
    plot_data = data.get("plot", {})

    general = GeneralConfig(**general_data)

    default_dataset = DatasetConfig()
    dataset = DatasetConfig(
        dataset_kind=dataset_data.get("dataset_kind", default_dataset.dataset_kind),
        dataset_root=Path(dataset_data.get("dataset_root", default_dataset.dataset_root)).expanduser().resolve(),
        clone_types=list(dataset_data.get("clone_types", default_dataset.clone_types)),
        anchor_filename=dataset_data.get("anchor_filename", default_dataset.anchor_filename),
        train_size=float(dataset_data.get("train_size", default_dataset.train_size)),
        val_size=float(dataset_data.get("val_size", default_dataset.val_size)),
        test_size=float(dataset_data.get("test_size", default_dataset.test_size)),
        limit_indices=dataset_data.get("limit_indices", default_dataset.limit_indices),
        negative_pool=dataset_data.get("negative_pool", default_dataset.negative_pool),
        positive_ratio=float(dataset_data.get("positive_ratio", default_dataset.positive_ratio)),
        bigclonebench_root=Path(
            dataset_data.get("bigclonebench_root", default_dataset.bigclonebench_root)
        ).expanduser().resolve(),
        bigclonebench_labels_path=Path(
            dataset_data.get("bigclonebench_labels_path", default_dataset.bigclonebench_labels_path)
        ).expanduser().resolve(),
        bigclonebench_db_path=Path(
            dataset_data.get("bigclonebench_db_path", default_dataset.bigclonebench_db_path)
        ).expanduser().resolve(),
        bigclonebench_h2_jar=Path(
            dataset_data.get("bigclonebench_h2_jar", default_dataset.bigclonebench_h2_jar)
        ).expanduser().resolve(),
        bigclonebench_source_root=_path_or_none(
            dataset_data.get("bigclonebench_source_root", default_dataset.bigclonebench_source_root)
        ),
        bigclonebench_limit_pairs=dataset_data.get(
            "bigclonebench_limit_pairs", default_dataset.bigclonebench_limit_pairs
        ),
        bigclonebench_db_batch_size=int(
            dataset_data.get("bigclonebench_db_batch_size", default_dataset.bigclonebench_db_batch_size)
        ),
    )

    default_comp = CompilationConfig()
    compilation = CompilationConfig(
        enabled=bool(compilation_data.get("enabled", default_comp.enabled)),
        java_home=_path_or_none(compilation_data.get("java_home", default_comp.java_home)),
        vineflower_jar=Path(compilation_data.get("vineflower_jar", default_comp.vineflower_jar)).expanduser().resolve(),
        prefer_jars=bool(compilation_data.get("prefer_jars", default_comp.prefer_jars)),
        force_rebuild=bool(compilation_data.get("force_rebuild", default_comp.force_rebuild)),
        compile_timeout_sec=int(compilation_data.get("compile_timeout_sec", default_comp.compile_timeout_sec)),
        decompile_timeout_sec=int(compilation_data.get("decompile_timeout_sec", default_comp.decompile_timeout_sec)),
    )

    default_ast = ASTConfig()
    ast_cfg = ASTConfig(
        enabled_edge_types=list(ast_data.get("enabled_edge_types", default_ast.enabled_edge_types)),
        undirected_edges=bool(ast_data.get("undirected_edges", default_ast.undirected_edges)),
        seq_top_level_only=bool(ast_data.get("seq_top_level_only", default_ast.seq_top_level_only)),
        include_file_fallback_method=bool(
            ast_data.get("include_file_fallback_method", default_ast.include_file_fallback_method)
        ),
    )

    default_embed = EmbeddingConfig()
    embedding = EmbeddingConfig(
        model_name=embedding_data.get("model_name", default_embed.model_name),
        max_length=int(embedding_data.get("max_length", default_embed.max_length)),
        batch_size=int(embedding_data.get("batch_size", default_embed.batch_size)),
        embedding_device=embedding_data.get("embedding_device", default_embed.embedding_device),
    )

    default_cache = CacheConfig()
    cache = CacheConfig(
        root_dir=Path(cache_data.get("root_dir", default_cache.root_dir)).expanduser().resolve(),
        program_index_path=Path(
            cache_data.get("program_index_path", default_cache.program_index_path)
        ).expanduser().resolve(),
    )

    model = ModelConfig(**{**asdict(ModelConfig()), **model_data})
    training = TrainingConfig(**{**asdict(TrainingConfig()), **training_data})
    uncertainty = UncertaintyConfig(**{**asdict(UncertaintyConfig()), **uncertainty_data})

    default_plot = PlotConfig()
    plot = PlotConfig(
        output_dir=Path(plot_data.get("output_dir", default_plot.output_dir)).expanduser().resolve(),
        save_figures=bool(plot_data.get("save_figures", default_plot.save_figures)),
    )

    return PipelineConfig(
        general=general,
        dataset=dataset,
        compilation=compilation,
        ast=ast_cfg,
        embedding=embedding,
        cache=cache,
        model=model,
        training=training,
        uncertainty=uncertainty,
        plot=plot,
    )


def load_config(path: Path) -> PipelineConfig:
    """Load config JSON from disk."""

    path = path.expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    return from_dict(data)


def ensure_default_config(path: Path) -> PipelineConfig:
    """Create config file if missing and always return loaded config."""

    path = path.expanduser().resolve()
    if not path.exists():
        cfg = default_pipeline_config()
        save_config(cfg, path)
        return cfg
    return load_config(path)
