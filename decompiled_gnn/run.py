"""Command-line entrypoint for the decompiled GNN pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from ast_graph import ASTGraphBuilder
from config import PipelineConfig, ensure_default_config, load_config
from model import ProgramCloneModel
from orchestrator import PipelineOrchestrator
from program_store import ProgramStore
from stream_factory import create_dataset_stream
from training import (
    collect_valid_pairs,
    cosine_eval_pairs,
    evaluate_pairs,
    mc_dropout_eval,
    save_training_artifacts,
    train_loop,
)


def _load_cfg(path: Path) -> PipelineConfig:
    path = path.expanduser().resolve()
    if path.exists():
        return load_config(path)
    return ensure_default_config(path)


def build_cache(cfg: PipelineConfig) -> None:
    stream = create_dataset_stream(cfg)
    orchestrator = PipelineOrchestrator(cfg)
    payload = orchestrator.prepare_from_dataset(stream)
    print(f"Cached programs: {len(payload['items'])}")
    print(f"Failures: {len(payload['failures'])}")
    print(f"Program index: {cfg.cache.program_index_path}")


def train(cfg: PipelineConfig, skip_cache: bool = False) -> None:
    stream = create_dataset_stream(cfg)

    if not skip_cache or not cfg.cache.program_index_path.exists():
        orchestrator = PipelineOrchestrator(cfg)
        orchestrator.prepare_from_dataset(stream)

    store = ProgramStore(cfg.cache.program_index_path)
    edge_type_to_id = ASTGraphBuilder.edge_type_to_id(cfg.ast.enabled_edge_types)

    model = ProgramCloneModel(
        in_dim=cfg.model.in_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        num_edge_types=len(edge_type_to_id),
        edge_type_dim=cfg.model.edge_type_dim,
        dropout=cfg.model.dropout,
    )

    train_iter = stream.stream("train", infinite=True)
    val_iter = stream.stream("val", infinite=True)
    test_iter = stream.stream("test", infinite=True)

    model, history = train_loop(
        model=model,
        store=store,
        train_iter=train_iter,
        val_iter=val_iter,
        device=cfg.general.device,
        model_cfg=cfg.model,
        train_cfg=cfg.training,
        seed=cfg.general.seed,
    )

    out = save_training_artifacts(model, history, out_dir=str(cfg.cache.root_dir / "trained_models"))
    print(f"Saved model: {out['model_path']}")
    print(f"Saved history: {out['history_path']}")

    test_pairs = collect_valid_pairs(test_iter, store, target_pairs=cfg.training.test_pairs)
    test_metrics = evaluate_pairs(
        model=model,
        store=store,
        pairs=test_pairs,
        device=cfg.general.device,
        model_cfg=cfg.model,
    )
    cosine_metrics = cosine_eval_pairs(
        model=model,
        store=store,
        pairs=test_pairs,
        device=cfg.general.device,
        threshold=0.0,
    )
    uncertainty = mc_dropout_eval(
        model=model,
        store=store,
        pairs=test_pairs,
        device=cfg.general.device,
        model_cfg=cfg.model,
        uncertainty_cfg=cfg.uncertainty,
    )

    print("Test metrics (classifier):", test_metrics)
    print("Test metrics (cosine):", cosine_metrics)
    print("Uncertainty:", uncertainty)


def main() -> int:
    parser = argparse.ArgumentParser("decompiled_gnn pipeline")
    parser.add_argument(
        "command",
        choices=["build-cache", "train"],
        help="Pipeline command",
    )
    parser.add_argument(
        "--config",
        default="config.default.json",
        help="Path to pipeline config JSON",
    )
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="For 'train': skip rebuilding cache if index exists",
    )
    args = parser.parse_args()

    cfg = _load_cfg(Path(args.config))

    if args.command == "build-cache":
        build_cache(cfg)
    elif args.command == "train":
        train(cfg, skip_cache=args.skip_cache)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
