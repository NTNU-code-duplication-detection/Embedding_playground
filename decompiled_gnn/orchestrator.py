"""End-to-end orchestration for cache building and program index generation."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import torch

from artifact_pipeline import CompileDecompilePipeline
from ast_graph import ASTGraphBuilder
from cache import CacheManager
from config import PipelineConfig
from embeddings import NodeEmbedder


class PipelineOrchestrator:
    """Compose compile/decompile, AST extraction, and embedding stages."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.cache = CacheManager(cfg.cache.root_dir)
        self.graph_builder = ASTGraphBuilder(cfg.ast)
        edge_type_to_id = self.graph_builder.edge_type_to_id(cfg.ast.enabled_edge_types)
        self.embedder = NodeEmbedder(
            cfg=cfg.embedding,
            cache=self.cache,
            edge_type_to_id=edge_type_to_id,
            device=cfg.general.device,
            fallback_dim=cfg.model.in_dim,
        )
        self.compile_pipeline = CompileDecompilePipeline(cfg.compilation, self.cache)

    def prepare_program(self, source_path: Path) -> dict[str, Any]:
        """Build all cached artifacts for one source path."""

        source_path = source_path.expanduser().resolve()
        program_dir = self.cache.ensure_program_dirs(source_path)

        source_snapshot = None
        snapshot_dir = program_dir / "source_snapshot"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        if source_path.is_file():
            snapshot_file = snapshot_dir / source_path.name
            if not snapshot_file.exists() or self.cfg.compilation.force_rebuild:
                shutil.copy2(source_path, snapshot_file)
            source_snapshot = str(snapshot_file.resolve())
        else:
            source_snapshot = str(source_path)

        artifact_manifest = self.compile_pipeline.process(source_path)
        graph_input_path = Path(artifact_manifest.get("graph_input_path", source_path)).expanduser().resolve()

        graph_jsonl = self.cache.graph_jsonl_path(source_path)
        if graph_jsonl.exists() and not self.cfg.compilation.force_rebuild:
            graph_records = self.cache.read_jsonl(graph_jsonl)
        else:
            graph_records = self.graph_builder.build_records_from_path(graph_input_path)
            self.cache.write_jsonl(graph_jsonl, graph_records)

        embedding_pt = self.cache.embedding_pt_path(source_path)
        if embedding_pt.exists() and not self.cfg.compilation.force_rebuild:
            tensor_records = torch.load(embedding_pt, map_location="cpu")
        else:
            tensor_records = self.embedder.embed_graph_records(graph_records=graph_records, out_pt=embedding_pt)

        metadata = {
            "source_path": str(source_path),
            "source_snapshot": source_snapshot,
            "program_dir": str(program_dir),
            "artifact_manifest": str(self.cache.artifact_manifest_path(source_path)),
            "graph_input_path": str(graph_input_path),
            "graph_jsonl": str(graph_jsonl),
            "embedding_pt": str(embedding_pt),
            "num_methods": int(len(graph_records)),
            "num_method_records": int(len(tensor_records)),
            "edge_types": list(self.cfg.ast.enabled_edge_types),
            "model_name": self.cfg.embedding.model_name,
        }
        self.cache.write_json(self.cache.metadata_path(source_path), metadata)
        return metadata

    def prepare_sources(self, sources: list[Path]) -> dict[str, Any]:
        """Build artifacts for many sources and return an index payload."""

        items: dict[str, Any] = {}
        failures: list[dict[str, str]] = []

        unique_sources = sorted({path.expanduser().resolve() for path in sources})
        for source in unique_sources:
            try:
                metadata = self.prepare_program(source)
                items[str(source)] = metadata
            except Exception as exc:  # pragma: no cover - resilience over hard crash
                failures.append({"source_path": str(source), "error": str(exc)})

        payload = {"items": items, "failures": failures}
        index_path = self.cfg.cache.program_index_path.expanduser().resolve()
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def prepare_from_dataset(self, dataset_stream) -> dict[str, Any]:
        """Build artifacts for all sources referenced by a dataset stream."""

        return self.prepare_sources(dataset_stream.all_sources())
