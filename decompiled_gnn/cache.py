"""Filesystem cache utilities for decompiled GNN artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def stable_hash(text: str) -> str:
    """Deterministic SHA1 helper used for cache keys."""

    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


class CacheManager:
    """Read/write helper for all persistent pipeline caches."""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir.expanduser().resolve()
        self.programs_root = self.root_dir / "programs"
        self.node_root = self.root_dir / "node_embeddings"
        self.programs_root.mkdir(parents=True, exist_ok=True)
        self.node_root.mkdir(parents=True, exist_ok=True)

    def program_key(self, source_path: Path) -> str:
        """Stable key for a source path."""

        return stable_hash(str(source_path.expanduser().resolve()))

    def program_dir(self, source_path: Path) -> Path:
        """Program-specific cache directory."""

        return self.programs_root / f"prog_{self.program_key(source_path)}"

    def ensure_program_dirs(self, source_path: Path) -> Path:
        """Create program subdirectories used by all stages."""

        program_dir = self.program_dir(source_path)
        (program_dir / "compiled").mkdir(parents=True, exist_ok=True)
        (program_dir / "decompiled").mkdir(parents=True, exist_ok=True)
        (program_dir / "graphs").mkdir(parents=True, exist_ok=True)
        (program_dir / "embeddings").mkdir(parents=True, exist_ok=True)
        return program_dir

    def artifact_manifest_path(self, source_path: Path) -> Path:
        """Path for compile/decompile metadata."""

        return self.program_dir(source_path) / "artifact_manifest.json"

    def graph_jsonl_path(self, source_path: Path) -> Path:
        """Path for method-level AST graph records."""

        return self.program_dir(source_path) / "graphs" / "methods.jsonl"

    def embedding_pt_path(self, source_path: Path) -> Path:
        """Path for method embedding shard (.pt)."""

        return self.program_dir(source_path) / "embeddings" / "methods.pt"

    def metadata_path(self, source_path: Path) -> Path:
        """Path for final per-program metadata."""

        return self.program_dir(source_path) / "metadata.json"

    def node_embedding_dir(self, model_name: str) -> Path:
        """Model-specific directory for per-node embedding cache entries."""

        model_hash = stable_hash(model_name)
        out = self.node_root / model_hash
        out.mkdir(parents=True, exist_ok=True)
        return out

    def node_embedding_path(self, model_name: str, text: str, max_length: int) -> Path:
        """Path for one node embedding tensor."""

        key = stable_hash(f"{model_name}|{max_length}|{text}")
        return self.node_embedding_dir(model_name) / f"{key}.pt"

    @staticmethod
    def write_json(path: Path, payload: dict[str, Any]) -> None:
        """Write a JSON document."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def read_json(path: Path) -> dict[str, Any]:
        """Read a JSON document."""

        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
        """Write JSONL records."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def read_jsonl(path: Path) -> list[dict[str, Any]]:
        """Read JSONL records."""

        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
