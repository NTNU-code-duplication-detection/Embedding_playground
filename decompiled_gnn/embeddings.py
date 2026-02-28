"""Node embedding stage with persistent per-node and per-program caches."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModel, AutoTokenizer

from cache import CacheManager
from config import EmbeddingConfig


class NodeEmbedder:
    """Embed AST statement nodes and save both node-level and method-level artifacts."""

    def __init__(
        self,
        cfg: EmbeddingConfig,
        cache: CacheManager,
        edge_type_to_id: dict[str, int],
        device: str,
        fallback_dim: int = 768,
    ):
        self.cfg = cfg
        self.cache = cache
        self.edge_type_to_id = edge_type_to_id
        self.device = cfg.embedding_device or device
        self.fallback_dim = fallback_dim

        self._tokenizer = None
        self._model = None
        self._embedding_dim: Optional[int] = None

    def _ensure_model(self) -> None:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        if self._model is None:
            self._model = AutoModel.from_pretrained(self.cfg.model_name).to(self.device).eval()
            hidden = getattr(self._model.config, "hidden_size", None)
            if hidden is not None:
                self._embedding_dim = int(hidden)

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is not None:
            return self._embedding_dim
        return self.fallback_dim

    def _mean_pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def _embed_batch(self, texts: list[str]) -> torch.Tensor:
        self._ensure_model()
        assert self._tokenizer is not None
        assert self._model is not None

        tokens = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            output = self._model(**tokens)
            pooled = self._mean_pool(output.last_hidden_state, tokens["attention_mask"])
        return pooled.detach().cpu().to(torch.float32)

    def _edge_tensors(self, edges: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        src: list[int] = []
        dst: list[int] = []
        edge_ids: list[int] = []

        for edge in edges:
            edge_type = edge.get("type")
            if edge_type not in self.edge_type_to_id:
                continue
            src.append(int(edge["src"]))
            dst.append(int(edge["dst"]))
            edge_ids.append(int(self.edge_type_to_id[edge_type]))

        if not src:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,), dtype=torch.long)
        return torch.tensor([src, dst], dtype=torch.long), torch.tensor(edge_ids, dtype=torch.long)

    def _node_embedding(self, text: str) -> torch.Tensor:
        cache_path = self.cache.node_embedding_path(self.cfg.model_name, text, self.cfg.max_length)
        if cache_path.exists():
            emb = torch.load(cache_path, map_location="cpu")
            if isinstance(emb, torch.Tensor):
                self._embedding_dim = int(emb.shape[-1])
                return emb.to(torch.float32)

        emb = self._embed_batch([text])[0]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(emb, cache_path)
        self._embedding_dim = int(emb.shape[-1])
        return emb

    def _warm_missing_embeddings(self, graph_records: list[dict[str, Any]]) -> None:
        missing_texts: list[str] = []
        seen: set[str] = set()

        for record in graph_records:
            for node in record.get("nodes", []):
                text = node.get("code", "")
                if not text:
                    continue
                cache_path = self.cache.node_embedding_path(self.cfg.model_name, text, self.cfg.max_length)
                if cache_path.exists() or text in seen:
                    continue
                seen.add(text)
                missing_texts.append(text)

        if not missing_texts:
            return

        self._ensure_model()
        batch_size = max(1, self.cfg.batch_size)
        for offset in range(0, len(missing_texts), batch_size):
            batch = missing_texts[offset : offset + batch_size]
            embs = self._embed_batch(batch)
            for text, emb in zip(batch, embs):
                cache_path = self.cache.node_embedding_path(self.cfg.model_name, text, self.cfg.max_length)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(emb, cache_path)
                self._embedding_dim = int(emb.shape[-1])

    def embed_graph_records(
        self,
        graph_records: list[dict[str, Any]],
        out_pt: Path,
    ) -> list[dict[str, Any]]:
        """Convert graph records into tensor records and save them to `out_pt`."""

        self._warm_missing_embeddings(graph_records)

        tensor_records: list[dict[str, Any]] = []
        for record in graph_records:
            nodes = record.get("nodes", [])
            edges = record.get("edges", [])
            texts = [node.get("code", "") for node in nodes]

            if texts:
                x = torch.stack([self._node_embedding(text) for text in texts], dim=0)
            else:
                x = torch.zeros((0, self.embedding_dim), dtype=torch.float32)

            edge_index, edge_type = self._edge_tensors(edges)
            tensor_records.append(
                {
                    "method_id": record.get("method_id"),
                    "method_name": record.get("method_name"),
                    "file": record.get("file"),
                    "nodes": nodes,
                    "edges": edges,
                    "x": x,
                    "edge_index": edge_index,
                    "edge_type": edge_type,
                    "num_nodes": int(x.shape[0]),
                    "num_edges": int(edge_index.shape[1]),
                }
            )

        out_pt = out_pt.expanduser().resolve()
        out_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor_records, out_pt)
        return tensor_records
