"""
End-to-end: Java -> Tree-sitter AST chunk-graph -> GraphCodeBERT mean-pooled node embeddings
-> R-GCN program encoder -> Siamese contrastive training (InfoNCE).

Notes:
- Uses your TreeSitterJavaGraphBuilder (already defined above in your snippet).
- Uses HuggingFace "feature-extraction" pipeline (created), but for *correct* mean pooling
  with padding masks we run tokenizer+model directly (same model as the pipeline).
- Uses CUDA if available, else MPS (Apple Silicon), else CPU.
- Dataset generator yields Java source strings (one file content at a time).
- You must provide/plug an augmentation that creates semantically-preserving "views".
  Default here is identity (two identical views) to make the pipeline runnable.

Requires:
  pip install torch transformers torch-geometric
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, pipeline

from torch_geometric.data import Data, Batch
from torch_geometric.nn import RGCNConv, global_mean_pool

sys.path.append(os.path.abspath(".."))

from data.datagenerators.bigclonebench_dataset_gen import default_generator

# ----------------------------
# Device selection (CUDA > MPS > CPU)
# ----------------------------

def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------
# Edge schema (match your builder's edges)
# ----------------------------

EDGE_TYPE_TO_ID = {
    "AST_CHILD": 0,
    "AST_PARENT": 1,
    "AST_SIBLING_NEXT": 2,
    "AST_SIBLING_PREV": 3,
    # Add later when you implement:
    # "DATA_FLOW": 4,
    # "CALLS": 5,
}

NUM_RELATIONS = len(EDGE_TYPE_TO_ID)


# ----------------------------
# Graph JSON -> PyG Data
# ----------------------------

def graph_to_pyg_data(graph: Dict[str, Any], node_x: torch.Tensor) -> Data:
    nodes = graph["nodes"]
    edges = graph["edges"]
    n = len(nodes)

    if node_x.shape[0] != n:
        raise ValueError(f"node_x rows ({node_x.shape[0]}) != #nodes ({n})")

    src_list: List[int] = []
    dst_list: List[int] = []
    et_list: List[int] = []

    for e in edges:
        et_name = e["type"]
        if et_name not in EDGE_TYPE_TO_ID:
            continue
        src_list.append(int(e["src"]))
        dst_list.append(int(e["dst"]))
        et_list.append(int(EDGE_TYPE_TO_ID[et_name]))

    if len(src_list) == 0:
        # Ensure model doesn't crash if your selection yields no edges
        src_list = list(range(n))
        dst_list = list(range(n))
        et_list = [0] * n

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_type = torch.tensor(et_list, dtype=torch.long)

    # Single-graph batch vector (all nodes belong to graph 0 in this Data object)
    batch = torch.zeros(n, dtype=torch.long)

    return Data(x=node_x, edge_index=edge_index, edge_type=edge_type, batch=batch)


# ----------------------------
# GraphCodeBERT embedder (pipeline + correct mean pooling)
# ----------------------------

class HFNodeEmbedder:
    """
    Creates:
      pipe = pipeline("feature-extraction", model=model_name)
    But uses tokenizer+model forward for correct attention-mask mean pooling.
    """

    def __init__(
        self,
        model_name: str = "microsoft/graphcodebert-base",
        max_length: int = 128,
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_best_device()
        self.model_name = model_name
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        from functools import lru_cache

        self._max_length = max_length

        @lru_cache(maxsize=200_000)
        def _cached_tokenize(text: str):
            # return python lists (hashable/serializable-ish), not torch tensors
            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=self._max_length,
                add_special_tokens=True,
            )
            return enc["input_ids"], enc["attention_mask"]

        self._cached_tokenize = _cached_tokenize

        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Create HF pipeline as requested (useful for quick debugging / sanity checks)
        # Note: transformers pipeline device uses -1 for CPU, 0+ for CUDA. MPS is not supported here.
        pipe_device = 0 if self.device.type == "cuda" else -1
        self.pipe = pipeline(
            "feature-extraction",
            model=self.model,
            tokenizer=self.tokenizer,
            device=pipe_device,
        )

        self.hidden_size = int(self.model.config.hidden_size)

    @torch.no_grad()
    def embed_texts_mean_pool(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Returns: [len(texts), hidden_size]
        Mean-pools token embeddings using attention_mask (ignores padding).
        Uses cached tokenization for speed.
        """
        all_vecs: List[torch.Tensor] = []

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            # Some models may not have pad_token set; fallback to EOS or 0
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]

            # 1) Cached tokenize each text
            encoded = [self._cached_tokenize(t) for t in chunk]  # list[(input_ids, attention_mask)]
            max_len = max(len(ids) for ids, _ in encoded)

            # 2) Pad into tensors
            input_ids = torch.full((len(encoded), max_len), pad_id, dtype=torch.long, device=self.device)
            attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long, device=self.device)

            for j, (ids, mask) in enumerate(encoded):
                L = len(ids)
                input_ids[j, :L] = torch.tensor(ids, dtype=torch.long, device=self.device)
                attention_mask[j, :L] = torch.tensor(mask, dtype=torch.long, device=self.device)

            # 3) Forward
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            token_emb = outputs.last_hidden_state                   # [B, T, H]
            mask_f = attention_mask.unsqueeze(-1).float()           # [B, T, 1]

            summed = (token_emb * mask_f).sum(dim=1)                # [B, H]
            counts = mask_f.sum(dim=1).clamp(min=1.0)               # [B, 1]
            vecs = summed / counts                                  # [B, H]

            all_vecs.append(vecs.detach().cpu())

        return torch.cat(all_vecs, dim=0)

    def node_text(self, node: Dict[str, Any], include_type: bool = True) -> str:
        code = node.get("code", "")
        if include_type:
            return f'{node.get("type","Node")}: {code}'
        return code

    def embed_graphs(self, graphs: List[Dict[str, Any]], batch_size_texts: int = 64) -> List[torch.Tensor]:
        """
        Efficiently embeds all nodes across multiple graphs in one batched forward pass.
        Returns list of node_x tensors per graph.
        """
        # Flatten node texts
        all_texts: List[str] = []
        sizes: List[int] = []
        for g in graphs:
            nodes = g["nodes"]
            sizes.append(len(nodes))
            all_texts.extend([self.node_text(n, include_type=True) for n in nodes])

        if len(all_texts) == 0:
            return [torch.empty((0, self.hidden_size), dtype=torch.float) for _ in graphs]

        all_vecs = self.embed_texts_mean_pool(all_texts, batch_size=batch_size_texts)  # [sumN, H]

        # Split back
        out: List[torch.Tensor] = []
        offset = 0
        for sz in sizes:
            out.append(all_vecs[offset : offset + sz])
            offset += sz
        return out


# ----------------------------
# Program encoder GNN (R-GCN) + Siamese projection head
# ----------------------------

class ProgramRGCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        num_relations: int = NUM_RELATIONS,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(RGCNConv(in_dim, out_dim, num_relations=num_relations))
        else:
            self.convs.append(RGCNConv(in_dim, hidden_dim, num_relations=num_relations))
            for _ in range(num_layers - 2):
                self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations))
            self.convs.append(RGCNConv(hidden_dim, out_dim, num_relations=num_relations))

        self.dropout = dropout
        self.out_dim = out_dim

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch

        for li, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if li != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # graph-level embedding
        z = global_mean_pool(x, batch)  # [num_graphs, out_dim]
        return z


class SiameseProgramEncoder(nn.Module):
    def __init__(self, gnn: ProgramRGCN, proj_dim: int = 128):
        super().__init__()
        self.gnn = gnn
        self.proj = nn.Sequential(
            nn.Linear(gnn.out_dim, gnn.out_dim),
            nn.ReLU(),
            nn.Linear(gnn.out_dim, proj_dim),
        )

    def forward(self, batch_data: Batch) -> torch.Tensor:
        z = self.gnn(batch_data)      # [B, out_dim]
        z = self.proj(z)              # [B, proj_dim]
        z = F.normalize(z, dim=-1)
        return z


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    InfoNCE / NT-Xent with in-batch negatives.
    z1, z2: [B, D]
    """
    if z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have same shape")

    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)           # [2B, D]
    sim = (z @ z.t()) / temperature          # [2B, 2B]

    # mask self similarity
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    # positives i <-> i+B
    pos = torch.arange(B, device=z.device)
    labels = torch.cat([pos + B, pos], dim=0)  # [2B]
    loss = F.cross_entropy(sim, labels)
    return loss


# ----------------------------
# Pair generation (plug in your semantics-preserving transforms)
# ----------------------------

def make_two_views(java_source: str) -> Tuple[str, str]:
    """
    Replace this with your real semantics-preserving augmentation.
    For now: identity views so the full pipeline is runnable.
    """
    return java_source, java_source


# ----------------------------
# Dataset iterator (wrap your default_generator)
# ----------------------------

class PairStream:
    """
    Takes an underlying generator that yields Java source strings.
    Produces (viewA, viewB) string pairs.
    """
    def __init__(self, base_iter: Iterable[str]):
        self.base_iter = base_iter

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        for src in self.base_iter:
            yield make_two_views(src)


# ----------------------------
# Training pipeline
# ----------------------------

@dataclass
class TrainConfig:
    model_name: str = "microsoft/graphcodebert-base"
    max_length: int = 64

    # batching
    batch_size_pairs: int = 8                  # number of (A,B) pairs per step
    embed_text_batch_size: int = 64            # batching for node text encoding

    # GNN
    gnn_hidden_dim: int = 256
    gnn_out_dim: int = 256
    gnn_layers: int = 3
    gnn_dropout: float = 0.1
    proj_dim: int = 128

    # optimization
    lr: float = 2e-4
    weight_decay: float = 1e-2
    temperature: float = 0.2
    grad_clip: float = 1.0

    # steps
    max_steps: int = 2000
    log_every: int = 20

    debug_cache = True


def build_batch_from_pairs(
    pairs: List[Tuple[str, str]],
    embedder: HFNodeEmbedder,
    device: torch.device,
) -> Tuple[Batch, Batch]:
    """
    For a list of (codeA, codeB):
      - build graphs
      - embed nodes (batched across all graphs)
      - convert to PyG Data
      - batch them into Batch objects (one for A, one for B)
    """
    graphs_a: List[Dict[str, Any]] = []
    graphs_b: List[Dict[str, Any]] = []

    for code_a, code_b in pairs:
        ga = TreeSitterJavaGraphBuilder(code_a).build()
        gb = TreeSitterJavaGraphBuilder(code_b).build()
        graphs_a.append(ga)
        graphs_b.append(gb)

    # Embed nodes for all graphs in A and B (separately to keep alignment simple)
    node_x_list_a = embedder.embed_graphs(graphs_a, batch_size_texts=embedder.max_length if False else 64)
    node_x_list_b = embedder.embed_graphs(graphs_b, batch_size_texts=embedder.max_length if False else 64)

    data_list_a: List[Data] = []
    data_list_b: List[Data] = []

    for ga, xa in zip(graphs_a, node_x_list_a):
        da = graph_to_pyg_data(ga, xa)
        data_list_a.append(da)

    for gb, xb in zip(graphs_b, node_x_list_b):
        db = graph_to_pyg_data(gb, xb)
        data_list_b.append(db)

    batch_a = Batch.from_data_list(data_list_a).to(device)
    batch_b = Batch.from_data_list(data_list_b).to(device)
    return batch_a, batch_b


from typing import Iterable, Tuple, List, Optional
from tqdm.auto import tqdm
import torch

def train_siamese(
    pair_iter: Iterable[Tuple[str, str]],
    cfg: TrainConfig = TrainConfig(),
    *,
    save_dir: Optional[str] = None,
    save_every: int = 500,
) -> SiameseProgramEncoder:
    device = get_best_device()
    print(f"[train] device={device}")

    # Embedder (GraphCodeBERT) - frozen feature extractor
    embedder = HFNodeEmbedder(
        model_name=cfg.model_name,
        max_length=cfg.max_length,
        device=device,
    )
    embedder.model.eval()
    for p in embedder.model.parameters():
        p.requires_grad = False

    # Cache testing (optional, cheap, run once)
    if getattr(cfg, "debug_cache", True):
        texts = ["Statement: return x;", "Statement: return x;"] * 100
        _ = embedder.embed_texts_mean_pool(texts, batch_size=32)
        print("cache after 1st:", embedder._cached_tokenize.cache_info())
        _ = embedder.embed_texts_mean_pool(texts, batch_size=32)
        print("cache after 2nd:", embedder._cached_tokenize.cache_info())

    # GNN + Siamese head
    gnn = ProgramRGCN(
        in_dim=embedder.hidden_size,
        hidden_dim=cfg.gnn_hidden_dim,
        out_dim=cfg.gnn_out_dim,
        num_layers=cfg.gnn_layers,
        dropout=cfg.gnn_dropout,
        num_relations=NUM_RELATIONS,
    ).to(device)

    model = SiameseProgramEncoder(gnn=gnn, proj_dim=cfg.proj_dim).to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Progress bar over steps (not over items in pair_iter, since that's an iterator)
    pbar = tqdm(range(1, cfg.max_steps + 1), desc="Training", dynamic_ncols=True)

    buf: List[Tuple[str, str]] = []

    # Helper: optional checkpoint saving
    def _maybe_save(step: int):
        if save_dir is None:
            return
        if step % save_every != 0:
            return
        ckpt = {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "cfg": cfg.__dict__ if hasattr(cfg, "__dict__") else None,
        }
        import os
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        path = os.path.join(save_dir, f"siamese_step_{step}.pt")
        torch.save(ckpt, path)

    data_iter = iter(pair_iter)

    for step in pbar:
        # Fill buffer to batch size
        while len(buf) < cfg.batch_size_pairs:
            try:
                buf.append(next(data_iter))
            except StopIteration:
                # No more data; stop training early (cleanly)
                pbar.close()
                return model

        batch_pairs = buf[:cfg.batch_size_pairs]
        buf = buf[cfg.batch_size_pairs:]

        # Build batched graphs for both views
        batch_a, batch_b = build_batch_from_pairs(batch_pairs, embedder, device)

        # Forward
        z1 = model(batch_a)  # [B, D], normalized
        z2 = model(batch_b)

        loss = nt_xent_loss(z1, z2, temperature=cfg.temperature)

        # Similarity diagnostics (cheap + very useful)
        with torch.no_grad():
            pos_sim = (z1 * z2).sum(dim=1).mean().item()
            neg_sim = (z1 * z2.roll(1, 0)).sum(dim=1).mean().item() if z1.size(0) > 1 else float("nan")
            gap = pos_sim - neg_sim if z1.size(0) > 1 else float("nan")
            # Optional: nodes count helps understand speed
            nodes_a = int(batch_a.x.size(0)) if hasattr(batch_a, "x") and batch_a.x is not None else -1
            nodes_b = int(batch_b.x.size(0)) if hasattr(batch_b, "x") and batch_b.x is not None else -1

        # Backward
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        # tqdm status
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "gap": f"{gap:.3f}" if z1.size(0) > 1 else "nan",
            "pos": f"{pos_sim:.3f}",
            "neg": f"{neg_sim:.3f}" if z1.size(0) > 1 else "nan",
            "nodes": f"{nodes_a}/{nodes_b}",
        })

        # Optional periodic print (keep if you want logs in files)
        if cfg.log_every and step % cfg.log_every == 0:
            print(f"[step {step:>5}] loss={loss.item():.4f} pos={pos_sim:.3f} neg={neg_sim:.3f} gap={gap:.3f}")

        _maybe_save(step)

    return model


# ----------------------------
# Example usage with your generator
# ----------------------------

# Your generator (as described) yields Java source code strings (file content).
# def default_generator(dataset_root: str | Path = DEFAULT_BIGCLONEBENCH_ROOT):
#     dataset_root = Path(dataset_root).expanduser().resolve()
#     default_path = dataset_root / "default"
#     for java_file in default_path.glob('*.java'):
#         yield _read_java(java_file)

def run_training_with_default_generator(dataset_root: str | Path = None, cfg: TrainConfig = TrainConfig()):
    base_iter = default_generator(dataset_root)
    pair_iter = PairStream(base_iter)  # yields (viewA, viewB)
    model = train_siamese(pair_iter, cfg=cfg, save_dir="checkpoints", save_every=50)
    return model


# ----------------------------
# Quick smoke test on a tiny stream (no dataset needed)
# ----------------------------

def tiny_stream() -> Iterator[str]:
    yield """
    class Demo {
      int foo(int a, int b) {
        int x = a + 1;
        if (x > b) { x = x - b; } else { x = x + b; }
        return x;
      }
    }
    """.strip()
    yield """
    class Demo2 {
      int bar(int n) {
        int s = 0;
        for (int i=0;i<n;i++) { s += i; }
        return s;
      }
    }
    """.strip()

def smoke_test():
    cfg = TrainConfig(max_steps=50, log_every=10, batch_size_pairs=2)
    model = train_siamese(PairStream(tiny_stream()), cfg=cfg)
    return model


# If you want to run interactively:
#model = smoke_test()
model = run_training_with_default_generator("/path/to/BigCloneBench", TrainConfig(max_steps=200))
