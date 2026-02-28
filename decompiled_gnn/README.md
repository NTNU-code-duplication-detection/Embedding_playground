# decompiled_gnn

Modular end-to-end pipeline for clone detection on decompiled Java code:

1. Pair-stream dataset splitting (`train`/`val`/`test`) created once.
   Supports `custom` dataset and `bigclonebench` via `dataset.dataset_kind`.
2. Optional compile + decompile stage (JDK + Vineflower).
3. AST statement graph extraction with configurable edges.
4. Node embeddings (default GraphCodeBERT) with disk cache per node.
5. Program-level GNN + MLP head training with configurable BCE or contrastive loss.
6. Validation/test metrics, cosine-similarity evaluation, and MC-dropout uncertainty.
7. Plot utilities for loss/AUC/F1/recall/accuracy trends.

## Main Files

- `config.default.json`: single giant config file.
- `config.py`: typed config dataclasses + JSON load/save.
- `dataset_stream.py`: singleton pair stream generator with fixed splits.
- `artifact_pipeline.py`: optional compile/decompile pipeline.
- `ast_graph.py`: AST + optional edge generation (`AST`, `SEQ`, `IF_THEN`, `IF_ELSE`, `DATA_FLOW`).
- `embeddings.py`: per-node mean-pooled transformer embeddings with persistent cache.
- `orchestrator.py`: runs all caching stages and writes `program_index.json`.
- `model.py`: edge-type-aware GNN encoder + final MLP head.
- `training.py`: train/eval loops, validation scheduler, cosine eval, MC-dropout uncertainty.
  Early stopping is configurable via `training.early_stopping_*` in `config.default.json`
  (default: metric=`f1`, patience=`200` steps, `min_delta=0.005`).
- `plotting.py`: training/test visualization.
- `decompiled_gnn_pipeline.ipynb`: notebook wiring all components.

## Quick Start

```python
from pathlib import Path
from config import load_config
from orchestrator import PipelineOrchestrator
from stream_factory import create_dataset_stream

cfg = load_config(Path("config.default.json"))
stream = create_dataset_stream(cfg)
orchestrator = PipelineOrchestrator(cfg)
program_index = orchestrator.prepare_from_dataset(stream)
```

## BigCloneBench Notes

- Set `dataset.dataset_kind = "bigclonebench"` in `config.default.json`.
- Use `dataset.clone_types` for clone family filtering (examples: `WT3_T4`, `MT3`, `ST3`, `VST3`).
- The stream reads pair labels from `dataset.bigclonebench_labels_path` and function metadata from
  `dataset.bigclonebench_db_path` using `dataset.bigclonebench_h2_jar`.
- You must point `dataset.bigclonebench_source_root` to extracted IJaDataset sources
  (for example `.../ijadataset/bcb_reduced`), otherwise source fragments cannot be resolved.
- For BigCloneBench fragments, `compilation.enabled = false` is typically preferred.
