# embed_cache

Embeds each node's `code` into a fixed vector using a Hugging Face encoder (e.g. GraphCodeBERT),
caches embeddings by content hash, and writes tensor-only shard files for fast training.

This stage exists to:
- Pay the transformer cost once.
- Train GNN/MLP many times from cached tensors.

## Inputs
- `methods.jsonl` from `ast_dataset`.

## Outputs
- `embed_cache/shards/methods_00000.pt`, `methods_00001.pt`, ...
  Each shard is a list of dicts:
  - `method_id`
  - `file`
  - `x`: node features `[N, D]`
  - `edge_index`: `[2, E]`
  - `edge_type`: `[E]`
- `embed_manifest.json` describing the run.
- A cache directory of per-node vectors: `node_cache/<sha1>.pt`

## Typical usage
```bash
python3 -m embed_cache.cli \
  --in-jsonl "./methods.jsonl" \
  --out "./embed_cache" \
  --cache "./node_cache" \
  --model "microsoft/graphcodebert-base" \
  --device "mps" \
  --batch-size 32 \
  --max-length 256 \
  --shard-size 2000
```