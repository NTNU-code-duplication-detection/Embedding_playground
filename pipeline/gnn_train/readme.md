# gnn_train

Trains a lightweight, edge-type-aware GNN encoder on method graphs and a pairwise classifier for clone detection.

This stage expects you to provide a generator that yields:
- `(method_id_a, method_id_b, label)` where label is 0/1
and `method_id_*` must exist in the embed_cache shard index.

## Inputs
- `embed_cache/shards/methods_*.pt` produced by embed_cache.
- A pair generator (your code) providing training pairs and labels.

## Outputs
- `gnn_models/method_encoder.pt`
- `gnn_models/pair_classifier.pt`
- `gnn_index.json` mapping method_id -> shard location (built automatically)

## Model overview
- Node features: `x` from the embed_cache stage (`[N, 768]` for GraphCodeBERT).
- Message passing: mean aggregation over edges with a small embedding per edge type.
- Readout: mean over node embeddings -> method embedding `[H]`.
- Pair classifier: MLP over `[a, b, |a-b|, a*b]` -> clone probability.

## Typical usage (sanity check)
This uses a dummy generator (self-pairs vs random) just to test the pipeline:
```bash
python3 -m gnn_train.cli \
  --shards "./embed_cache/shards" \
  --index "./gnn_index.json" \
  --device "mps" \
  --steps 300 \
  --batch-pairs 32
```