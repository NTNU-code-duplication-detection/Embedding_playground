# Chunk-Level GNN for Code Clone Detection

A hybrid architecture combining pre-trained code embeddings (UniXcoder) with
Graph Neural Networks for code clone detection on BigCloneBench.

## Architecture

```
Java function (source code)
  |
  v
Tree-sitter chunking --> list of chunks (STRAIGHT / CONTROL)
  |
  v
UniXcoder embedding --> each chunk gets a 768-dim vector
  |
  v
Graph construction --> nodes = chunks, edges = sequential + parent-child
  |
  v
GNN message passing (2 layers GCNConv) --> graph-level 128-dim embedding
  |
  v
Siamese comparison: cosine_similarity(emb_A, emb_B) --> clone / not-clone
```

## Key Design Decisions

- **Chunk-level nodes** (not token-level): Each graph node represents a
  meaningful code segment (a statement group or control structure), not
  an individual AST token. This gives much smaller graphs than MagNET
  (~5-30 nodes vs hundreds/thousands) with richer node features.

- **Pre-trained embeddings**: Node features come from UniXcoder (768-dim),
  which already encodes semantic understanding of code. MagNET learns
  node embeddings from scratch (128-dim vocabulary lookup).

- **MVP edge types**: Sequential (chunk_i <-> chunk_{i+1}) and parent-child
  (from AST nesting). Data-flow edges are deferred to Phase 2.

## MVP vs Future Work

| Feature | MVP | Future |
|---------|-----|--------|
| Node features | UniXcoder 768-dim | Same |
| Edge types | Sequential + parent-child | + Data-flow, sibling |
| GNN layers | 2x GCNConv | GATConv, cross-attention |
| Pooling | global_mean_pool | Set2Set |
| Edge embeddings | None (homogeneous) | Typed edge embeddings |
| LR scheduler | Constant LR | ReduceLROnPlateau |

## Usage

```bash
# Set paths (adjust to your environment)
BCB_ROOT=~/Multigraph_match_optimized/data/data_source/dataset_bigclonebench
CACHE_DIR=~/chunk_gnn_cache
OUTPUT_DIR=~/chunk_gnn_out

# 1. Explore the BCB dataset
python scripts/01_explore_bcb.py --bcb_root $BCB_ROOT

# 2. Pre-compute chunk graphs for all BCB functions
python scripts/02_precompute_graphs.py --bcb_root $BCB_ROOT --cache_dir $CACHE_DIR --device cuda

# 3. Train the model
python scripts/03_train.py --config configs/bcb_mvp.json --bcb_root $BCB_ROOT --cache_dir $CACHE_DIR --output_dir $OUTPUT_DIR

# 4. Evaluate a checkpoint
python scripts/04_evaluate.py --config configs/bcb_mvp.json --bcb_root $BCB_ROOT --cache_dir $CACHE_DIR --checkpoint $OUTPUT_DIR/run_XXXXXX/checkpoints/best_model.pt
```

## Full Implementation Plan

See: `/Documents/chunk-gnn-mvp-plan.md`
