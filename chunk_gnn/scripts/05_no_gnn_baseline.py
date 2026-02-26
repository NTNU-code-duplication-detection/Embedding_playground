"""
05_no_gnn_baseline.py — Evaluate clone detection WITHOUT the GNN.

Mean-pools raw UniXcoder chunk embeddings per function, then computes
cosine similarity between pairs. This tells us whether the GNN is
actually adding value or if we're just doing expensive averaging.

If this baseline matches the GNN's F1@optimal (~0.90), the GNN
contributes nothing and the performance comes entirely from UniXcoder.

Usage:
    python -m chunk_gnn.scripts.05_no_gnn_baseline \
        --bcb_root ~/datasets/bigclonebench \
        --cache_dir ~/datasets/chunk_gnn_cache
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from chunk_gnn.data.bcb_loader import BCBLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_mean_embedding(cache_dir: Path, func_id: str) -> torch.Tensor | None:
    """Load cached graph and return mean-pooled node features."""
    path = cache_dir / f"{func_id}.pt"
    if not path.exists():
        return None

    data = torch.load(path, weights_only=False, map_location="cpu")
    x = data.x.float()  # (num_chunks, 768)
    return x.mean(dim=0)  # (768,)


def main():
    parser = argparse.ArgumentParser(
        description="No-GNN baseline: mean-pool UniXcoder embeddings"
    )
    parser.add_argument("--bcb_root", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    # Load BCB pairs
    log.info("Loading BCB labels...")
    loader = BCBLoader(bcb_root=args.bcb_root)
    test_pairs = loader.get_split("test")
    log.info("Test pairs: %d", len(test_pairs))

    # Pre-load all unique function embeddings (mean-pooled)
    log.info("Loading and mean-pooling embeddings...")
    unique_ids = set()
    for p in test_pairs:
        unique_ids.add(p.id1)
        unique_ids.add(p.id2)

    embeddings: dict[str, torch.Tensor] = {}
    missing = 0
    for func_id in unique_ids:
        emb = load_mean_embedding(cache_dir, func_id)
        if emb is not None:
            embeddings[func_id] = emb
        else:
            missing += 1

    log.info(
        "Loaded %d function embeddings (%d missing)",
        len(embeddings), missing,
    )

    # Compute cosine similarities for all test pairs
    log.info("Computing cosine similarities...")
    sims = []
    labels = []
    types = []
    skipped = 0

    for pair in test_pairs:
        if pair.id1 not in embeddings or pair.id2 not in embeddings:
            skipped += 1
            continue

        emb1 = embeddings[pair.id1]
        emb2 = embeddings[pair.id2]

        cos_sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        sims.append(cos_sim)
        labels.append(1 if pair.clone_label == 1 else 0)
        types.append(pair.clone_type)

    log.info("Evaluated %d pairs (%d skipped)", len(sims), skipped)

    sims_arr = np.array(sims)
    labels_arr = np.array(labels)

    # --- Similarity distribution stats ---
    clone_sims = sims_arr[labels_arr == 1]
    nonclone_sims = sims_arr[labels_arr == 0]

    log.info("")
    log.info("=" * 60)
    log.info("SIMILARITY DISTRIBUTION (no GNN, raw UniXcoder mean-pool)")
    log.info("=" * 60)
    log.info("Clone pairs:     mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
             clone_sims.mean(), clone_sims.std(), clone_sims.min(), clone_sims.max())
    log.info("Non-clone pairs: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
             nonclone_sims.mean(), nonclone_sims.std(), nonclone_sims.min(), nonclone_sims.max())

    # --- F1 at t=0 ---
    preds_t0 = (sims_arr > 0).astype(int)
    f1_t0 = f1_score(labels_arr, preds_t0, zero_division=0)
    p_t0 = precision_score(labels_arr, preds_t0, zero_division=0)
    r_t0 = recall_score(labels_arr, preds_t0, zero_division=0)

    # --- Threshold sweep ---
    best_f1 = 0.0
    best_threshold = 0.0
    thresholds = np.linspace(-1.0, 1.0, 1000)

    for t in thresholds:
        preds = (sims_arr > t).astype(int)
        f1 = f1_score(labels_arr, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)

    preds_opt = (sims_arr > best_threshold).astype(int)
    p_opt = precision_score(labels_arr, preds_opt, zero_division=0)
    r_opt = recall_score(labels_arr, preds_opt, zero_division=0)

    # --- Per-type recall at optimal threshold ---
    types_arr = np.array(types)
    for ctype in sorted(set(types)):
        if ctype == "Non_Clone":
            continue
        mask = (types_arr == ctype) & (labels_arr == 1)
        if mask.sum() == 0:
            continue
        type_recall = recall_score(labels_arr[mask], preds_opt[mask], zero_division=0)
        log.info("  %s recall: %.4f (n=%d)", ctype, type_recall, mask.sum())

    log.info("")
    log.info("=" * 60)
    log.info("NO-GNN BASELINE RESULTS")
    log.info("=" * 60)
    log.info("F1@t=0:      P=%.4f  R=%.4f  F1=%.4f", p_t0, r_t0, f1_t0)
    log.info("F1@optimal:  P=%.4f  R=%.4f  F1=%.4f  (t=%.4f)",
             p_opt, r_opt, best_f1, best_threshold)
    log.info("")
    log.info("Compare with GNN:")
    log.info("  MVP:     F1@t=0=0.575, F1@opt=0.906 (t=0.869)")
    log.info("  Phase1:  F1@t=0=0.567, F1@opt=0.903 (t=0.980)")


if __name__ == "__main__":
    main()
