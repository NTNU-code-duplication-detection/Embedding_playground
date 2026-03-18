"""
Visual walkthrough of search_similar using IR-Plag-Dataset.

Shows exactly what data goes into FAISS and what comes back.

Run:
    python -m faiss.demo
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# pylint: disable=wrong-import-position
from data.data_generators.sourcecodeplag_dataset_gen import (
    original_non_plagiarized_generator,
    original_plagiarized_generator,
)
from faiss.index_store import reset_index
from faiss.search import add_to_index, search_similar
# pylint: enable=wrong-import-position

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUERY_IDX = 0
K = 4
BATCH_QUERY_INDICES = [0, 3, 6, 9]
BATCH_K = 3

# ---------------------------------------------------------------------------
# Load a small slice of the dataset
# ---------------------------------------------------------------------------

codes, labels = [], []

for sample in original_plagiarized_generator():
    codes.append(sample.code_b)
    labels.append("plagiarised")
    if len(codes) == 6:
        break

for sample in original_non_plagiarized_generator():
    codes.append(sample.code_b)
    labels.append("non-plagiarised")
    if len(codes) == 12:
        break

print("=" * 60)
print("STEP 1 -- Raw Java files loaded from IR-Plag-Dataset")
print("=" * 60)
for i, (code, label) in enumerate(zip(codes, labels)):
    preview = code.strip().splitlines()[0][:60]
    print(f"  [{i:>2}]  {label:<16}  {preview!r}")

# ---------------------------------------------------------------------------
# Embed (TF-IDF proxy for GNN vectors)
# ---------------------------------------------------------------------------

vectorizer = TfidfVectorizer(
    analyzer="word",
    token_pattern=r"[A-Za-z_]\w*",
    max_features=8,          # keep tiny so we can print the whole vector
    sublinear_tf=True,
)
matrix = vectorizer.fit_transform(codes).toarray().astype(np.float32)
norms = np.linalg.norm(matrix, axis=1, keepdims=True)
norms[norms == 0] = 1.0
vectors = matrix / norms

print()
print("=" * 60)
print("STEP 2 -- Vectors passed INTO FAISS index")
print(f"         shape: {vectors.shape}  (files x features)")
print(f"         feature names: {vectorizer.get_feature_names_out().tolist()}")
print("=" * 60)
for i, (vec, label) in enumerate(zip(vectors, labels)):
    formatted = "  ".join(f"{v:+.3f}" for v in vec)
    print(f"  [{i:>2}]  {label:<16}  [ {formatted} ]")

# ---------------------------------------------------------------------------
# Build persistent index
# ---------------------------------------------------------------------------

reset_index()
add_to_index(vectors)

# ---------------------------------------------------------------------------
# Single query
# ---------------------------------------------------------------------------

query_vec = vectors[[QUERY_IDX]]   # shape (1, 8)

print()
print("=" * 60)
print(f"STEP 3 -- Query vector (file #{QUERY_IDX}, {labels[QUERY_IDX]})")
print("=" * 60)
formatted = "  ".join(f"{v:+.3f}" for v in query_vec[0])
print(f"  shape: {query_vec.shape}")
print(f"  values: [ {formatted} ]")

print()
print("=" * 60)
print(f"STEP 4 -- Output from search_similar  (k={K})")
print("=" * 60)

indices = search_similar(query_vec, K)

print(f"  raw indices array:  {indices}")
print(f"  shape:              {indices.shape}  (queries x k)")
print()
print(f"  Decoded results for query #{QUERY_IDX} [{labels[QUERY_IDX]}]:")
print(f"  {'Rank':<6} {'Index':<8} {'Label':<18} {'Code preview'}")
print(f"  {'-'*70}")
for rank, idx in enumerate(indices[0], start=1):
    preview = codes[idx].strip().splitlines()[0][:45]
    marker = " <-- this is the query file" if idx == QUERY_IDX else ""
    print(f"  {rank:<6} {idx:<8} {labels[idx]:<18} {preview!r}{marker}")

# ---------------------------------------------------------------------------
# Batch query
# ---------------------------------------------------------------------------

batch_vecs = vectors[BATCH_QUERY_INDICES]   # shape (4, 8)

print()
print("=" * 60)
print(f"STEP 5 -- Batch query: {len(BATCH_QUERY_INDICES)} files at once  (k={BATCH_K})")
print("=" * 60)
print(f"  Input shape:  {batch_vecs.shape}  (queries x features)")

batch_indices = search_similar(batch_vecs, BATCH_K)

print(f"  Output shape: {batch_indices.shape}  (queries x k)")
print()
print(f"  {'Query idx':<12} {'Label':<18} {'Neighbour indices':<20} {'Neighbour labels'}")
print(f"  {'-'*70}")
for q_idx, neighbours in zip(BATCH_QUERY_INDICES, batch_indices):
    neighbour_labels = [labels[n] for n in neighbours]
    print(f"  {q_idx:<12} {labels[q_idx]:<18} {neighbours.tolist()!s:<20} {neighbour_labels}")
