"""Pytest tests demonstrating search_similar on IR-Plag-Dataset.

GNN embeddings are simulated with TF-IDF vectors over Java source code,
since a trained GNN is not yet available. The FAISS search logic is
identical regardless of how the vectors are produced.

Run from the project root:
    pytest faiss/test_search.py -v
"""

# pylint: disable=redefined-outer-name

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# pylint: disable=wrong-import-position
from data.data_generators.sourcecodeplag_dataset_gen import (
    original_non_plagiarized_generator,
    original_plagiarized_generator,
)
from faiss.search import add_to_index, search_similar
# pylint: enable=wrong-import-position


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_samples(max_samples: int = 60):
    """Return (codes, labels) from IR-Plag-Dataset."""
    codes, labels = [], []

    for sample in original_plagiarized_generator():
        codes.append(sample.code_b)
        labels.append("plagiarised")
        if len(codes) >= max_samples // 2:
            break

    for sample in original_non_plagiarized_generator():
        codes.append(sample.code_b)
        labels.append("non-plagiarised")
        if len(codes) >= max_samples:
            break

    return codes, labels


def _embed(codes: list[str]) -> np.ndarray:
    """TF-IDF float32 matrix -- stand-in for GNN output vectors."""
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[A-Za-z_]\w*",
        max_features=512,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(codes).toarray().astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """Load 60 samples, embed them, build a temp index, return (vectors, labels, index_path)."""
    codes, labels = _load_samples(max_samples=60)
    vectors = _embed(codes)
    index_path = tmp_path_factory.mktemp("faiss") / "index.faiss"
    add_to_index(vectors, index_path=str(index_path))
    return vectors, labels, index_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_output_shape(dataset):
    """search_similar returns an array of shape (Q, k)."""
    vectors, _, index_path = dataset
    k = 5
    result = search_similar(vectors[:3], k, index_path=index_path)
    assert result.shape == (3, k), f"Expected (3, {k}), got {result.shape}"


def test_self_is_nearest_neighbour(dataset):
    """A vector queried against the full index should be its own closest match."""
    vectors, _, index_path = dataset
    result = search_similar(vectors[[0]], 1, index_path=index_path)
    assert result[0, 0] == 0, (
        f"Expected self (idx 0) as nearest neighbour, got {result[0, 0]}"
    )


def test_indices_in_valid_range(dataset):
    """All returned indices must be valid positions in the dataset."""
    vectors, _, index_path = dataset
    n = vectors.shape[0]
    result = search_similar(vectors[:10], 5, index_path=index_path)
    assert result.min() >= 0 and result.max() < n, (
        f"Indices out of range [0, {n}): min={result.min()}, max={result.max()}"
    )


def test_plagiarised_clusters_near_plagiarised(dataset):
    """Plagiarised files should have predominantly plagiarised nearest neighbours."""
    vectors, labels, index_path = dataset
    plag_indices = [i for i, lbl in enumerate(labels) if lbl == "plagiarised"]
    query = vectors[plag_indices[:5]]
    result = search_similar(query, 5, index_path=index_path)

    for row_idx, neighbours in enumerate(result):
        neighbour_labels = [labels[n] for n in neighbours if n != plag_indices[row_idx]]
        plag_count = neighbour_labels.count("plagiarised")
        assert plag_count >= 2, (
            f"Query {plag_indices[row_idx]} (plagiarised): expected >=2 plagiarised "
            f"neighbours, got {plag_count} in {neighbour_labels}"
        )


def test_batch_vs_single_consistency(dataset):
    """Batch search must return the same results as individual single queries."""
    vectors, _, index_path = dataset
    query_indices = [0, 5, 15]
    k = 4

    batch_result = search_similar(vectors[query_indices], k, index_path=index_path)

    for pos, idx in enumerate(query_indices):
        single_result = search_similar(vectors[[idx]], k, index_path=index_path)
        assert list(batch_result[pos]) == list(single_result[0]), (
            f"Mismatch for query {idx}: "
            f"batch={list(batch_result[pos])}, single={list(single_result[0])}"
        )


def test_k_clamped_to_dataset_size(dataset):
    """Requesting more neighbours than dataset size should not raise an error."""
    vectors, _, index_path = dataset
    n = vectors.shape[0]
    result = search_similar(vectors[:2], n + 100, index_path=index_path)
    assert result.shape[1] == n, (
        f"k should be clamped to {n}, got {result.shape[1]}"
    )


def test_dimension_mismatch_raises(tmp_path):
    """Mismatched vector dimensions must raise ValueError."""
    index_path = tmp_path / "dim_test.faiss"
    add_to_index(np.random.rand(10, 64).astype(np.float32), index_path=str(index_path))
    queries = np.random.rand(2, 32).astype(np.float32)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        search_similar(queries, 3, index_path=index_path)


def test_non_2d_input_raises(tmp_path):
    """1-D input arrays must raise ValueError."""
    index_path = tmp_path / "2d_test.faiss"
    add_to_index(np.random.rand(10, 64).astype(np.float32), index_path=str(index_path))
    flat_query = np.random.rand(64).astype(np.float32)
    with pytest.raises(ValueError):
        search_similar(flat_query, 3, index_path=index_path)
