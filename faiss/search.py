"""Nearest-neighbour search over GNN embeddings using a persistent FAISS index.

Primary entry points
--------------------
    add_to_index(vectors)            -- store new GNN embeddings in the index file
    search_similar(query_vectors, k) -- find k nearest neighbours from the index

The index is persisted to faiss/data/index.faiss via index_store.py so that
vectors added in one session are available in the next.
"""

from pathlib import Path

import numpy as np


def add_to_index(
    vectors: np.ndarray,
    index_path: Path | str | None = None,
) -> int:
    """Add GNN embedding vectors to the persistent index.

    Loads the existing index from disk (or creates one), appends the new
    vectors, and saves back to disk.

    Args:
        vectors:    2-D float32 array of shape (N, D).
        index_path: Override the default index file location.

    Returns:
        Total number of vectors in the index after the add.
    """
    from faiss.index_store import add_vectors, DEFAULT_INDEX_PATH  # pylint: disable=import-outside-toplevel
    path = Path(index_path) if index_path else DEFAULT_INDEX_PATH
    return add_vectors(vectors, path)


def search_similar(
    query_vectors: np.ndarray,
    k: int,
    index_path: Path | str | None = None,
) -> np.ndarray:
    """Search the persistent index for the k nearest neighbours of each query.

    Args:
        query_vectors: 2-D float32 array of shape (Q, D).
        k:             Number of nearest neighbours to return per query.
        index_path:    Override the default index file location.

    Returns:
        int64 array of shape (Q, k) -- indices into the indexed vector set.
    """
    from faiss.index_store import search, DEFAULT_INDEX_PATH  # pylint: disable=import-outside-toplevel
    path = Path(index_path) if index_path else DEFAULT_INDEX_PATH
    return search(query_vectors, k, path)
