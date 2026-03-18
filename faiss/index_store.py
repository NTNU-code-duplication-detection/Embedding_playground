"""File handler for persisting and loading a FAISS index.

The index is stored as a single binary file managed by faiss.write_index /
faiss.read_index.  All add and search operations go through this module so
the on-disk index is always kept in sync with in-memory state.
"""

from pathlib import Path

import numpy as np

from faiss._backend import _faiss

DEFAULT_INDEX_PATH = Path(__file__).resolve().parent / "data" / "index.faiss"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _create_index(dim: int):
    """Return a new empty flat L2 index for vectors of size dim."""
    return _faiss.IndexFlatL2(dim)


def _load_or_create(index_path: Path, dim: int):
    """Load the index from disk if it exists, otherwise create a fresh one."""
    if index_path.exists():
        return _faiss.read_index(str(index_path))
    return _create_index(dim)


def _save(index, index_path: Path) -> None:
    """Persist the index to disk, creating parent directories as needed."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    _faiss.write_index(index, str(index_path))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_vectors(
    vectors: np.ndarray,
    index_path: Path | str = DEFAULT_INDEX_PATH,
) -> int:
    """Add vectors to the on-disk index and save.

    If the index file does not exist yet it is created automatically.

    Args:
        vectors:    2-D float32 array of shape (N, D) — new GNN embeddings.
        index_path: Path to the .faiss index file.

    Returns:
        Total number of vectors stored in the index after the add.
    """
    index_path = Path(index_path)
    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2-D array of shape (N, D).")

    dim = vectors.shape[1]
    index = _load_or_create(index_path, dim)

    if index.d != dim:
        raise ValueError(
            f"Dimension mismatch: index has dim {index.d}, "
            f"but vectors have dim {dim}."
        )

    index.add(vectors)
    _save(index, index_path)
    return index.ntotal


def search(
    query_vectors: np.ndarray,
    k: int,
    index_path: Path | str = DEFAULT_INDEX_PATH,
) -> np.ndarray:
    """Load the on-disk index and return the k nearest neighbours for each query.

    Args:
        query_vectors: 2-D float32 array of shape (Q, D).
        k:             Number of nearest neighbours per query.
        index_path:    Path to the .faiss index file.

    Returns:
        int64 array of shape (Q, k) — indices into the stored vector set.
    """
    index_path = Path(index_path)

    if not index_path.exists():
        raise FileNotFoundError(
            f"No index found at '{index_path}'. "
            "Call add_vectors() first to build the index."
        )

    query_vectors = np.asarray(query_vectors, dtype=np.float32)

    if query_vectors.ndim != 2:
        raise ValueError("query_vectors must be a 2-D array of shape (Q, D).")

    index = _faiss.read_index(str(index_path))

    if index.d != query_vectors.shape[1]:
        raise ValueError(
            f"Dimension mismatch: index has dim {index.d}, "
            f"but query_vectors have dim {query_vectors.shape[1]}."
        )

    k = min(k, index.ntotal)
    _, indices = index.search(query_vectors, k)
    return indices


def index_size(index_path: Path | str = DEFAULT_INDEX_PATH) -> int:
    """Return the number of vectors currently stored in the index."""
    index_path = Path(index_path)
    if not index_path.exists():
        return 0
    return _faiss.read_index(str(index_path)).ntotal


def reset_index(index_path: Path | str = DEFAULT_INDEX_PATH) -> None:
    """Delete the index file, starting fresh on the next add."""
    index_path = Path(index_path)
    if index_path.exists():
        index_path.unlink()
