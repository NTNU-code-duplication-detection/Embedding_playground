"""Loads the real faiss library, bypassing our local 'faiss/' folder.

Both search.py and index_store.py import _faiss from here so neither
has to depend on the other, eliminating the cyclic import.
"""

import os
import sys
from pathlib import Path


def _load_faiss():
    """Return the installed faiss module.

    Strategy:
      1. Save all local faiss.* entries from sys.modules and sys.path.
      2. Temporarily remove the project root from sys.path and clear faiss
         from sys.modules so a plain 'import faiss' finds the installed package.
      3. Import and cache it under '_real_faiss'.
      4. Restore sys.path and our local faiss.* modules so the rest of the
         project continues to work normally.
    """
    if "_real_faiss" in sys.modules:
        return sys.modules["_real_faiss"]

    project_root = str(Path(__file__).resolve().parents[1])

    local_faiss = {k: v for k, v in sys.modules.items()
                   if k == "faiss" or k.startswith("faiss.")}
    original_path = sys.path[:]

    sys.path = [p for p in sys.path if os.path.abspath(p) != project_root]
    for k in local_faiss:
        sys.modules.pop(k, None)

    try:
        import faiss as real_faiss  # pylint: disable=import-outside-toplevel
        sys.modules["_real_faiss"] = real_faiss
    finally:
        sys.path = original_path
        for k, v in local_faiss.items():
            sys.modules[k] = v

    return real_faiss


_faiss = _load_faiss()
