"""Module pipeline/augment_pipeline/discovery.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class BucketIndex:
    """
    Represents one 'bucket' (e.g., GCJ folder "1".."12") with discovered java files.
    """
    bucket: str
    files: Tuple[Path, ...]


def _list_buckets(root: Path) -> List[Path]:
    buckets = [p for p in root.iterdir() if p.is_dir()]
    # Prefer numeric ordering when possible
    def key_fn(p: Path) -> Tuple[int, str]:
        try:
            return (0, int(p.name))
        except ValueError:
            return (1, p.name)
    return sorted(buckets, key=key_fn)


def discover_java_files_by_bucket(
    *,
    root: Path,
    glob: str = "*.java",
    limit_buckets: Optional[int] = None,
    max_files_per_bucket: Optional[int] = None,
) -> Dict[str, List[Path]]:
    """
    Discover .java files under root, grouped by immediate child directory name (bucket).

    For GCJ compiled dataset:
      root/gcj_compiled/1/.../*.java => bucket "1"
    """
    buckets = _list_buckets(root)
    if limit_buckets is not None:
        buckets = buckets[: max(0, limit_buckets)]

    out: Dict[str, List[Path]] = {}
    for b in buckets:
        files = sorted(b.rglob(glob))
        if max_files_per_bucket is not None:
            files = files[: max(0, max_files_per_bucket)]
        out[b.name] = files
    return out
