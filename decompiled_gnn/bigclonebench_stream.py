"""BigCloneBench pair stream compatible with the custom PairDatasetStream interface."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
from typing import Iterator, Literal, Optional

from config import DatasetConfig

Pair = tuple[str, str, int]
SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class BigCloneBenchPairRow:
    """One labeled pair row from `clone_labels_typed.txt`."""

    id1: int
    id2: int
    label: int
    clone_type: str


@dataclass(frozen=True)
class FunctionMeta:
    """Metadata for one function ID from BigCloneBench `functions` table."""

    function_id: int
    project: str
    type: str
    name: str
    startline: int
    endline: int


class BigCloneBenchResolver:
    """Resolve function IDs to fragment `.java` files using the H2 DB metadata."""

    def __init__(self, cfg: DatasetConfig, cache_root: Path):
        self.cfg = cfg
        self.cache_root = cache_root.expanduser().resolve()
        self.cache_dir = self.cache_root / "bigclonebench"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.fragments_dir = self.cache_dir / "fragments"
        self.fragments_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_cache_path = self.cache_dir / "function_metadata_cache.json"
        self.metadata_cache = self._load_metadata_cache()

        self.source_root = (
            cfg.bigclonebench_source_root.expanduser().resolve()
            if cfg.bigclonebench_source_root is not None
            else None
        )

        self.db_h2_path = self._prepare_working_db_copy()
        self.db_url = self._h2_url_from_h2_file(self.db_h2_path)

    def _prepare_working_db_copy(self) -> Path:
        src_h2 = self.cfg.bigclonebench_db_path.expanduser().resolve()
        if not src_h2.exists():
            raise FileNotFoundError(f"BigCloneBench DB not found: {src_h2}")

        db_copy_dir = self.cache_dir / "db_copy"
        db_copy_dir.mkdir(parents=True, exist_ok=True)

        dst_h2 = db_copy_dir / src_h2.name
        src_trace = src_h2.with_suffix(".trace.db")
        dst_trace = dst_h2.with_suffix(".trace.db")

        if (not dst_h2.exists()) or (src_h2.stat().st_mtime > dst_h2.stat().st_mtime):
            shutil.copy2(src_h2, dst_h2)
            if src_trace.exists():
                shutil.copy2(src_trace, dst_trace)

        return dst_h2

    @staticmethod
    def _h2_url_from_h2_file(h2_path: Path) -> str:
        s = str(h2_path)
        if s.endswith(".h2.db"):
            base = s[: -len(".h2.db")]
        else:
            base = str(h2_path.with_suffix(""))
        return f"jdbc:h2:{base};FILE_LOCK=NO"

    def _load_metadata_cache(self) -> dict[str, dict]:
        if not self.metadata_cache_path.exists():
            return {}
        return json.loads(self.metadata_cache_path.read_text(encoding="utf-8"))

    def _save_metadata_cache(self) -> None:
        self.metadata_cache_path.write_text(
            json.dumps(self.metadata_cache, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _query_function_meta(self, ids: list[int]) -> dict[int, FunctionMeta]:
        if not ids:
            return {}

        in_clause = ",".join(str(i) for i in ids)
        query = (
            "SELECT ID,PROJECT,TYPE,NAME,STARTLINE,ENDLINE "
            f"FROM FUNCTIONS WHERE ID IN ({in_clause})"
        )

        with tempfile.NamedTemporaryFile(prefix="bcb_functions_", suffix=".csv", delete=False) as tmp:
            csv_out = Path(tmp.name)

        sql = f"CALL CSVWRITE('{csv_out.as_posix()}', '{query}')"
        cmd = [
            "java",
            "-cp",
            str(self.cfg.bigclonebench_h2_jar.expanduser().resolve()),
            "org.h2.tools.Shell",
            "-url",
            self.db_url,
            "-user",
            "sa",
            "-password",
            "",
            "-sql",
            sql,
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )

        if proc.returncode != 0:
            csv_out.unlink(missing_ok=True)
            raise RuntimeError(
                "Failed querying BigCloneBench H2 DB. "
                f"stderr_tail={proc.stderr[-1000:]} stdout_tail={proc.stdout[-1000:]}"
            )

        out: dict[int, FunctionMeta] = {}
        try:
            with csv_out.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    function_id = int(row["ID"])
                    out[function_id] = FunctionMeta(
                        function_id=function_id,
                        project=row["PROJECT"],
                        type=row["TYPE"],
                        name=row["NAME"],
                        startline=int(row["STARTLINE"]),
                        endline=int(row["ENDLINE"]),
                    )
        finally:
            csv_out.unlink(missing_ok=True)

        return out

    def ensure_metadata(self, ids: set[int]) -> None:
        missing = [i for i in ids if str(i) not in self.metadata_cache]
        if not missing:
            return

        batch_size = max(1, int(self.cfg.bigclonebench_db_batch_size))
        for offset in range(0, len(missing), batch_size):
            batch = missing[offset : offset + batch_size]
            rows = self._query_function_meta(batch)
            for fid in batch:
                meta = rows.get(fid)
                self.metadata_cache[str(fid)] = (
                    {
                        "function_id": meta.function_id,
                        "project": meta.project,
                        "type": meta.type,
                        "name": meta.name,
                        "startline": meta.startline,
                        "endline": meta.endline,
                    }
                    if meta is not None
                    else {}
                )

        self._save_metadata_cache()

    def _meta(self, fid: int) -> Optional[FunctionMeta]:
        raw = self.metadata_cache.get(str(fid))
        if not raw:
            return None
        return FunctionMeta(
            function_id=int(raw["function_id"]),
            project=str(raw["project"]),
            type=str(raw["type"]),
            name=str(raw["name"]),
            startline=int(raw["startline"]),
            endline=int(raw["endline"]),
        )

    def _candidate_source_paths(self, meta: FunctionMeta) -> list[Path]:
        if self.source_root is None:
            return []
        root = self.source_root
        cands = [
            root / meta.type / meta.name,
            root / meta.project / meta.type / meta.name,
            root / meta.project / meta.name,
            root / meta.name,
        ]
        uniq: list[Path] = []
        seen = set()
        for cand in cands:
            key = str(cand)
            if key not in seen:
                seen.add(key)
                uniq.append(cand)
        return uniq

    def _resolve_source_file(self, meta: FunctionMeta) -> Optional[Path]:
        if self.source_root is None or not self.source_root.exists():
            return None

        for cand in self._candidate_source_paths(meta):
            if cand.exists() and cand.is_file():
                return cand.resolve()

        # Fallback: expensive recursive search by file name.
        for cand in self.source_root.rglob(meta.name):
            if cand.is_file():
                return cand.resolve()

        return None

    def resolve_fragment(self, fid: int) -> Optional[Path]:
        frag = self.fragments_dir / f"{fid}.java"
        if frag.exists():
            return frag.resolve()

        self.ensure_metadata({fid})
        meta = self._meta(fid)
        if meta is None:
            return None

        source_file = self._resolve_source_file(meta)
        if source_file is None:
            return None

        lines = source_file.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        start = max(1, meta.startline)
        end = max(start, meta.endline)

        if start <= len(lines):
            chunk = "".join(lines[start - 1 : min(end, len(lines))])
        else:
            chunk = source_file.read_text(encoding="utf-8", errors="replace")

        if not chunk.strip():
            chunk = source_file.read_text(encoding="utf-8", errors="replace")

        frag.write_text(chunk, encoding="utf-8")
        return frag.resolve()


class BigCloneBenchPairStream:
    """Stream BigCloneBench pair samples with train/val/test splits."""

    def __init__(self, cfg: DatasetConfig, seed: int = 0, cache_root: Optional[Path] = None):
        self.cfg = cfg
        self.seed = seed
        self.cache_root = (
            cache_root.expanduser().resolve() if cache_root is not None else Path("./cache").resolve()
        )

        self.allowed_clone_types = {
            t.strip() for t in cfg.clone_types if t.strip() and t.strip().lower() != "all"
        }
        self.accept_all_clone_types = len(self.allowed_clone_types) == 0

        self.rows = self._load_rows()
        self.split_rows = self._split_rows(self.rows)
        self.resolver = BigCloneBenchResolver(cfg, self.cache_root)

    def _resolve_split_counts(self, total: int) -> tuple[int, int, int]:
        train_size = self.cfg.train_size
        val_size = self.cfg.val_size
        test_size = self.cfg.test_size

        float_mode = all(v <= 1.0 for v in (train_size, val_size, test_size))
        if float_mode:
            ratio_sum = train_size + val_size + test_size
            if ratio_sum <= 0:
                raise ValueError("train_size + val_size + test_size must be > 0")
            train_n = int(round(total * train_size / ratio_sum))
            val_n = int(round(total * val_size / ratio_sum))
            test_n = total - train_n - val_n
            return train_n, val_n, test_n

        train_n = int(train_size)
        val_n = int(val_size)
        test_n = int(test_size)
        if train_n + val_n + test_n > total:
            raise ValueError("Requested split sizes exceed available rows")
        if train_n + val_n + test_n < total:
            train_n += total - (train_n + val_n + test_n)
        return train_n, val_n, test_n

    def _load_rows(self) -> list[BigCloneBenchPairRow]:
        labels_path = self.cfg.bigclonebench_labels_path.expanduser().resolve()
        if not labels_path.exists():
            raise FileNotFoundError(f"BigCloneBench labels file not found: {labels_path}")

        rows: list[BigCloneBenchPairRow] = []
        limit_pairs = self.cfg.bigclonebench_limit_pairs

        with labels_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 7:
                    continue

                try:
                    id1 = int(parts[0])
                    id2 = int(parts[1])
                    clone_type = parts[5]
                    score = float(parts[6])
                except ValueError:
                    continue

                label = 1 if score >= 0.5 else 0
                if label == 1:
                    if not self.accept_all_clone_types and clone_type not in self.allowed_clone_types:
                        continue
                else:
                    # For negatives, we only keep explicit non-clone labels.
                    if clone_type != "Non_Clone":
                        continue

                rows.append(
                    BigCloneBenchPairRow(
                        id1=id1,
                        id2=id2,
                        label=label,
                        clone_type=clone_type,
                    )
                )

                if limit_pairs is not None and len(rows) >= int(limit_pairs):
                    break

        if not rows:
            raise RuntimeError("No BigCloneBench rows selected. Check clone_types or labels file.")

        return rows

    def _split_rows(self, rows: list[BigCloneBenchPairRow]) -> dict[str, list[BigCloneBenchPairRow]]:
        rng = random.Random(self.seed)
        shuffled = rows[:]
        rng.shuffle(shuffled)

        train_n, val_n, test_n = self._resolve_split_counts(len(shuffled))
        train = shuffled[:train_n]
        val = shuffled[train_n : train_n + val_n]
        test = shuffled[train_n + val_n : train_n + val_n + test_n]

        return {"train": train, "val": val, "test": test}

    def split_summary(self) -> dict[str, int]:
        """Return number of pair rows in each split."""

        return {k: len(v) for k, v in self.split_rows.items()}

    def _split_pos_neg(self, split: SplitName) -> tuple[list[BigCloneBenchPairRow], list[BigCloneBenchPairRow]]:
        rows = self.split_rows[split]
        pos = [row for row in rows if row.label == 1]
        neg = [row for row in rows if row.label == 0]
        if not pos:
            raise RuntimeError(f"No positive rows in split '{split}'")
        if not neg:
            raise RuntimeError(f"No negative rows in split '{split}'")
        return pos, neg

    def _resolve_row(self, row: BigCloneBenchPairRow) -> Optional[Pair]:
        a = self.resolver.resolve_fragment(row.id1)
        b = self.resolver.resolve_fragment(row.id2)
        if a is None or b is None:
            return None
        return (str(a), str(b), int(row.label))

    def stream(self, split: SplitName, infinite: bool = True, seed_offset: int = 0) -> Iterator[Pair]:
        """Yield singleton pairs `(path_a, path_b, label)` for the selected split."""

        if self.resolver.source_root is None or not self.resolver.source_root.exists():
            raise RuntimeError(
                "BigCloneBench source root not found. Set "
                "'dataset.bigclonebench_source_root' to extracted IJaDataset sources."
            )

        pos_rows, neg_rows = self._split_pos_neg(split)
        rng = random.Random(self.seed + seed_offset + {"train": 0, "val": 1000, "test": 2000}[split])

        emitted = 0
        attempts = 0
        max_attempts = max(10000, len(self.split_rows[split]) * 20)
        while True:
            use_pos = rng.random() < self.cfg.positive_ratio
            row = rng.choice(pos_rows if use_pos else neg_rows)
            pair = self._resolve_row(row)
            attempts += 1

            if pair is not None:
                emitted += 1
                yield pair
                if not infinite and emitted >= max(1, len(self.split_rows[split])):
                    break

            if attempts >= max_attempts and emitted == 0:
                raise RuntimeError(
                    "Unable to resolve any BigCloneBench source pair to files. "
                    "Check bigclonebench_source_root and dataset extraction."
                )

    def sample_one(self, split: SplitName) -> Pair:
        """Convenience method to fetch one sample pair."""

        return next(self.stream(split=split, infinite=True))

    def pair_count_hint(self, split: SplitName) -> int:
        """Conservative split size hint."""

        return max(1, len(self.split_rows[split]))

    def all_sources(self) -> list[Path]:
        """Return all resolved fragment files referenced by selected rows."""

        if self.resolver.source_root is None or not self.resolver.source_root.exists():
            raise RuntimeError(
                "BigCloneBench source root not found. Set "
                "'dataset.bigclonebench_source_root' to extracted IJaDataset sources."
            )

        ids = {
            fid
            for rows in self.split_rows.values()
            for row in rows
            for fid in (row.id1, row.id2)
        }
        self.resolver.ensure_metadata(ids)

        out: list[Path] = []
        for fid in sorted(ids):
            frag = self.resolver.resolve_fragment(fid)
            if frag is not None:
                out.append(frag)
        return out
