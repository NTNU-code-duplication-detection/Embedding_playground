"""
CLI module
"""
#!/usr/bin/env python3
from __future__ import annotations

from typing import List, Optional, Set, Tuple
import argparse
from pathlib import Path

import random
import torch

from pair_dataset.config import PairDatasetConfig
from pair_dataset.generators import positive_pairs, negative_pairs
from gnn_train_program.program_store import ProgramStore
from gnn_train_program.train import train_loop_program
from gnn_train_program.split import extract_idx_from_path, make_index_sets


def filter_pairs_by_index(pair_iter, allowed_idxs: set[str], *, clone_type: str, dataset_root: Path):
    """Filter an infinite (a,b,y) stream so that the *anchor* (a) belongs to allowed_idxs."""
    while True:
        a, b, y = next(pair_iter)

        if clone_type == "googlejam":
            ia = extract_gj_idx(a, dataset_root)
        else:
            ia = extract_idx_from_path(a)

        # Anchor determines the "instance". If we can't parse index, skip.
        if ia is None:
            continue

        if ia in allowed_idxs:
            yield (a, b, y)


def interleave(pos_iter, neg_iter, pos_ratio: float = 0.5, seed: int = 0):
    """
    Yields from pos/neg iterators with a given ratio.
    """
    rng = random.Random(seed)
    while True:
        if rng.random() < pos_ratio:
            yield next(pos_iter)
        else:
            yield next(neg_iter)


def list_googlejam_indices(root: Path) -> List[str]:
    # indices are top-level directory names under root (e.g. 1..12)
    out: List[str] = []
    if not root.exists():
        return out
    for p in root.iterdir():
        if p.is_dir() and p.name.isdigit():
            out.append(p.name)
    return sorted(out, key=lambda s: int(s))


def extract_gj_idx(path: str, dataset_root: Path) -> Optional[str]:
    """Return the top-level bucket (e.g. '1'..'12') for a file under dataset_root."""
    try:
        rel = Path(path).resolve().relative_to(dataset_root.resolve())
    except Exception:
        return None

    parts = rel.parts
    if not parts:
        return None

    first = parts[0]
    return first if first.isdigit() else None


def make_index_sets_googlejam(dataset_root: Path, limit_indices: Optional[int], val_ratio: float, seed: int = 0) -> Tuple[Set[str], Set[str]]:
    idxs = list_googlejam_indices(dataset_root)
    if limit_indices is not None:
        idxs = idxs[: int(limit_indices)]
    rng = random.Random(seed)
    rng.shuffle(idxs)
    n_val = max(1, int(round(len(idxs) * float(val_ratio)))) if idxs else 0
    val = set(idxs[:n_val])
    train = set(idxs[n_val:])
    # guard: if split collapses, put at least 1 in train when possible
    if not train and idxs:
        train = set(idxs)
        val = set()
    return train, val


def iter_java_files_under(root: Path) -> List[Path]:
    # collect all .java files recursively under root
    files: List[Path] = []
    if not root.exists():
        return files
    for p in root.rglob("*.java"):
        if p.is_file():
            files.append(p)
    return files


def googlejam_positive_pairs(dataset_root: Path, *, infinite: bool = True, seed: int = 0, limit_indices: Optional[int] = None):
    rng = random.Random(seed)
    idxs = list_googlejam_indices(dataset_root)
    if limit_indices is not None:
        idxs = idxs[: int(limit_indices)]

    # pre-index files per idx for speed
    files_by_idx = {i: iter_java_files_under((dataset_root / i).resolve()) for i in idxs}

    while True:
        if not idxs:
            raise StopIteration
        i = rng.choice(idxs)
        files = files_by_idx.get(i, [])
        if len(files) < 2:
            if not infinite:
                raise StopIteration
            continue
        a, b = rng.sample(files, 2)
        yield (str(a.resolve()), str(b.resolve()), 1)
        if not infinite:
            return


def googlejam_negative_pairs(dataset_root: Path, *, infinite: bool = True, seed: int = 0, limit_indices: Optional[int] = None):
    rng = random.Random(seed)
    idxs = list_googlejam_indices(dataset_root)
    if limit_indices is not None:
        idxs = idxs[: int(limit_indices)]

    files_by_idx = {i: iter_java_files_under((dataset_root / i).resolve()) for i in idxs}

    while True:
        if len(idxs) < 2:
            raise StopIteration
        i, j = rng.sample(idxs, 2)
        fi = files_by_idx.get(i, [])
        fj = files_by_idx.get(j, [])
        if not fi or not fj:
            if not infinite:
                raise StopIteration
            continue
        a = rng.choice(fi)
        b = rng.choice(fj)
        yield (str(a.resolve()), str(b.resolve()), 0)
        if not infinite:
            return


def main() -> int:
    """
    Main entry point
    """
    ap = argparse.ArgumentParser("Train program-level clone detector")
    ap.add_argument("--program-index", required=True, help="program_artifacts/program_index.json")
    ap.add_argument("--dataset-root", required=True, help=".../data/code-clone-dataset/dataset")
    ap.add_argument("--clone-type", default="type-3", choices=["type-1", "type-2", "type-3", "googlejam"])
    ap.add_argument("--neg-pool", default="same_clone_type", choices=["same_clone_type", "base"])

    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-pairs", type=int, default=32)
    ap.add_argument("--pos-ratio", type=float, default=0.5)

    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--in-dim", type=int, default=768)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--limit-indices", type=int, default=None)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--val-pairs", type=int, default=200)

    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()

    if args.clone_type == "googlejam":
        train_idxs, val_idxs = make_index_sets_googlejam(dataset_root, args.limit_indices, args.val_ratio, seed=args.seed)

        pos = googlejam_positive_pairs(dataset_root, infinite=True, seed=args.seed, limit_indices=args.limit_indices)
        neg = googlejam_negative_pairs(dataset_root, infinite=True, seed=args.seed + 123, limit_indices=args.limit_indices)

        all_pairs = interleave(pos, neg, pos_ratio=args.pos_ratio, seed=args.seed)
        train_iter = filter_pairs_by_index(all_pairs, train_idxs, clone_type=args.clone_type, dataset_root=dataset_root)

        pos_v = googlejam_positive_pairs(dataset_root, infinite=True, seed=args.seed + 999, limit_indices=args.limit_indices)
        neg_v = googlejam_negative_pairs(dataset_root, infinite=True, seed=args.seed + 999 + 123, limit_indices=args.limit_indices)
        all_pairs_v = interleave(pos_v, neg_v, pos_ratio=args.pos_ratio, seed=args.seed + 999)
        val_iter = filter_pairs_by_index(all_pairs_v, val_idxs, clone_type=args.clone_type, dataset_root=dataset_root)

    else:
        # Pair generators yield (anchor_path, other_path, label)
        ds_cfg = PairDatasetConfig(
            root=dataset_root,
            clone_type=args.clone_type,
            negative_pool=args.neg_pool,
            seed=args.seed,
            limit_indices=args.limit_indices,
        )

        train_idxs, val_idxs = make_index_sets(args.limit_indices, args.val_ratio, seed=args.seed)

        pos = positive_pairs(ds_cfg, infinite=True)
        neg = negative_pairs(ds_cfg, infinite=True)
        all_pairs = interleave(pos, neg, pos_ratio=args.pos_ratio, seed=args.seed)

        train_iter = filter_pairs_by_index(all_pairs, train_idxs, clone_type=args.clone_type, dataset_root=dataset_root)

        # Separate val stream (fresh generators, same config/seed offset)
        pos_v = positive_pairs(ds_cfg, infinite=True)
        neg_v = negative_pairs(ds_cfg, infinite=True)
        all_pairs_v = interleave(pos_v, neg_v, pos_ratio=args.pos_ratio, seed=args.seed + 999)
        val_iter = filter_pairs_by_index(all_pairs_v, val_idxs, clone_type=args.clone_type, dataset_root=dataset_root)

    store = ProgramStore(Path(args.program_index).expanduser().resolve(), max_cached=64)

    enc, clf = train_loop_program(
        pair_iter=train_iter,
        store=store,
        device=args.device,
        in_dim=args.in_dim,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        steps=args.steps,
        batch_pairs=args.batch_pairs,
        lr=args.lr,
        weight_decay=args.wd,
        dropout=args.dropout,
        log_every=args.log_every,
        seed=args.seed,
        eval_every=args.eval_every,
        val_iter=val_iter,
        val_pairs=args.val_pairs,
    )

    out_dir = Path("./gnn_models_program").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(enc.state_dict(), out_dir / "method_encoder.pt")
    torch.save(clf.state_dict(), out_dir / "pair_classifier.pt")

    print(f"Saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
