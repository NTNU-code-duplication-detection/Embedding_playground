# pair_dataset

Generates positive and negative (anchor, other, label) pairs from the on-disk dataset:

data/code-clone-dataset/dataset/
  base/<idx>/main.java
  type-1/<idx>/*.java
  type-2/<idx>/*.java
  type-3/<idx>/*.java

Where `base/<idx>/main.java` is the anchor program and each `type-k/<idx>/*.java` contains clones of that anchor.

## Generators

### Positive pairs
`positive_pairs(cfg)` yields:
(anchor_path, clone_path, 1)
where clone_path comes from the same index under cfg.clone_type.

### Negative pairs
`negative_pairs(cfg)` yields:
(anchor_path, nonclone_path, 0)
where nonclone_path is sampled from a different index.

Negative sampling pools:
- `same_clone_type` (default): sample from other indices inside `type-k/`
- `base`: sample other anchors in `base/`

## Preview
```bash
python3 -m pair_dataset.cli_preview \
  --root "./data/code-clone-dataset/dataset" \
  --clone-type type-3 \
  --neg-pool same_clone_type \
  --n 5
```