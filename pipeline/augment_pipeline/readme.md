# augment_pipeline

Self-supervised pair generation for Java clone training.

## What it does
- Positive pair: (original_program.java, augmented_copy.java, label=1)
- Negative pair: (programA.java, programB.java, label=0)

The positive label is "same underlying program", not "same problem". This is for contrastive pretraining or weak supervision.

## CLI preview
```bash
python3 -m augment_pipeline.cli_preview \
  --root "../data/gcj_compiled" \
  --out "./aug_cache_gcj" \
  --limit-buckets 4 \
  --n 10 \
  --pos-ratio 0.5 \
  --seed 0
