# gnn_train_program

Program-level training wrapper.

Each dataset `.java` file is treated as a standalone “project”.
We encode each method graph with a small GNN, then mean-pool method embeddings to produce a program embedding.
Clone detection is trained on program embeddings.

Inputs:
- program_artifacts/program_index.json from `program_project`
- dataset folder (base/type-k) for generating pairs

Usage:
1) Build artifacts:
```bash
python3 -m program_project.cli \
  --dataset-root "../data/code-clone-dataset/dataset" \
  --clone-type type-3 \
  --out "./program_artifacts" \
  --jdk-home "$JAVA_HOME" \
  --vineflower "../chatgpt/vineflower-1.11.2.jar" \
  --model "microsoft/graphcodebert-base" \
  --device "mps"
```