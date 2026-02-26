# program_project

Builds program-level artifacts for the code-clone dataset where each `.java` file is treated as a standalone “project”.

Dataset layout:
- base/<idx>/main.java  (anchor)
- type-1/<idx>/*.java, type-2/<idx>/*.java, type-3/<idx>/*.java  (clones)

This stage:
1) Copies a program into a deterministic workdir
2) Compiles with `javac` (JDK21)
3) Packages `program.jar`
4) Runs your existing stages:
   - `decompiler.cli` (Vineflower)
   - `ast_dataset.cli` -> methods.jsonl
   - `embed_cache.cli` -> embed_cache/shards/*.pt
5) Writes `program_index.json` mapping source_path -> artifact_dir + shards_dir

## Usage
From `pipeline/`:
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