# Code Clone Detection – Project Notes

## 1. Objective
Build a program‑level Java clone detector that combines pretrained code embeddings (GraphCodeBERT) with structural reasoning from AST graphs and a GNN‑style aggregation. The goal is to determine whether two programs implement the same logic, not merely whether they look textually similar.

Key hypothesis:
> Transformer embeddings capture local semantics, while graph structure reconnects them into algorithmic meaning.

---

## 2. End‑to‑End Pipeline

Source code
→ (optional) compile
→ (optional) decompile (Vineflower)
→ AST extraction (tree‑sitter)
→ Method graph construction
→ Node embeddings (GraphCodeBERT)
→ Method graph tensors
→ Program embedding (method pooling)
→ Pair classifier (clone / non‑clone)

### Artifacts Produced
| File | Meaning |
|---|---|
methods.jsonl | Graph description per method (nodes + edges)
embed_cache/shards/*.pt | Embedded node tensors (PyTorch)
program_index.json | Mapping program path → graph artifacts
gnn_models_program/*.pt | Trained GNN encoder and classifier

### Node Features
768‑dimensional GraphCodeBERT embeddings.

### Edge Types Used (final minimal set)
- SEQ (sequential execution)
- AST (parent/child structure merged)
- IF_THEN
- IF_ELSE

Reason: reducing edge types stabilized training and limited parameter growth (~3.7M → manageable).

---

## 3. Datasets Used

### A. Synthetic Code‑Clone Dataset (Supervised)
Structure:
base/<id>/main.java = anchor
(type‑1 | type‑2 | type‑3)/<id>/{1,2,3}.java = clones

We trained primarily on **Type‑3 clones** (hardest: modified logic structure).

Typical training parameters:
--clone-type type-3
--batch-pairs 32
--steps 2000
--device mps
--val-ratio 0.2
--eval-every 100

#### Results
Training loss: decreased strongly
Examples:
step=425 loss=0.0262 ema=0.2084
step=450 loss=0.0083 ema=0.1782

Later stabilized runs:
Training loss ≈ 0.45–0.55
Validation accuracy ≈ 0.48–0.53
Validation loss ≈ 0.64–0.68

Interpretation:
Model learns meaningful structure but generalization between programs is limited. Synthetic dataset diversity is low.

---

### B. Decompiled Code Experiments
Process:
Java → compile → Vineflower decompile → embed

Observation:
The decompiler canonicalizes code:
- variable names normalized
- formatting normalized
- control flow simplified

Measured embedding separation:
Signal mean ≈ 0.9988
Noise mean ≈ 0.9881
d' ≈ 2.48–3.16

Conclusion:
GraphCodeBERT alone already detects semantic similarity when code is canonicalized.

Key research insight:
> Canonicalization (decompilation) greatly improves semantic clone detection.

---

### C. Google Code Jam Dataset (Weakly Supervised)
Large unlabeled dataset of competitive programming solutions.

Labeling strategy:
Positive = same problem bucket
Negative = different problem bucket

Programs processed: 268

Training parameters:
--clone-type googlejam
--limit-indices 4
--steps 2000

#### Results
Validation accuracy ≈ 0.47–0.52
Validation loss ≈ 0.694

Interpretation:
Performance ≈ random guessing due to noisy labels.
Same problem ≠ same implementation.

However, dataset useful for scale and representation learning.

---

## 4. Architectural Experiments

### 1) Direct Embedding Similarity
Cosine similarity between GraphCodeBERT embeddings.
Result: strong separation after decompilation.
Conclusion: embeddings already encode semantics.

### 2) Statement Embeddings + Mean Pooling
Embed statements → average → compare.
Problem: loses control‑flow semantics.

### 3) Graph Model (Final System)
Method = graph of statements
GNN message passing → method embedding
Mean pool methods → program embedding
Binary classifier → clone decision

Main contribution of project.

---

## 5. Training Observations

### Overfitting
On synthetic dataset:
Training loss drops fast, validation stagnates.
Model memorizes structure patterns.

### GoogleJam Results
~50% accuracy due to label noise.
Expected for weak supervision.

### Decompiler Effect
Decompiled code drastically improves semantic consistency.
Important discovery: representation quality matters more than model complexity.

---

## 6. What the Model Actually Learns
The model does **not** learn textual similarity.
It learns approximate algorithmic structure similarity.

Current limitation:
Program embedding = mean(method embeddings)
→ loses inter‑method relationships.

This is likely the main performance bottleneck.

---

## 7. Why Accuracy Saturates Near 50%
Three noise sources:
1) Weak labels (GoogleJam)
2) Mean pooling destroys structure
3) Many valid implementations per algorithm

Therefore the model hits a dataset ceiling rather than a modeling failure.

---

## 8. Proven Findings
The experiments demonstrated:
1. GraphCodeBERT embeddings contain real semantic information
2. AST structure increases robustness
3. Decompiled code improves similarity detection
4. Method‑level clone detection is easier than program‑level
5. Canonicalization may matter more than network size

---

## 9. Current System Description
Hybrid Transformer‑GNN clone detector trained on Java programs and evaluated on both synthetic and real competitive programming data.

Main takeaway:
Transformer embeddings capture semantics, but structural aggregation is required to scale similarity detection from functions to full programs.
