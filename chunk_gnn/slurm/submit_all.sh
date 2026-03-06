#!/bin/bash
######################################################################
# submit_all.sh — Submit all 4 IDUN experiments in one go.
#
# Each experiment uses the same SLURM script with different configs.
# Results land in ~/chunk_gnn_out/{experiment_name}/
#
# Usage:
#   bash slurm/submit_all.sh
######################################################################

SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="${HOME}/dataset_loader/chunk_gnn/configs"

echo "Submitting 4 chunk-GNN experiments to IDUN..."
echo ""

# Exp 1: GCNConv baseline (reproduce Run 7 on A100)
CONFIG="${CONFIG_DIR}/bcb_classifier.json" \
EXPERIMENT=exp1_gcn_baseline \
sbatch "${SLURM_DIR}/train.slurm"

# Exp 2: GATConv (learned attention weights)
CONFIG="${CONFIG_DIR}/bcb_gat_classifier.json" \
EXPERIMENT=exp2_gat \
sbatch "${SLURM_DIR}/train.slurm"

# Exp 3: Deeper GCNConv (3 layers + residual)
CONFIG="${CONFIG_DIR}/bcb_gcn_deep.json" \
EXPERIMENT=exp3_gcn_deep \
sbatch "${SLURM_DIR}/train.slurm"

# Exp 4: RGCNConv (edge-type-aware message passing)
CONFIG="${CONFIG_DIR}/bcb_rgcn_classifier.json" \
EXPERIMENT=exp4_rgcn \
sbatch "${SLURM_DIR}/train.slurm"

echo ""
echo "Submitted 4 jobs. Check queue: squeue -u \$USER"
