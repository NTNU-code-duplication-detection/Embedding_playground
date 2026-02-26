#!/bin/bash
######################################################################
# precompute.slurm — Pre-compute chunk graphs for all BCB functions
#
# Runs UniXcoder inference on all BCB functions to produce cached
# graph objects. One-time cost, supports resuming.
#
# Usage:
#   sbatch precompute.slurm
######################################################################

#SBATCH --job-name=chunk-precomp
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=0-08:00:00       # 8 hours (generous, likely ~1-2h)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1        # A100 (same as MagNET for consistency)
#SBATCH --output=precompute_%j.out
#SBATCH --error=precompute_%j.err

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VENV_DIR="${HOME}/magnet-venv"
BCB_ROOT="${HOME}/Multigraph_match_optimized/data/data_source/dataset_bigclonebench"
CACHE_DIR="${HOME}/chunk_gnn_cache"
SCRIPT_DIR="${HOME}/dataset_loader/chunk_gnn"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Chunk-GNN Pre-compute ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"

module purge
module load Python/3.11.3-GCCcore-12.3.0
source "${VENV_DIR}/bin/activate" || { echo "FATAL: venv not found"; exit 1; }

echo "Python: $(python3 --version)"
echo "BCB root: ${BCB_ROOT}"
echo "Cache dir: ${CACHE_DIR}"
nvidia-smi || echo "WARNING: nvidia-smi failed"
echo ""

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo "Pre-flight checks..."
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" \
    || { echo "FATAL: CUDA not available"; exit 1; }
python3 -c "import tree_sitter_java" \
    || { echo "FATAL: tree_sitter_java not installed"; exit 1; }
python3 -c "import transformers" \
    || { echo "FATAL: transformers not installed"; exit 1; }
[[ -f "${BCB_ROOT}/clone_labels.txt" ]] \
    || { echo "FATAL: clone_labels.txt not found at ${BCB_ROOT}"; exit 1; }
echo "  Pre-flight: ALL PASSED"
echo ""

# ---------------------------------------------------------------------------
# Run pre-compute
# ---------------------------------------------------------------------------
cd "${HOME}/dataset_loader"

python3 -u chunk_gnn/scripts/02_precompute_graphs.py \
    --bcb_root "${BCB_ROOT}" \
    --cache_dir "${CACHE_DIR}" \
    --device cuda \
    --batch_size 32 \
    --max_chunks 50 \
    --log_interval 500

EXIT_CODE=$?

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Exit code: ${EXIT_CODE}"
echo "Cache contents:"
ls -lh "${CACHE_DIR}" 2>/dev/null | head -20
echo "Total cache size:"
du -sh "${CACHE_DIR}" 2>/dev/null

exit ${EXIT_CODE}
