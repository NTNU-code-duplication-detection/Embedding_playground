#!/bin/bash
######################################################################
# train.slurm — Train the chunk-level GNN on BigCloneBench
#
# Requires pre-computed graphs (run precompute.slurm first).
#
# Usage:
#   sbatch train.slurm
######################################################################

#SBATCH --job-name=chunk-gnn
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=3-00:00:00       # 3 days (generous, likely much faster)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5        # 4 DataLoader workers + 1 main process
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1        # A100 (same as MagNET for consistency)
#SBATCH --output=chunk_gnn_%j.out
#SBATCH --error=chunk_gnn_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=skjalgct@ntnu.no

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VENV_DIR="${HOME}/magnet-venv"
BCB_ROOT="${HOME}/Multigraph_match_optimized/data/data_source/dataset_bigclonebench"
CACHE_DIR="${HOME}/chunk_gnn_cache"
OUTPUT_DIR="${HOME}/chunk_gnn_out"
CONFIG="${HOME}/dataset_loader/chunk_gnn/configs/bcb_mvp.json"
WORK_DIR="${HOME}/dataset_loader"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log()       { echo "[$(timestamp)] $*"; }
die()       { echo "[$(timestamp)] FATAL: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Job info
# ---------------------------------------------------------------------------
log "=== Chunk-GNN Training ==="
log "Job ID:     ${SLURM_JOB_ID}"
log "Job name:   ${SLURM_JOB_NAME}"
log "Node:       $(hostname)"
log "Partition:  ${SLURM_JOB_PARTITION}"
log "GPUs:       ${SLURM_GPUS_ON_NODE:-unknown}"
log "CPUs:       ${SLURM_CPUS_PER_TASK}"
log "Memory:     ${SLURM_MEM_PER_NODE} MB"
log ""

# ---------------------------------------------------------------------------
# Step 1: Load modules and activate venv
# ---------------------------------------------------------------------------
log "Step 1: Loading modules and activating environment"
module purge
module load Python/3.11.3-GCCcore-12.3.0
source "${VENV_DIR}/bin/activate" || die "Failed to activate venv"
log "  Python: $(python3 --version 2>&1)"

# ---------------------------------------------------------------------------
# Step 2: Pre-flight checks
# ---------------------------------------------------------------------------
log ""
log "Step 2: Pre-flight checks"

python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" \
    || die "CUDA not available"
python3 -c "import torch_geometric" \
    || die "torch_geometric not importable"
python3 -c "import tree_sitter_java" \
    || die "tree_sitter_java not importable"

[[ -d "${CACHE_DIR}" ]] || die "Cache dir not found: ${CACHE_DIR}"
CACHED_FILES=$(ls "${CACHE_DIR}"/*.pt 2>/dev/null | wc -l)
log "  Cached graph files: ${CACHED_FILES}"
[[ "${CACHED_FILES}" -gt 0 ]] || die "No .pt files in cache dir"

[[ -f "${CONFIG}" ]] || die "Config not found: ${CONFIG}"
[[ -f "${BCB_ROOT}/clone_labels.txt" ]] || die "clone_labels.txt not found"

log "  Pre-flight: ALL PASSED"

# ---------------------------------------------------------------------------
# Step 3: Show GPU info
# ---------------------------------------------------------------------------
log ""
log "Step 3: GPU Information"
nvidia-smi || log "WARNING: nvidia-smi failed"
log ""

# ---------------------------------------------------------------------------
# Step 4: Train
# ---------------------------------------------------------------------------
log "Step 4: Starting training"
log "  Config:     ${CONFIG}"
log "  BCB root:   ${BCB_ROOT}"
log "  Cache dir:  ${CACHE_DIR}"
log "  Output dir: ${OUTPUT_DIR}"
log ""

cd "${WORK_DIR}"

python3 -u chunk_gnn/scripts/03_train.py \
    --config "${CONFIG}" \
    --bcb_root "${BCB_ROOT}" \
    --cache_dir "${CACHE_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --device cuda \
    --num_workers 4 \
    2>&1

TRAIN_EXIT=$?

log ""
if [[ "${TRAIN_EXIT}" -eq 0 ]]; then
    log "=== Training completed successfully ==="
else
    log "=== Training FAILED with exit code ${TRAIN_EXIT} ==="
fi

# ---------------------------------------------------------------------------
# Step 5: Show output
# ---------------------------------------------------------------------------
log ""
log "Step 5: Output files"
log "  Output dir: ${OUTPUT_DIR}"

if [[ -d "${OUTPUT_DIR}" ]]; then
    LATEST=$(ls -td "${OUTPUT_DIR}"/run_* 2>/dev/null | head -1)
    if [[ -n "${LATEST}" ]]; then
        log "  Latest run: ${LATEST}"
        log "  Contents:"
        find "${LATEST}" -type f | while read -r line; do
            log "    ${line}"
        done
        if [[ -f "${LATEST}/results.json" ]]; then
            log ""
            log "  Results summary:"
            python3 -c "
import json
with open('${LATEST}/results.json') as f:
    r = json.load(f)
print(f'    Best epoch: {r[\"best_epoch\"]}')
print(f'    Best val F1: {r[\"best_val_f1\"]:.4f}')
"
        fi
    fi
fi

log ""
log "SLURM output: chunk_gnn_${SLURM_JOB_ID}.out"
log "SLURM errors: chunk_gnn_${SLURM_JOB_ID}.err"
log ""
log "=== Job finished at $(timestamp) ==="

exit ${TRAIN_EXIT}
