#!/bin/bash
######################################################################
# optuna.slurm — Optuna hyperparameter optimization for GATConv
#
# Runs N_TRIALS Optuna trials, each training a GATConv model with
# different hyperparameters. Uses SQLite storage for crash recovery.
#
# Usage:
#   sbatch slurm/optuna.slurm
#   N_TRIALS=30 sbatch slurm/optuna.slurm
######################################################################

#SBATCH --job-name=optuna-gat
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|a100|h100"
#SBATCH --output=optuna_gat_%j.out
#SBATCH --error=optuna_gat_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=skjalgct@ntnu.no

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VENV_DIR="${HOME}/magnet-venv"
BCB_ROOT="${HOME}/Multigraph_match_optimized/data/data_source/dataset_bigclonebench"
CACHE_DIR="${HOME}/chunk_gnn_cache"
OUTPUT_DIR="${HOME}/chunk_gnn_out/optuna_gat"
WORK_DIR="${HOME}/dataset_loader"
N_TRIALS="${N_TRIALS:-50}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log()       { echo "[$(timestamp)] $*"; }
die()       { echo "[$(timestamp)] FATAL: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Job info
# ---------------------------------------------------------------------------
log "=== Optuna Hyperparameter Optimization ==="
log "Job ID:     ${SLURM_JOB_ID}"
log "Node:       $(hostname)"
log "N_TRIALS:   ${N_TRIALS}"
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
# Step 2: Install optuna if needed
# ---------------------------------------------------------------------------
log "Step 2: Checking/installing optuna"
python3 -c "import optuna" 2>/dev/null || {
    log "  Installing optuna..."
    pip install --quiet optuna
}
python3 -c "import optuna; print(f'  optuna {optuna.__version__}')"

# ---------------------------------------------------------------------------
# Step 3: Pre-flight checks
# ---------------------------------------------------------------------------
log ""
log "Step 3: Pre-flight checks"
python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA'" || die "CUDA not available"
python3 -c "import torch_geometric" || die "torch_geometric not importable"

[[ -d "${CACHE_DIR}" ]] || die "Cache dir not found: ${CACHE_DIR}"
CACHED_FILES=$(ls "${CACHE_DIR}"/*.pt 2>/dev/null | wc -l)
log "  Cached graph files: ${CACHED_FILES}"
[[ "${CACHED_FILES}" -gt 0 ]] || die "No .pt files in cache dir"

log "  Pre-flight: ALL PASSED"

# ---------------------------------------------------------------------------
# Step 4: GPU info
# ---------------------------------------------------------------------------
log ""
log "Step 4: GPU Information"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
log ""

# ---------------------------------------------------------------------------
# Step 5: Run Optuna
# ---------------------------------------------------------------------------
log "Step 5: Starting Optuna search (${N_TRIALS} trials)"
log "  Output dir: ${OUTPUT_DIR}"
log ""

cd "${WORK_DIR}"

python3 -u chunk_gnn/scripts/07_optuna.py \
    --bcb_root "${BCB_ROOT}" \
    --cache_dir "${CACHE_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --device cuda \
    --num_workers 4 \
    --n_trials "${N_TRIALS}" \
    2>&1

EXIT_CODE=$?

log ""
if [[ "${EXIT_CODE}" -eq 0 ]]; then
    log "=== Optuna search completed successfully ==="
else
    log "=== Optuna search FAILED with exit code ${EXIT_CODE} ==="
fi

# Show summary if exists
if [[ -f "${OUTPUT_DIR}/optuna_summary.json" ]]; then
    log ""
    log "Summary:"
    python3 -c "
import json
with open('${OUTPUT_DIR}/optuna_summary.json') as f:
    s = json.load(f)
print(f'  Best trial: {s[\"best_trial\"]}')
print(f'  Best val F1: {s[\"best_val_f1\"]:.4f}')
print(f'  Best params:')
for k, v in s['best_params'].items():
    print(f'    {k}: {v}')
"
fi

log ""
log "=== Job finished at $(timestamp) ==="
exit ${EXIT_CODE}
