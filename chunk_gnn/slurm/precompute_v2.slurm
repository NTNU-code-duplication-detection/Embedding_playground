#!/bin/bash
######################################################################
# precompute_v2.slurm — Rebuild chunk graph cache with data flow edges
#
# Rebuilds the graph cache with:
#   - Proper edge type storage (eliminates heuristic inference)
#   - Data flow edges (variable def-use tracking across chunks)
#   - 4 edge types: self-loop(0), sequential(1), parent-child(2), data-flow(3)
#
# Saves to a NEW cache dir (chunk_gnn_cache_v2) — old cache preserved.
#
# Usage:
#   sbatch slurm/precompute_v2.slurm
######################################################################

#SBATCH --job-name=precomp-v2
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|a100|h100"
#SBATCH --output=precompute_v2_%j.out
#SBATCH --error=precompute_v2_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=skjalgct@ntnu.no

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VENV_DIR="${HOME}/magnet-venv"
BCB_ROOT="${HOME}/Multigraph_match_optimized/data/data_source/dataset_bigclonebench"
CACHE_DIR="${HOME}/chunk_gnn_cache_v2"
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
log "=== Chunk-GNN Cache Rebuild v2 (Data Flow Edges) ==="
log "Job ID:     ${SLURM_JOB_ID}"
log "Node:       $(hostname)"
log "Cache dir:  ${CACHE_DIR} (NEW — old cache preserved)"
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
python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA'" || die "CUDA not available"
python3 -c "import tree_sitter_java" || die "tree_sitter_java not installed"
python3 -c "import transformers" || die "transformers not installed"
python3 -c "import torch_geometric" || die "torch_geometric not installed"
[[ -f "${BCB_ROOT}/clone_labels.txt" ]] || die "clone_labels.txt not found"

# Verify the updated code has data flow support
python3 -c "
from chunk_gnn.data.graph_builder import EDGE_DATA_FLOW, NUM_EDGE_TYPES
assert NUM_EDGE_TYPES == 4, f'Expected 4 edge types, got {NUM_EDGE_TYPES}'
print(f'  Edge types: {NUM_EDGE_TYPES} (including DATA_FLOW={EDGE_DATA_FLOW})')
" || die "Updated graph_builder not found — did you SCP the new code?"

python3 -c "
from chunk_gnn.data.chunker import Chunk
import dataclasses
fields = {f.name for f in dataclasses.fields(Chunk)}
assert 'variables' in fields, 'Chunk.variables field not found'
print('  Chunk dataclass has variables field: OK')
" || die "Updated chunker not found — did you SCP the new code?"

log "  Pre-flight: ALL PASSED"

# ---------------------------------------------------------------------------
# Step 3: GPU info
# ---------------------------------------------------------------------------
log ""
log "Step 3: GPU Information"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
log ""

# ---------------------------------------------------------------------------
# Step 4: Rebuild cache
# ---------------------------------------------------------------------------
log "Step 4: Rebuilding cache with data flow edges"
log "  Output: ${CACHE_DIR}"
log ""

cd "${WORK_DIR}"

python3 -u chunk_gnn/scripts/02_precompute_graphs.py \
    --bcb_root "${BCB_ROOT}" \
    --cache_dir "${CACHE_DIR}" \
    --device cuda \
    --batch_size 32 \
    --max_chunks 50 \
    --log_interval 500

EXIT_CODE=$?

# ---------------------------------------------------------------------------
# Step 5: Verify output
# ---------------------------------------------------------------------------
log ""
if [[ "${EXIT_CODE}" -eq 0 ]]; then
    log "=== Cache rebuild completed successfully ==="
else
    log "=== Cache rebuild FAILED with exit code ${EXIT_CODE} ==="
fi

log ""
log "Cache contents:"
PT_COUNT=$(ls "${CACHE_DIR}"/*.pt 2>/dev/null | wc -l)
EMPTY_COUNT=$(ls "${CACHE_DIR}"/*.empty 2>/dev/null | wc -l)
log "  .pt files:    ${PT_COUNT}"
log "  .empty files:  ${EMPTY_COUNT}"
log "  Total size:    $(du -sh "${CACHE_DIR}" 2>/dev/null | cut -f1)"

# Spot-check: verify a random .pt file has edge_type
if [[ "${PT_COUNT}" -gt 0 ]]; then
    SAMPLE=$(ls "${CACHE_DIR}"/*.pt 2>/dev/null | head -1)
    python3 -c "
import torch
data = torch.load('${SAMPLE}', weights_only=False, map_location='cpu')
assert hasattr(data, 'edge_type') and data.edge_type is not None, 'edge_type missing!'
n_edges = data.edge_index.size(1)
types = data.edge_type.unique().tolist()
df_count = (data.edge_type == 3).sum().item()
print(f'  Sample graph: {data.num_nodes} nodes, {n_edges} edges, types={types}, data_flow={df_count}')
" || log "WARNING: spot-check failed"
fi

log ""
log "=== Job finished at $(timestamp) ==="
exit ${EXIT_CODE}
