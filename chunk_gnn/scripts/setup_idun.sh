#!/bin/bash
# Setup script for chunk_gnn on IDUN.
# Reuses the existing magnet-venv, just installs missing dependencies.
#
# Usage (on IDUN):
#   bash ~/dataset_loader/chunk_gnn/scripts/setup_idun.sh

set -euo pipefail

VENV_DIR="${HOME}/magnet-venv"

echo "=== Chunk-GNN Setup on IDUN ==="
echo "Reusing existing venv: ${VENV_DIR}"

# Load Python module and activate venv
module purge
module load Python/3.11.3-GCCcore-12.3.0
source "${VENV_DIR}/bin/activate" || { echo "FATAL: venv not found"; exit 1; }

echo "Python: $(python3 --version)"

# Install missing dependencies (tree-sitter-java and h5py)
echo ""
echo "Installing additional dependencies..."
pip install tree-sitter tree-sitter-java h5py

# Verify all required packages
echo ""
echo "Verifying dependencies..."
python3 -c "
import torch; print(f'  torch: {torch.__version__}')
import torch_geometric; print(f'  torch_geometric: {torch_geometric.__version__}')
from torch_geometric.nn import GCNConv, global_mean_pool; print('  GCNConv + global_mean_pool: OK')
from torch_geometric.data import Data, Batch; print('  Data + Batch: OK')
import transformers; print(f'  transformers: {transformers.__version__}')
import tree_sitter; print(f'  tree_sitter: OK')
import tree_sitter_java; print(f'  tree_sitter_java: OK')
import h5py; print(f'  h5py: {h5py.__version__}')
from sklearn.metrics import f1_score, precision_score, recall_score; print('  sklearn metrics: OK')
import numpy; print(f'  numpy: {numpy.__version__}')
print()
print('All dependencies verified!')
"

echo ""
echo "=== Setup complete ==="
