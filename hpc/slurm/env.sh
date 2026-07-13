#!/usr/bin/env bash
# Cluster-specific RUNTIME environment for the Slurm jobs. Edit this ONCE for
# your cluster; every hpc/slurm/*.slurm script sources it. (Slurm resource
# directives like partition/account are NOT here -- they must be #SBATCH lines
# or sbatch CLI flags; see the .slurm headers.)
set -euo pipefail

# --- 1. load a conda/python that has the `dasfwi` env -----------------------
# Option A (module + conda), typical on HPC:
#   module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true
#   source "$(conda info --base)/etc/profile.d/conda.sh"
#   conda activate "${CONDA_ENV:-dasfwi}"
#   PYTHON_BIN=python
# Option B (absolute interpreter path, no activation):
#   PYTHON_BIN=/path/to/envs/dasfwi/bin/python
#
# Default: try conda, fall back to whatever `python` is on PATH.
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${CONDA_ENV:-dasfwi}"
    fi
    PYTHON_BIN=python
fi
export PYTHON_BIN

# --- 2. paths (only if the side-by-side layout is not used) -----------------
# The repo is self-contained via ADFWI_local/; set these only to override.
# export ADFWI_ROOT=/path/to/DASFWI/ADFWI_local
# export MARMOUSI_DIR=/path/to/Data_downloads/marmousi2
# export DASFWI_RESULTS=/scratch/$USER/dasfwi_results

# --- 3. sanity: confirm a GPU is visible (jobs auto-pick cuda->mps->cpu) -----
"$PYTHON_BIN" - <<'PY' || true
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(),
      "device_count", torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
