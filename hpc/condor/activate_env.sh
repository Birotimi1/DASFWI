#!/usr/bin/env bash
# Cluster-specific env activation for the OrangeGrid HTCondor wrappers. Sourced
# by run_combo.sh and run_standalone.sh. Edit ONCE for your account.
#
# Goal: make `python` on an execute node be the interpreter of the conda env
# that has the DASFWI stack (numpy 1.24.4 / scipy 1.10.1 / obspy / pysdtw /
# geomloss / POT + a CUDA torch build). On OrangeGrid this is Miniforge.
#
# Set DASFWI_ENV to the env name (default "dasfwi"; use "adfwi" to reuse the
# env you already built on OrangeGrid -- but see the version note in
# hpc/condor/README.md). If PYTHON_BIN is already exported (e.g. via the .sub
# `environment=` line) it is used as-is and no activation happens.
set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" ]]; then
    _env="${DASFWI_ENV:-dasfwi}"
    # find a conda/mamba base (Miniforge first on OrangeGrid), then activate
    if command -v conda >/dev/null 2>&1; then
        _base="$(conda info --base)"
    else
        for _c in "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3" \
                  "$HOME/anaconda3" /opt/miniforge3 /opt/conda; do
            [[ -f "$_c/etc/profile.d/conda.sh" ]] && { _base="$_c"; break; }
        done
    fi
    if [[ -z "${_base:-}" ]]; then
        echo "activate_env.sh: no conda/miniforge found; set PYTHON_BIN or edit this file" >&2
        exit 3
    fi
    source "$_base/etc/profile.d/conda.sh"
    conda activate "$_env"
    PYTHON_BIN=python
fi
export PYTHON_BIN

# one-line sanity to stdout (shows in the job .out): interpreter + CUDA
"$PYTHON_BIN" - <<'PY' || true
import torch, sys
print(f"python={sys.version.split()[0]} torch={torch.__version__} "
      f"cuda={torch.cuda.is_available()} "
      f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
PY
