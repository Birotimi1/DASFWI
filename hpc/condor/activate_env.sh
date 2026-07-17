#!/usr/bin/env bash
# Conda activation for the OrangeGrid HTCondor wrappers. Sourced by
# run_combo.sh / run_standalone.sh / fs_check.sh. Edit ONCE for your account.
#
# Goal: make `python` on an execute node be the interpreter of the conda env
# that has the DASFWI stack (numpy 1.24.4 / scipy 1.10.1 / obspy / pysdtw /
# geomloss / POT + a CUDA torch build).
#
# This follows the OrangeGrid canonical pattern (Research Computing's PyTorch
# example): Miniforge installed at $HOME/miniconda3, activated with
#     eval "$(/home/$(whoami)/miniconda3/bin/conda shell.bash hook)"
#     conda activate <env>
#
# Set DASFWI_ENV to your env name (default "dasfwi"; use "adfwi" to reuse the
# torch 2.6+cu124 env you already built on OrangeGrid). If PYTHON_BIN is already
# exported (e.g. via a .sub `environment=` line) it is used as-is, no activation.
set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" ]]; then
    _env="${DASFWI_ENV:-dasfwi}"

    # A condor job runs with a scrubbed environment (no getenv), so $HOME may
    # be UNSET -- OrangeGrid's own wrappers therefore use /home/$(whoami).
    # Guarantee HOME (python libs -- matplotlib/obspy caches -- want it too),
    # then look for Miniforge there.
    export HOME="${HOME:-/home/$(whoami)}"
    _home="$HOME"

    # OrangeGrid: Miniforge at $HOME/miniconda3 (the documented location).
    _conda="${_home}/miniconda3/bin/conda"
    if [[ ! -x "$_conda" ]]; then
        # fall back to other common install roots / an already-on-PATH conda
        for _c in "$_home/miniforge3/bin/conda" "$_home/mambaforge/bin/conda" \
                  "$_home/anaconda3/bin/conda" /opt/miniforge3/bin/conda \
                  /opt/conda/bin/conda; do
            [[ -x "$_c" ]] && { _conda="$_c"; break; }
        done
    fi
    if [[ ! -x "$_conda" ]] && command -v conda >/dev/null 2>&1; then
        _conda="$(command -v conda)"
    fi
    if [[ ! -x "$_conda" ]]; then
        echo "activate_env.sh: no conda found (looked for Miniforge at" \
             "$_home/miniconda3); set PYTHON_BIN or edit this file" >&2
        exit 3
    fi

    # canonical OrangeGrid activation. conda's shell hook / `activate` reference
    # unset vars, so relax `set -u` across them -- but still fail loudly if the
    # env does not activate (otherwise we'd silently run the wrong python).
    set +u
    eval "$("$_conda" shell.bash hook)"
    if ! conda activate "$_env"; then
        echo "activate_env.sh: 'conda activate $_env' failed --" \
             "does the env exist? (set DASFWI_ENV / see hpc/condor/README.md)" >&2
        exit 4
    fi
    set -u
    PYTHON_BIN=python
fi
export PYTHON_BIN

# headless execute nodes: never try an X backend for the final.png plots, and
# keep matplotlib's font cache somewhere writable (condor scratch, else /tmp).
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}}}"

# one-line sanity to stdout (shows in the job .out): interpreter + CUDA
"$PYTHON_BIN" - <<'PY' || true
import torch, sys
print(f"python={sys.version.split()[0]} torch={torch.__version__} "
      f"cuda={torch.cuda.is_available()} "
      f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
PY
