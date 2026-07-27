#!/usr/bin/env bash
# Conda activation for PSC Bridges-2 (SLURM). The scheduler-agnostic wrappers in
# hpc/condor/ (run_standalone.sh / run_combo*.sh) source THIS file instead of
# their own activate_env.sh when DASFWI_ACTIVATE points here -- the *.sbatch
# files in this directory set that. Same contract as activate_env.sh: leave
# `python` pointing at the DASFWI conda env and export PYTHON_BIN.
#
# Bridges-2 specifics (verified 2026-07-27 on bridges2-login012):
#   * $HOME (/jet/home/$USER) is only 25 GB -> the multi-GB conda env and all
#     data/results live in Ocean: $PROJECT = /ocean/projects/ees260010p/$USER.
#   * We use our OWN Miniforge at $PROJECT/miniforge3 (set up by
#     setup_bridges2.sh), NOT the system `anaconda3` module -- this reproduces
#     the validated conda-forge stack (env.yml) exactly and avoids the
#     `defaults` channel. So NO `module load` is needed here.
#   * torch is the cu124 pip wheel, which BUNDLES its CUDA runtime; loading the
#     system `cuda/12.4.0` module would only risk LD_LIBRARY_PATH clashes, so we
#     deliberately do not. The H100 nodes' driver is what torch needs, and it is
#     always present on a GPU node.
#
# Overrides: PYTHON_BIN (skip activation entirely), DASFWI_ENV (env name,
# default "dasfwi"), DASFWI_CONDA_ROOT (Miniforge location).
set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" ]]; then
    _env="${DASFWI_ENV:-dasfwi}"

    # $PROJECT is set by PSC's login profile, but a batch job may not source it.
    # Fall back to the known Ocean path for this allocation.
    _project="${PROJECT:-/ocean/projects/ees260010p/$(whoami)}"
    _root="${DASFWI_CONDA_ROOT:-${_project}/miniforge3}"

    _conda="${_root}/bin/conda"
    if [[ ! -x "$_conda" ]]; then
        # tolerate a few alternative names/locations before giving up
        for _c in "${_project}/miniconda3/bin/conda" \
                  "${_project}/mambaforge/bin/conda" \
                  "${HOME}/miniforge3/bin/conda"; do
            [[ -x "$_c" ]] && { _conda="$_c"; break; }
        done
    fi
    if [[ ! -x "$_conda" ]] && command -v conda >/dev/null 2>&1; then
        _conda="$(command -v conda)"
    fi
    if [[ ! -x "$_conda" ]]; then
        echo "activate_bridges2.sh: no conda found (looked for Miniforge at" \
             "${_root}); run hpc/slurm/setup_bridges2.sh first, or set" \
             "DASFWI_CONDA_ROOT / PYTHON_BIN" >&2
        exit 3
    fi

    # conda's hook / activate touch unset vars -> relax `set -u` across them,
    # but still fail loudly if the env does not activate.
    set +u
    eval "$("$_conda" shell.bash hook)"
    if ! conda activate "$_env"; then
        echo "activate_bridges2.sh: 'conda activate $_env' failed -- does the" \
             "env exist? (run setup_bridges2.sh / set DASFWI_ENV)" >&2
        exit 4
    fi
    set -u
    PYTHON_BIN=python
fi
export PYTHON_BIN

# headless compute node: Agg backend, and a writable matplotlib cache. SLURM
# gives $LOCAL (node-local scratch) and $TMPDIR; fall back to /tmp.
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${LOCAL:-${TMPDIR:-/tmp}}}"

# one-line sanity to the job .out: interpreter + which GPU SLURM handed us.
"$PYTHON_BIN" - <<'PY' || true
import torch, sys
print(f"python={sys.version.split()[0]} torch={torch.__version__} "
      f"cuda={torch.cuda.is_available()} "
      f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
PY
