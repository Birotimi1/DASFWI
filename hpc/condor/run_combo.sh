#!/usr/bin/env bash
# HTCondor wrapper: one misfit x optimizer cell of the full-Marmousi2 DAS
# campaign. Called by marmousi_full_das.sub with the combo line as two args:
#   run_combo.sh <misfit> <optimizer>
set -euo pipefail

MISFIT="$1"
OPTIMIZER="$2"

# --- environment ------------------------------------------------------------
# Preferred: set PYTHON_BIN in the submit file's environment line to the
# absolute python of the dasfwi env. Fallback: activate conda env "dasfwi".
if [[ -z "${PYTHON_BIN:-}" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-dasfwi}"
    PYTHON_BIN=python
fi

# --- run --------------------------------------------------------------------
# initialdir in the submit file is the DASFWI repo root, so relative paths
# work; ADFWI_ROOT / MARMOUSI_DIR / DASFWI_RESULTS may be exported via the
# submit file's environment line if the side-by-side layout is not used.
echo "host=$(hostname) misfit=${MISFIT} optimizer=${OPTIMIZER}"
exec "$PYTHON_BIN" hpc/marmousi_full_das/run_one.py \
     --misfit "$MISFIT" --optimizer "$OPTIMIZER"
