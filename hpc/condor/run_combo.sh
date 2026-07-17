#!/usr/bin/env bash
# HTCondor wrapper: one misfit x optimizer cell of the full-Marmousi2 DAS
# campaign. marmousi_full_das.sub passes the combos.txt line as two args:
#   run_combo.sh <misfit> <optimizer>
set -euo pipefail

MISFIT="$1"
OPTIMIZER="$2"

# activate the conda env (edit hpc/condor/activate_env.sh once for your account)
source "$(dirname "$0")/activate_env.sh"

# initialdir in the .sub is the DASFWI repo root, so relative paths work.
# ADFWI_ROOT / MARMOUSI_DIR / DASFWI_RESULTS may be exported via the .sub
# `environment=` line if the side-by-side data layout is not used.
echo "host=$(hostname) misfit=${MISFIT} optimizer=${OPTIMIZER} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
exec "$PYTHON_BIN" hpc/marmousi_full_das/run_one.py \
     --misfit "$MISFIT" --optimizer "$OPTIMIZER"
