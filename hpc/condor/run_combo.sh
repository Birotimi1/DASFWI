#!/usr/bin/env bash
# HTCondor wrapper: one misfit x optimizer cell of the full-Marmousi2 DAS
# campaign. marmousi_full_das.sub passes the combos.txt line via
# `arguments = $(combo)`; condor's old-syntax arguments split on whitespace,
# so this normally arrives as two args:
#   run_combo.sh <misfit> <optimizer>
# Also accept the whole line as ONE arg ("l2 sgd") so a quoted/new-syntax
# arguments line can never break the campaign.
set -euo pipefail

if [[ $# -eq 1 ]]; then
    read -r MISFIT OPTIMIZER <<<"$1"
else
    MISFIT="${1:?usage: run_combo.sh <misfit> <optimizer>}"
    OPTIMIZER="${2:?usage: run_combo.sh <misfit> <optimizer>}"
fi
: "${OPTIMIZER:?run_combo.sh: could not parse optimizer from: $*}"

# activate the conda env (edit hpc/condor/activate_env.sh once for your account)
source "$(dirname "$0")/activate_env.sh"

# initialdir in the .sub is the DASFWI repo root, so relative paths work.
# ADFWI_ROOT / MARMOUSI_DIR / DASFWI_RESULTS may be exported via the .sub
# `environment=` line if the side-by-side data layout is not used.
echo "host=$(hostname) misfit=${MISFIT} optimizer=${OPTIMIZER} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
exec "$PYTHON_BIN" hpc/marmousi_full_das/run_one.py \
     --misfit "$MISFIT" --optimizer "$OPTIMIZER"
