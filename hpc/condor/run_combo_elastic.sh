#!/usr/bin/env bash
# HTCondor wrapper: one cell of the full-Marmousi2 ELASTIC 3-parameter DAS
# campaign (Vp + Vs + density), A/B on the illumination preconditioner.
# elastic_full_das.sub passes each combos.txt line via `arguments = $(combo)`;
# condor's old-syntax arguments split on whitespace, so this normally arrives as
# three args:
#   run_combo_elastic.sh <misfit> <optimizer> <precond>   (precond = illum|off)
# Also accept the whole line as ONE arg ("l2 sgd illum") for safety.
set -euo pipefail

if [[ $# -eq 1 ]]; then
    read -r MISFIT OPTIMIZER PRECOND <<<"$1"
else
    MISFIT="${1:?usage: run_combo_elastic.sh <misfit> <optimizer> <precond>}"
    OPTIMIZER="${2:?usage: run_combo_elastic.sh <misfit> <optimizer> <precond>}"
    PRECOND="${3:-illum}"
fi
: "${OPTIMIZER:?run_combo_elastic.sh: could not parse optimizer from: $*}"
PRECOND="${PRECOND:-illum}"

source "$(dirname "$0")/activate_env.sh"

echo "host=$(hostname) misfit=${MISFIT} optimizer=${OPTIMIZER} precond=${PRECOND} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
exec "$PYTHON_BIN" hpc/elastic_full_das/run_one.py \
     --misfit "$MISFIT" --optimizer "$OPTIMIZER" --precond "$PRECOND"
