#!/usr/bin/env bash
# HTCondor wrapper: one misfit x optimizer cell of the full-Marmousi2 ELASTIC
# 3-parameter DAS campaign (Vp + Vs + density). elastic_full_das.sub passes the
# combos.txt line via `arguments = $(combo)`; condor's old-syntax arguments
# split on whitespace, so this normally arrives as two args:
#   run_combo_elastic.sh <misfit> <optimizer>
# Also accept the whole line as ONE arg ("l2 sgd") for safety.
set -euo pipefail

if [[ $# -eq 1 ]]; then
    read -r MISFIT OPTIMIZER <<<"$1"
else
    MISFIT="${1:?usage: run_combo_elastic.sh <misfit> <optimizer>}"
    OPTIMIZER="${2:?usage: run_combo_elastic.sh <misfit> <optimizer>}"
fi
: "${OPTIMIZER:?run_combo_elastic.sh: could not parse optimizer from: $*}"

source "$(dirname "$0")/activate_env.sh"

echo "host=$(hostname) misfit=${MISFIT} optimizer=${OPTIMIZER} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
exec "$PYTHON_BIN" hpc/elastic_full_das/run_one.py \
     --misfit "$MISFIT" --optimizer "$OPTIMIZER"
