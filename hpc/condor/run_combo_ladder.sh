#!/usr/bin/env bash
# HTCondor wrapper: one cell of the Phase-1 CYCLE-SKIP LADDER — the acoustic
# 45-combo grid repeated across starting-model rungs. skip_ladder.sub passes each
# combos_ladder.txt line via `arguments = $(combo)`; condor's old-syntax arguments
# split on whitespace, so this normally arrives as three args:
#   run_combo_ladder.sh <misfit> <optimizer> <rung>     (rung = s12|s16|s20|...)
# Also accepts the whole line as ONE arg ("l2 sgd s12") for safety.
set -euo pipefail

if [[ $# -eq 1 ]]; then
    read -r MISFIT OPTIMIZER RUNG <<<"$1"
else
    MISFIT="${1:?usage: run_combo_ladder.sh <misfit> <optimizer> <rung>}"
    OPTIMIZER="${2:?usage: run_combo_ladder.sh <misfit> <optimizer> <rung>}"
    RUNG="${3:?usage: run_combo_ladder.sh <misfit> <optimizer> <rung>}"
fi
: "${RUNG:?run_combo_ladder.sh: could not parse rung from: $*}"

source "${DASFWI_ACTIVATE:-$(dirname "$0")/activate_env.sh}"

echo "host=$(hostname) misfit=${MISFIT} optimizer=${OPTIMIZER} rung=${RUNG} iters=${ITERS:-default} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
# ITERS (env) overrides run_one.py's 300 default. run_one saves only AFTER the
# loop finishes, so on a metered cluster we set ITERS to fit the walltime with
# margin (a killed cell wastes all its SU); unset -> the 300 default.
exec "$PYTHON_BIN" hpc/marmousi_full_das/run_one.py \
     --misfit "$MISFIT" --optimizer "$OPTIMIZER" --start-rung "$RUNG" \
     ${ITERS:+--iterations "$ITERS"}
