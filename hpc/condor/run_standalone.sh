#!/usr/bin/env bash
# HTCondor wrapper for the standalone scripts + the search tools. run.sub calls:
#   run_standalone.sh <kind> <misfit> <optimizer> [extra args...]
# acoustic/elastic/field take --misfit/--optimizer; ladder/matrix take only the
# extra args (their own --misfits/--optimizers/... lists), so misfit/optimizer
# are ignored for those.
set -euo pipefail

KIND="${1:?kind required}"; MISFIT="${2:-gc}"; OPT="${3:-adam}"; shift 3 || true
EXTRA=("$@")

case "$KIND" in
    genobs)         SCRIPT=hpc/marmousi_full_das/generate_obs.py; USE_MO=0 ;;
    genobs_elastic) SCRIPT=hpc/elastic_full_das/generate_obs.py;  USE_MO=0 ;;
    acoustic) SCRIPT=hpc/standalone/run_acoustic_das.py; USE_MO=1 ;;
    elastic)  SCRIPT=hpc/standalone/run_elastic_das.py;  USE_MO=1 ;;
    field)    SCRIPT=hpc/standalone/run_field_das.py;    USE_MO=1 ;;
    ladder)   SCRIPT=inversion/run_starting_model_ladder.py; USE_MO=0 ;;
    matrix)   SCRIPT=inversion/run_technique_matrix.py;      USE_MO=0 ;;
    *) echo "kind must be genobs|genobs_elastic|acoustic|elastic|field|ladder|matrix, got: $KIND" >&2; exit 2 ;;
esac

source "$(dirname "$0")/activate_env.sh"

echo "host=$(hostname) kind=$KIND script=$SCRIPT CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
# ${EXTRA[@]+...} guards the empty-array-under-set-u case in bash 3.2
if [[ "$USE_MO" == "1" ]]; then
    exec "$PYTHON_BIN" "$SCRIPT" --misfit "$MISFIT" --optimizer "$OPT" \
         ${EXTRA[@]+"${EXTRA[@]}"}
else
    exec "$PYTHON_BIN" "$SCRIPT" ${EXTRA[@]+"${EXTRA[@]}"}
fi
