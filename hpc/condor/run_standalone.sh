#!/usr/bin/env bash
# HTCondor wrapper for the standalone one-file scripts:
#   run_standalone.sh <acoustic|elastic> [any run_*_das.py args...]
# e.g. run_standalone.sh acoustic --misfit gc --optimizer adam
#      run_standalone.sh acoustic --conventional          # pressure control
#      run_standalone.sh elastic  --misfit sinkhorn --optimizer sgd
set -euo pipefail

KIND="$1"; shift
case "$KIND" in
    acoustic) SCRIPT=hpc/standalone/run_acoustic_das.py ;;
    elastic)  SCRIPT=hpc/standalone/run_elastic_das.py ;;
    *) echo "first arg must be acoustic|elastic, got: $KIND" >&2; exit 2 ;;
esac

if [[ -z "${PYTHON_BIN:-}" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-dasfwi}"
    PYTHON_BIN=python
fi

echo "host=$(hostname) script=$SCRIPT args=$*"
exec "$PYTHON_BIN" "$SCRIPT" "$@"
