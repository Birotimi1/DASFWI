#!/bin/bash
# Slurm SINGLE run of a standalone script (acoustic / elastic / field), or the
# starting-model ladder. Parameters come from environment variables passed with
# --export; sensible defaults otherwise.
#
#   mkdir -p logs
#   # acoustic DAS, gc + adam:
#   sbatch --export=ALL,KIND=acoustic,MISFIT=gc,OPT=adam hpc/slurm/run.slurm
#   # the observable A/B control (pressure receivers):
#   sbatch --export=ALL,KIND=acoustic,MISFIT=gc,OPT=adam,EXTRA=--conventional hpc/slurm/run.slurm
#   # elastic Vp+Vs:
#   sbatch --export=ALL,KIND=elastic,MISFIT=gc,OPT=adam hpc/slurm/run.slurm
#   # field 78A-32 with a data-driven starting model:
#   sbatch --export=ALL,KIND=field,MISFIT=gc,EXTRA="--well 78A-32 --shots 318 --starting traveltime --dz 5 --dt 4e-4 --nt 6000" hpc/slurm/run.slurm
#   # degradation ladder:
#   sbatch --export=ALL,KIND=ladder hpc/slurm/run.slurm
#
#SBATCH --job-name=dasfwi-run
#SBATCH --gres=gpu:1              # some clusters: --gpus=1 or --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
### TODO (cluster-specific):
### #SBATCH --partition=gpu
### #SBATCH --account=YOUR_ALLOCATION

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source hpc/slurm/env.sh

KIND="${KIND:?set KIND=acoustic|elastic|field|ladder via --export}"
MISFIT="${MISFIT:-gc}"
OPT="${OPT:-adam}"
EXTRA="${EXTRA:-}"

case "$KIND" in
  acoustic) SCRIPT=hpc/standalone/run_acoustic_das.py ;;
  elastic)  SCRIPT=hpc/standalone/run_elastic_das.py ;;
  field)    SCRIPT=hpc/standalone/run_field_das.py ;;
  ladder)   SCRIPT=inversion/run_starting_model_ladder.py ;;
  *) echo "KIND must be acoustic|elastic|field|ladder, got: $KIND" >&2; exit 2 ;;
esac

echo "host=$(hostname) kind=$KIND script=$SCRIPT CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
if [[ "$KIND" == "ladder" ]]; then
    exec "$PYTHON_BIN" "$SCRIPT" $EXTRA
else
    exec "$PYTHON_BIN" "$SCRIPT" --misfit "$MISFIT" --optimizer "$OPT" $EXTRA
fi
