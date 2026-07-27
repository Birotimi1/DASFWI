#!/usr/bin/env bash
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=dasfwi
#SBATCH --output=output/%x.%j.out
#SBATCH --error=output/%x.%j.err
# ---------------------------------------------------------------------------
# ONE DASFWI job on Bridges-2 GPU-shared -> one H100 (2 SU / GPU-hour).
#
# ADFWI is single-GPU per process, so we take ONE of a shared node's 8 H100s;
# SLURM hands us proportional CPU/RAM (~256 GB) and sets CUDA_VISIBLE_DEVICES,
# so the runners' auto-picked "cuda" IS the assigned card.
#
# Args after the script name flow straight into the scheduler-agnostic
# dispatcher, exactly as HTCondor's run.sub does:
#     run_standalone.sh <kind> <misfit> <optimizer> [extra...]
# Passed POSITIONALLY (not --export) so comma-bearing values like
# "--bands 2.0,3.0,4.5,full" survive intact.
#
# Submit via hpc/slurm/submit.sh (sets per-kind job-name + walltime). Direct:
#   sbatch -J dasfwi_genobs hpc/slurm/bridges2.sbatch genobs
#   sbatch -J dasfwi_adaptive hpc/slurm/bridges2.sbatch adaptive gc adam \
#          --objective adaptive --start-rung s20 --flip-lo 3 --flip-hi 8
# ---------------------------------------------------------------------------
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

# make the condor wrappers activate the Bridges-2 env instead of OrangeGrid's.
export DASFWI_ACTIVATE="hpc/slurm/activate_bridges2.sh"

echo "=== SLURM ${SLURM_JOB_ID:-?} on $(hostname) | partition=${SLURM_JOB_PARTITION:-?} gpus=${SLURM_GPUS:-?} ==="
echo "    argv: $*"

exec hpc/condor/run_standalone.sh "$@"
