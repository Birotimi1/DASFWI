#!/bin/bash
# Slurm ARRAY job: the full-Marmousi2 DAS campaign, 40 combos (8 misfits x
# 5 optimizers), one GPU per task. Task i runs line i of combos.txt.
#
#   mkdir -p logs
#   python hpc/marmousi_full_das/generate_obs.py      # ONCE, before submitting
#   sbatch hpc/slurm/campaign.slurm
#   squeue -u $USER ; sacct -j <jobid>
#
#SBATCH --job-name=dasfwi-campaign
#SBATCH --array=1-40
#SBATCH --gres=gpu:1              # some clusters: --gpus=1 or --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
### TODO (cluster-specific): uncomment and set for your site
### #SBATCH --partition=gpu
### #SBATCH --account=YOUR_ALLOCATION
### #SBATCH --qos=YOUR_QOS

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source hpc/slurm/env.sh

read MISFIT OPT <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" hpc/marmousi_full_das/combos.txt)"
echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID} misfit=${MISFIT} optimizer=${OPT} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
exec "$PYTHON_BIN" hpc/marmousi_full_das/run_one.py \
     --misfit "$MISFIT" --optimizer "$OPT"
