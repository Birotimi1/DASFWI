#!/bin/bash
# Slurm ARRAY job: full-Marmousi2 DAS campaign, 40 combos (8 misfits x 5
# optimizers), one A6000 GPU per task. Task i runs line i of combos.txt.
# Tuned for Syracuse OrangeGrid; see hpc/slurm/README.md.
#
#   mkdir -p logs
#   python hpc/marmousi_full_das/generate_obs.py      # ONCE, before submitting
#   sbatch hpc/slurm/campaign.slurm
#   squeue -u $USER
#
#SBATCH --job-name=dasfwi-campaign
#SBATCH --array=1-40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_zone2,gpu
#SBATCH --constraint=gpu_type:A6000     # verify exact string: sinfo -o "%f"
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Birotimi@syr.edu
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --requeue
### add only if your OrangeGrid allocation requires it:
### #SBATCH --account=YOUR_ALLOCATION

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source hpc/slurm/env.sh

read MISFIT OPT <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" hpc/marmousi_full_das/combos.txt)"
echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID} misfit=${MISFIT} optimizer=${OPT} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
srun "$PYTHON_BIN" hpc/marmousi_full_das/run_one.py \
     --misfit "$MISFIT" --optimizer "$OPT"
