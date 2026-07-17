#!/usr/bin/env bash
# Liu-style multi-GPU fan-out on ONE machine (no scheduler).
#
# Liu runs his ADFWI test matrices by pinning each experiment script to one
# GPU of a multi-GPU server (device = "cuda:0" ... "cuda:7") and launching
# them concurrently; ADFWI is single-GPU per process (gpu_num is inert,
# multi-GPU is upstream future work). This script automates exactly that
# placement for our campaign combos: combo i runs on cuda:(i mod NGPU),
# NGPU at a time, in Liu-style batches.
#
# Usage (from the DASFWI repo root, env active, obs data generated):
#   ./hpc/launch_gpus.sh <NGPU> [combos_file]
#   DRY_RUN=1 ./hpc/launch_gpus.sh 4          # print the plan only
#
# For scheduler-managed clusters use hpc/condor/ instead: request_gpus=1
# per job is the same one-process-one-GPU strategy, with condor doing the
# pinning via CUDA_VISIBLE_DEVICES.
set -euo pipefail

NGPU=${1:?usage: launch_gpus.sh <NGPU> [combos_file]}
COMBOS=${2:-hpc/marmousi_full_das/combos.txt}
PYTHON_BIN=${PYTHON_BIN:-python}
mkdir -p logs

i=0
while read -r MISFIT OPT; do
    [[ -z "$MISFIT" ]] && continue
    GPU=$((i % NGPU))
    LOG="logs/gpu${GPU}_${MISFIT}_${OPT}.out"
    echo "[gpu $GPU] $MISFIT $OPT -> $LOG"
    if [[ -z "${DRY_RUN:-}" ]]; then
        CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" \
            hpc/marmousi_full_das/run_one.py \
            --misfit "$MISFIT" --optimizer "$OPT" > "$LOG" 2>&1 &
    fi
    i=$((i + 1))
    # Liu-style batches: fill all NGPU cards, wait for the batch to finish
    if (( i % NGPU == 0 )) && [[ -z "${DRY_RUN:-}" ]]; then
        wait
    fi
done < "$COMBOS"
[[ -z "${DRY_RUN:-}" ]] && wait
echo "done: $i combos"
