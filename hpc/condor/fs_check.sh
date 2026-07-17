#!/usr/bin/env bash
# Pre-flight: report what an OrangeGrid execute node can see. Never fails hard
# (so the report always lands in the .out); read it and act on any MISSING.
set -uo pipefail

echo "=== DASFWI OrangeGrid pre-flight ==="
echo "host       = $(hostname)"
echo "pwd        = $(pwd)               (this is initialdir = the repo root)"
echo "user       = $(whoami)"
echo "date       = $(date -u)"

echo "--- shared filesystem visibility (repo + data) ---"
for p in \
    "hpc/marmousi_full_das/run_one.py:repo code" \
    "ADFWI_local/ADFWI/__init__.py:bundled ADFWI engine" \
    "../Data_downloads/marmousi2:Marmousi2 SEGY (campaign)" \
    "../DAS_VSP/78A-32:FORGE field data (field runs)"; do
    path="${p%%:*}"; label="${p##*:}"
    if [[ -e "$path" ]]; then echo "  OK      $label  ($path)"
    else echo "  MISSING $label  ($path)  <-- stage this or fix the layout"; fi
done

echo "--- conda / python env (DASFWI_ENV=${DASFWI_ENV:-dasfwi}) ---"
# Run in a subshell: activate_env.sh `exit`s on failure, and it is normally
# sourced, so isolating it here keeps a bad env from aborting this diagnostic.
if ( source "hpc/condor/activate_env.sh" ) >/dev/null 2>&1; then
    echo "  conda env activates OK"
else
    echo "  MISSING/BROKEN conda env  <-- edit hpc/condor/activate_env.sh (DASFWI_ENV)"
fi

echo "--- GPU ---"
command -v nvidia-smi >/dev/null 2>&1 && \
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader \
    || echo "  nvidia-smi not found (no GPU on this slot?)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "=== end pre-flight ==="
