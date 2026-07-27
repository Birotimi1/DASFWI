#!/usr/bin/env bash
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --job-name=dasfwi_array
#SBATCH --output=output/%x.%A_%a.out
#SBATCH --error=output/%x.%A_%a.err
# ---------------------------------------------------------------------------
# One cell of a DASFWI CAMPAIGN array on Bridges-2 GPU-shared (one H100 each).
# The SLURM analog of HTCondor's `queue combo from <file>` -- submit_array.sh
# sets --array=0-(N-1)%K from the combos file, and each array task runs the
# (task_id+1)-th line through the given wrapper. Those wrappers
# (run_combo.sh / run_combo_ladder.sh / run_combo_elastic.sh) are the SAME ones
# HTCondor uses and accept the whole combo line as a single argument.
#
#   sbatch --array=0-134%10 hpc/slurm/bridges2_array.sbatch \
#          hpc/marmousi_full_das/combos_ladder.txt hpc/condor/run_combo_ladder.sh
# ---------------------------------------------------------------------------
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export DASFWI_ACTIVATE="hpc/slurm/activate_bridges2.sh"

COMBOS="${1:?usage: bridges2_array.sbatch <combos-file> <wrapper>}"
WRAPPER="${2:?usage: bridges2_array.sbatch <combos-file> <wrapper>}"
IDX="${SLURM_ARRAY_TASK_ID:?must be run as a job array (use submit_array.sh)}"

LINE="$(sed -n "$((IDX + 1))p" "$COMBOS")"
if [[ -z "${LINE//[[:space:]]/}" ]]; then
    echo "array task $IDX: blank/missing line $((IDX + 1)) of $COMBOS -- nothing to do"
    exit 0
fi
echo "=== array ${SLURM_ARRAY_JOB_ID} task $IDX on $(hostname): '$LINE' via $(basename "$WRAPPER") ==="
exec "$WRAPPER" "$LINE"
