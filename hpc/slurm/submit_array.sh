#!/usr/bin/env bash
# Submit a DASFWI CAMPAIGN as a SLURM job array on Bridges-2 GPU-shared.
# The analog of `condor_submit hpc/condor/skip_ladder.sub` (or marmousi_full_das.sub).
#
#   hpc/slurm/submit_array.sh <combos-file> <wrapper> [max_concurrent] [walltime]
#
#   # Phase-1 cycle-skip ladder (3-token lines "misfit optimizer rung"):
#   hpc/slurm/submit_array.sh hpc/marmousi_full_das/combos_ladder.txt \
#                             hpc/condor/run_combo_ladder.sh 10
#   # the 45-combo acoustic base campaign (2-token lines "misfit optimizer"):
#   hpc/slurm/submit_array.sh hpc/marmousi_full_das/combos.txt \
#                             hpc/condor/run_combo.sh 10
#   # the elastic campaign (3-token lines "misfit optimizer precond"):
#   hpc/slurm/submit_array.sh hpc/elastic_full_das/combos.txt \
#                             hpc/condor/run_combo_elastic.sh 8
#
# max_concurrent (%K) caps how many array tasks run AT ONCE -- this is your
# throttle on the H100 SU BURN RATE and on the shared pool. Default 8, so burn
# is <= 8 x 2 = 16 SU per wall-hour. Each task = one H100 = 2 SU/hour.
# ALWAYS calibrate one run first (submit.sh calibrate) to know the per-job cost.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"; cd "$REPO"

COMBOS="${1:?usage: submit_array.sh <combos-file> <wrapper> [max_concurrent] [walltime]}"
WRAPPER="${2:?usage: submit_array.sh <combos-file> <wrapper> [max_concurrent] [walltime]}"
MAXC="${3:-8}"
WALLTIME="${4:-04:00:00}"

[[ -f "$COMBOS" ]]  || { echo "no combos file: $COMBOS" >&2; exit 2; }
[[ -f "$WRAPPER" ]] || { echo "no wrapper: $WRAPPER"    >&2; exit 2; }

N="$(grep -c '[^[:space:]]' "$COMBOS")"
[[ "$N" -ge 1 ]] || { echo "combos file $COMBOS has no jobs" >&2; exit 2; }
LAST=$((N - 1))

mkdir -p output logs
echo "campaign: $N jobs from $COMBOS via $(basename "$WRAPPER")"
echo "  throttle: <=$MAXC concurrent -> burn rate <= $((MAXC * 2)) SU/wall-hour; ${WALLTIME}/job cap"
exec sbatch --array="0-${LAST}%${MAXC}" --time="$WALLTIME" \
     --job-name="dasfwi_$(basename "$COMBOS" .txt)" \
     hpc/slurm/bridges2_array.sbatch "$COMBOS" "$WRAPPER"
