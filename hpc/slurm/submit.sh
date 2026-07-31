#!/usr/bin/env bash
# Submit ONE DASFWI job to Bridges-2 GPU-shared (one H100). The SLURM analog of
#   condor_submit hpc/condor/run.sub -a 'kind=<k>' -a 'extra=<...>'
#
#   hpc/slurm/submit.sh <kind> [misfit] [optimizer] [-- extra args...]
#
#   hpc/slurm/submit.sh genobs
#   hpc/slurm/submit.sh acoustic gc adam
#   hpc/slurm/submit.sh calibrate
#   hpc/slurm/submit.sh adaptive -- --objective adaptive --start-rung s20 --flip-lo 3 --flip-hi 8
#   hpc/slurm/submit.sh starter  -- --iters 80 --band 3.0
#   hpc/slurm/submit.sh pipeline -- --start route_b --bands 2.0,3.0,4.5,full --iters 50
#
# kinds: genobs genobs_elastic calibrate adaptive starter pipeline
#        acoustic elastic field ladder matrix   (see hpc/condor/run_standalone.sh)
#
# Every job is ONE H100 = 2 SU/hour. Per-kind default walltime below; override
# with WALLTIME=hh:mm:ss. Add --smoke inside the extra args for a 2-iter check.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

KIND="${1:?usage: submit.sh <kind> [misfit] [optimizer] [-- extra args...]}"; shift
MISFIT="gc"; OPT="adam"
# optional positional misfit/optimizer, before an optional "--" separator
if [[ $# -gt 0 && "$1" != "--" ]]; then MISFIT="$1"; shift; fi
if [[ $# -gt 0 && "$1" != "--" ]]; then OPT="$1";    shift; fi
[[ "${1:-}" == "--" ]] && shift
# whatever remains in "$@" is the extra-args passthrough (may contain commas)

# per-kind default walltime (H100 SU = 2/hour, so keep these honest)
case "$KIND" in
    smoke)                 DEF_T="00:20:00" ;;
    genobs|genobs_elastic) DEF_T="01:00:00" ;;
    calibrate)             DEF_T="02:00:00" ;;
    # starter: the traveltime misfit runs ~30-60 s/iter on acoustic (batch_size=5
    # + checkpoint recomputation + per-trace FFT), so 300 iterations needs 2.5-5 h.
    # The old 2 h default killed cells before they wrote their acceptance numbers,
    # which are only computed at the END. SLURM charges ACTUAL runtime, not the
    # reservation, so a generous cap is free insurance.
    starter|starter_elastic) DEF_T="08:00:00" ;;
    *)                     DEF_T="08:00:00" ;;
esac
WALLTIME="${WALLTIME:-$DEF_T}"

cd "$REPO"; mkdir -p output logs
# forward ITERS explicitly (don't trust the site's --export default)
SB_EXPORT="ALL"; [[ -n "${ITERS:-}" ]] && SB_EXPORT="ALL,ITERS=$ITERS"
echo "submitting: kind=$KIND misfit=$MISFIT opt=$OPT walltime=$WALLTIME iters=${ITERS:-default} extra=[$*]"
exec sbatch --export="$SB_EXPORT" --job-name="dasfwi_${KIND}" --time="$WALLTIME" \
     hpc/slurm/bridges2.sbatch "$KIND" "$MISFIT" "$OPT" "$@"
