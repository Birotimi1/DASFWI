#!/usr/bin/env bash
# FORGE FIELD acoustic inversion -- the deliverable. 10 cells, ~80 SU.
#
#   hpc/standalone/submit_forge_field.sh --list / --dry-run / --smoke / (submit)
#
# THE RECIPE, decided by MEASUREMENT on the FORGE synthetic, not by argument:
#   convsi   source-independent. Under a wrong wavelet it fits the data 2.5x
#            better than l2 while moving the model LESS (fit-per-move 20.3 vs
#            6.7). Park's gc was WORST of the three (shallow 235-261).
#   --window Park report STRONG SURFACE WAVES in the near-offset gathers, which
#            an ACOUSTIC code cannot model and explains by inventing
#            near-surface velocity. Windowing helped EVERY refiner
#            (convsi -54, gc -26, l2 -1 m/s).
#   --topo-air  the surface is a MEASURED 162 m ramp; a flat datum fabricates a
#            free-surface ghost 215 ms late = 8.6 half-cycles at 20 Hz.
#   30 AND 150 iterations, because shallow error grew MONOTONICALLY with
#            iterations on the synthetic and the 30-iteration cells were BEST.
#            Run both: on field data there is no truth, so the stopping point
#            must be observed rather than assumed.
#
# WHAT EACH CELL ANSWERS:
#   route_b vs traveltime  -- does a starter built WITHOUT PICKING match one
#                             built from picks? Park hand-pick 100 shot gathers.
#   gc traveltime          -- the PARK-COMPARABLE arm: their misfit, their
#                             starter style. The baseline to beat.
#   78B-32                 -- TWO-WELL CROSS-VALIDATION. 78A and 78B are
#                             INDEPENDENT data over shared geology, so models
#                             must agree where they overlap. Needs no truth --
#                             and Park CANNOT do this, they invert both wells
#                             together.
#
# ACCEPTANCE is already fixed in inversion/field_acceptance.py, BEFORE any of
# these run: first-arrival mismatch reduction >= 51.7% (Park INV2), cross-well
# agreement <= 10%, and the 58-32 sonic (loaded, forge/well_logs.py).
# NOTE the sonic starts at 655 m, so it validates the GRANITOID and is SILENT
# on the near surface -- exactly where the damage concentrates.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"; cd "$REPO"
MODE="${1:-submit}"
OPT="${DASFWI_OPT:-adam}"     # set from the synthetic optimizer sweep

check_fixes() {
    local bad=0
    grep -q '"--window"' hpc/standalone/run_field_das.py || {
        echo "*** run_field_das.py has no --window: the single clearest result" >&2
        echo "    from the synthetic cannot be applied. git pull (df49936)" >&2
        bad=1; }
    grep -q "SETTLED NEGATIVE" hpc/standalone/run_field_das.py || {
        echo "*** no lbfgs/nlcg guard -- both diverge. git pull" >&2; bad=1; }
    grep -q "forge_syn) SCRIPT=" hpc/condor/run_standalone.sh || {
        echo "*** run_standalone.sh missing kinds. git pull" >&2; bad=1; }
    [[ $bad -eq 0 ]] || exit 3
    echo "fixes present (window, lbfgs guard, driver routing).  optimizer=$OPT"
}

cells() {
cat <<EOF
A|--well 78A-32 --arm convsi --window --topo-air --starting route_b --optimizer $OPT --iterations 30
A|--well 78A-32 --arm convsi --window --topo-air --starting route_b --optimizer $OPT --iterations 150
B|--well 78A-32 --arm convsi --window --topo-air --starting traveltime --optimizer $OPT --iterations 30
B|--well 78A-32 --arm convsi --window --topo-air --starting traveltime --optimizer $OPT --iterations 150
C|--well 78A-32 --arm gc --window --topo-air --starting traveltime --optimizer $OPT --iterations 30
C|--well 78A-32 --arm gc --window --topo-air --starting traveltime --optimizer $OPT --iterations 150
D|--well 78B-32 --arm convsi --window --topo-air --starting route_b --optimizer $OPT --iterations 30
D|--well 78B-32 --arm convsi --window --topo-air --starting route_b --optimizer $OPT --iterations 150
E|--well 78A-32 --arm convsi --topo-air --starting route_b --optimizer $OPT --iterations 150
E|--well 78A-32 --arm switch --refiner convsi --window --topo-air --starting route_b --optimizer $OPT --iterations 150
EOF
}

case "$MODE" in
  --list) cells | while IFS='|' read -r g a; do printf '  %-2s %s\n' "$g" "$a"; done
          echo "--- $(cells | wc -l | tr -d ' ') cells ---" ;;
  --dry-run)
    check_fixes; fail=0
    while IFS='|' read -r g a; do
      if out=$(python hpc/standalone/run_field_das.py $a --smoke --dry-run 2>&1); then
        printf '  ok   %-2s %s\n' "$g" "$(printf '%s' "$out" | grep -o 'field_[^ ]*' | head -1)"
      else
        printf '  FAIL %-2s %s\n' "$g" "$a"
        printf '%s\n' "$out" | tail -4 | sed 's/^/       /'; fail=1
      fi
    done < <(cells)
    [[ $fail -eq 0 ]] || { echo "dry-run FAILED -- do not submit"; exit 4; }
    echo "all $(cells | wc -l | tr -d ' ') configs valid -- next: --smoke" ;;
  --smoke)
    check_fixes
    echo "smoke: route_b (the expensive starter path) and the switch arm"
    hpc/slurm/submit.sh field convsi "$OPT" -- --well 78A-32 --arm convsi --window \
        --topo-air --starting route_b --optimizer "$OPT" --smoke
    hpc/slurm/submit.sh field convsi "$OPT" -- --well 78A-32 --arm switch \
        --refiner convsi --window --topo-air --starting traveltime \
        --optimizer "$OPT" --smoke
    echo; echo "then: ls results/standalone_field/*smoke*/metrics.json"
    echo "      grep -h 'conditioning\\|air layer\\|skip=' output/*.out | tail" ;;
  submit)
    check_fixes; n=0
    while IFS='|' read -r g a; do
      hpc/slurm/submit.sh field convsi "$OPT" -- $a; n=$((n+1))
    done < <(cells)
    echo; echo "submitted $n cells (~80 SU) with optimizer=$OPT" ;;
  *) echo "usage: $0 [--list|--dry-run|--smoke]" >&2; exit 2 ;;
esac
