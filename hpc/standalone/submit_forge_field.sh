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
# LOGIN-NODE CPU BUDGET. The preflight was KILLED here: "cpu time 2105.1
# seconds exceeded limit 1800". torch/BLAS start one thread per core and the
# limit counts CPU-seconds = wall x threads, so a 60 s check on a 40-core login
# node bills ~2400 s. Capped to one thread it is 14x cheaper AND finishes
# sooner in wall time (measured: 222 -> 16 CPU-s, 58 -> 21 s wall). Applies to
# the preflight and to all 13 dry-runs; the GPU jobs are unaffected, they run
# on compute nodes through their own job script.
# NOT `export`: hpc/slurm/submit.sh passes --export=ALL, so an exported cap
# would RIDE INTO THE H100 JOBS and single-thread their CPU-side work (SEG-Y
# loading, preprocessing). Scope it to the login-node commands with a prefix.
LOGIN_CAP="env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1"
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
    # >>> THE INPUT DATA. Checked FIRST, on the login node, in a second. <<<
    # The first field smoke died with "no .sgy files under
    # /ocean/projects/ees260010p/DAS_VSP/78A-32" -- the SEG-Y had never been
    # transferred to the cluster, because everything before this ran on
    # Marmousi or on a synthetic that generates its own data. Every driver
    # check passed and the run still could not start. A missing INPUT is the
    # cheapest possible failure to detect and the most expensive to discover
    # after a queue wait.
    local dd="${FORGE_DAS_DIR:-}"
    if [[ -z "$dd" ]]; then
        echo "*** FORGE_DAS_DIR is not set. The loader would fall back to a" >&2
        echo "    path beside the repo, which is where the first smoke died." >&2
        echo "    export FORGE_DAS_DIR=/ocean/projects/ees260010p/\$USER/DAS_VSP" >&2
        bad=1
    else
        # >>> PREFLIGHT ON THE H100, not the login node. <<<
    # The jobs run on a GPU; validating on a login-node CPU tests a
    # configuration nothing ever runs. On the card it also checks what only the
    # card can show: that CUDA is really there, which device SLURM handed over,
    # and whether the wavefield fits in memory. It goes through
    # run_standalone.sh -- the SAME dispatch and env activation as the real
    # jobs -- because validating through a different path is exactly how the
    # PYTHONPATH mismatch survived to the cluster.
    # Cost is a few minutes of one H100, well under 1 SU. PF_LOCAL=1 falls back
    # to the login node (no queue wait, but it does NOT prove the GPU path).
    export DASFWI_ACTIVATE="${DASFWI_ACTIVATE:-hpc/slurm/activate_bridges2.sh}"
    if [[ -n "${PF_LOCAL:-}" ]]; then
        PF_RUN() { $LOGIN_CAP python forge/preflight.py "$@"; }
        echo "    preflight: LOGIN NODE (PF_LOCAL=1) -- GPU path NOT validated"
    elif command -v srun >/dev/null 2>&1; then
        PF_RUN() {
            srun --partition="${PF_PART:-GPU-shared}" \
                 --gpus="${PF_GPU:-h100-80:1}" --time="${PF_TIME:-00:20:00}" \
                 --job-name=dasfwi_preflight \
                 hpc/condor/run_standalone.sh preflight gc adam "$@" --device cuda
        }
        echo "    preflight: H100 via srun (waits for a slot; PF_LOCAL=1 to skip)"
    else
        PF_RUN() { $LOGIN_CAP python forge/preflight.py "$@"; }
        echo "    preflight: no srun found -> login node"
    fi
    for w in 78A-32 78B-32; do
        # per-user path: /tmp is SHARED on a login node
        PFLOG="${TMPDIR:-/tmp}/pf_${USER:-x}_$w.log"
            local n
            n=$(ls "$dd/$w"/*.sgy 2>/dev/null | wc -l | tr -d ' ')
            if [[ "$n" -lt 1 ]]; then
                echo "*** no .sgy under $dd/$w -- transfer the DAS_VSP data" >&2
                bad=1
            else
                echo "    $w: $n .sgy files"
            fi
        done
        ls "$dd"/*.las >/dev/null 2>&1 \
            || echo "    NOTE: no .las in $dd -- the 58-32 sonic is missing, so" \
                    "acceptance criterion 3 (well-log comparison) cannot run." >&2
    fi
    [[ $bad -eq 0 ]] || exit 3
    # >>> THE FULL-CHAIN PREFLIGHT, on the REAL data, before any SU is spent.
    # Thirteen invariants, every one of which has already failed silently on
    # this project. Cheap, and it is the check that replaces discovering these
    # one job at a time. <<<
    for w in 78A-32 78B-32; do
        # per-user path: /tmp is SHARED on a login node
        PFLOG="${TMPDIR:-/tmp}/pf_${USER:-x}_$w.log"
        PF_RUN --well "$w" --shots "${PF_SHOTS:-20}" \
            --dz "${PF_DZ:-10}" --f0 "${PF_F0:-10}" >"$PFLOG" 2>&1 || {
            echo "*** PREFLIGHT FAILED for $w -- do NOT submit:" >&2
            # ALWAYS tail the log. Grepping only for "FAIL" printed NOTHING
            # when preflight CRASHED before reaching any check (a bare
            # ModuleNotFoundError), leaving a blank error on the cluster.
            grep -E "FAIL|Error|error:" "$PFLOG" >&2 || true
            echo "    --- last 12 lines of $PFLOG ---" >&2
            tail -12 "$PFLOG" >&2
            exit 6; }
        echo "    preflight $w: $(grep -o '[0-9]*/[0-9]* PASSED' "$PFLOG") " \
             "$(grep -o 'on cuda\|on the login node\|on mps' "$PFLOG" | head -1)" \
             "$(grep -o 'NVIDIA [A-Z0-9 ]*' "$PFLOG" | head -1)"
    done
    echo "fixes present (window, lbfgs guard, driver routing).  optimizer=$OPT"
}

#: The other three optimizers on the ONE cell that matters most (route_b, 150).
#: Insurance against the synthetic->field transfer being imperfect: the
#: synthetic has TRUTH and is the right place to CHOOSE, but the field is the
#: deliverable and its acceptance criteria (first-arrival mismatch reduction,
#: cross-well agreement) work WITHOUT truth -- so the choice is checkable here
#: too, for 3 cells instead of 30. Enable with SWEEP=1.
SWEEP="${SWEEP:-0}"

cells() {
if [[ "$SWEEP" == "1" ]]; then
  for o in adam adamw nadam sgd; do
    [[ "$o" == "$OPT" ]] && continue
    echo "S|--well 78A-32 --arm convsi --window --topo-air --starting route_b --optimizer $o --iterations 150"
  done
fi
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
    check_fixes; fail=0; TAGS=$(mktemp)
    while IFS='|' read -r g a; do
      if out=$($LOGIN_CAP python hpc/standalone/run_field_das.py $a --dry-run 2>&1); then
        tag=$(printf '%s' "$out" | grep -o 'field_[^ ]*' | head -1)
        echo "$tag" >> "$TAGS"
        printf '  ok   %-2s %s\n' "$g" "$tag"
      else
        printf '  FAIL %-2s %s\n' "$g" "$a"
        printf '%s\n' "$out" | tail -4 | sed 's/^/       /'; fail=1
      fi
    done < <(cells)
    [[ $fail -eq 0 ]] || { echo "dry-run FAILED -- do not submit"; exit 4; }
    # TAGS MUST BE DISTINCT or two cells share a directory. The synthetic
    # campaign had this check and the field one did not -- which is how the
    # 30/150 pairs came to collide.
    # tags were captured in the loop ABOVE. Re-running all 13 dry-runs purely
    # to collect them doubled the login-node CPU bill for no information.
    n=$(cells | wc -l | tr -d ' ')
    u=$(sort -u "$TAGS" | wc -l | tr -d ' '); rm -f "$TAGS"
    [[ "$n" == "$u" ]] || { echo "TAG COLLISION: $n cells -> $u tags"; exit 5; }
    echo "all $n configs valid, all $u tags distinct -- next: --smoke" ;;
  --smoke)
    check_fixes
    echo "smoke: route_b (the expensive starter path) and the switch arm"
    hpc/slurm/submit.sh field convsi "$OPT" -- --well 78A-32 --arm convsi --window \
        --topo-air --starting route_b --optimizer "$OPT" --smoke
    hpc/slurm/submit.sh field convsi "$OPT" -- --well 78A-32 --arm switch \
        --refiner convsi --window --topo-air --starting traveltime \
        --optimizer "$OPT" --smoke
    echo
    echo "WAIT for both, then run this -- it FAILS LOUDLY if they produced"
    echo "nothing, instead of leaving an empty ls to interpret:"
    cat <<'CHK'
  n=$(ls results/standalone_field/*smoke*/metrics.json 2>/dev/null | wc -l)
  if [ "$n" -lt 2 ]; then
    echo "*** SMOKE FAILED: $n/2 metrics.json. The jobs did NOT run. Look at:"
    ls -t output/dasfwi_field.*.err | head -2 | xargs -I{} sh -c 'echo "--- {}"; tail -15 {}'
  else
    echo "smoke OK ($n/2). Field data + driver verified."
    grep -h "conditioning\|air layer\|skip=" output/dasfwi_field.*.out | tail -6
  fi
CHK
    ;;
  submit)
    check_fixes; n=0
    while IFS='|' read -r g a; do
      hpc/slurm/submit.sh field convsi "$OPT" -- $a; n=$((n+1))
    done < <(cells)
    echo; echo "submitted $n cells (~80 SU) with optimizer=$OPT" ;;
  *) echo "usage: $0 [--list|--dry-run|--smoke]" >&2; exit 2 ;;
esac
