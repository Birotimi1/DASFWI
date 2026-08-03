#!/usr/bin/env bash
# TF-phase (Fichtner 2008) comparison campaign on the Route B starters -- 12 cells.
#
#   hpc/marmousi_full_das/submit_tfphase.sh --list      # print, run nothing
#   hpc/marmousi_full_das/submit_tfphase.sh --dry-run   # validate all 12, no GPU
#   hpc/marmousi_full_das/submit_tfphase.sh --smoke     # 2 GPU jobs, both new paths
#   hpc/marmousi_full_das/submit_tfphase.sh             # submit all 12
#
# RUN IN THAT ORDER. --dry-run validates configuration and exits before the
# loop, so it cannot catch runtime errors; --smoke is what executes TF-phase on
# a GPU. The conditioning campaign lost all 16 cells to two runtime bugs that no
# dry-run could have seen.
#
# WHAT IT ANSWERS. TF-phase is the literature's standard single-misfit answer to
# cycle skipping and what both published DAS FWI studies use, so it is the
# direct COMPETITOR to our adaptive switch. Two questions, two blocks:
#
#   A. tfphase as a STANDALONE misfit, 4 optimizers x 2 starters = 8 cells
#      -> how does it place against the bars l2 0.846 (non-skip) and
#         switch 0.742 (skip)?
#   B. tfphase as the ROBUST STAGE INSIDE the switch, 2 optimizers x 2 = 4 cells
#      -> is a phase misfit a better skip-tolerant term than envelope? This is
#         the one that could IMPROVE our method rather than merely benchmark it.
#
# The unconditioned l2 / switch / envelope baselines are ALREADY on the board
# and are deliberately not resubmitted.
#
# COST: ~20 min/cell on an H100 = 2 SU/hour, so 12 x 0.33 x 2 ~= 8 SU.
#
# EXPECT A WEAK RESULT HERE, and read it carefully rather than as a verdict on
# the method: our source spans 3.0-6.25 Hz = 1.06 octaves, so the banded Gabor
# plane is only ~6 rows wide (the dry-run prints the number). TF-phase leans on
# UNWRAPPED LOW-frequency rows and there are barely any -- the same deficiency
# that made a multiscale cascade useless on this dataset. FORGE has 4.4 octaves.
# If tfphase underperforms, check the printed row count FIRST.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"; cd "$REPO"

MODE="${1:-submit}"
OPTS=(adam adamw sgd nadam)
SWITCH_OPTS=(adam adamw)
STARTERS=(i300_gc_adam_b3 i50_gc_adam_b3)

check_fixes() {
    local bad=0
    grep -q '"tfphase"' inversion/config.py || {
        echo "*** inversion/config.py: tfphase is not registered." >&2; bad=1; }
    grep -q 'SOLO_REFINER_ARMS' hpc/marmousi_full_das/run_switch.py || {
        echo "*** run_switch.py: tfphase is not wired as an arm -- a submit" >&2
        echo "    would die at argparse. Run: git pull  (fix is commit c4e4469)" >&2
        bad=1; }
    grep -q 'rb = ("" if args.robust == "envelope"' hpc/marmousi_full_das/run_switch.py || {
        echo "*** run_switch.py: --robust is missing from the output tag, so" >&2
        echo "    'switch --robust tfphase' would OVERWRITE plain 'switch'." >&2
        echo "    Run: git pull  (fix is commit c4e4469)" >&2; bad=1; }
    [[ $bad -eq 0 ]] || exit 3
    echo "fixes present (tfphase registered, wired as an arm, robust in the tag)"
}

cells() {                       # "arm optimizer starter robust"
    for st in "${STARTERS[@]}"; do
        for o in "${OPTS[@]}";        do echo "tfphase $o $st envelope"; done
        for o in "${SWITCH_OPTS[@]}"; do echo "switch  $o $st tfphase";  done
    done
}

run_cell() {                    # arm opt starter robust [extra...]
    local arm="$1" opt="$2" st="$3" rb="$4"; shift 4
    hpc/slurm/submit.sh switch gc "$opt" \
        --arm "$arm" --robust "$rb" --start route_b --starter "$st" \
        --optimizer "$opt" "$@"
}

case "$MODE" in
  --list)
    cells | while read -r arm opt st rb; do
        echo "hpc/slurm/submit.sh switch gc $opt --arm $arm --robust $rb" \
             "--start route_b --starter $st --optimizer $opt"
    done
    echo "--- $(cells | wc -l | tr -d ' ') cells ---"
    ;;

  --dry-run)
    check_fixes
    fail=0
    while read -r arm opt st rb; do
        # keep the Gabor-plane line: "6 rows" vs "3 rows *** THIN" is the single
        # most useful number for interpreting a weak result later
        out=$(python hpc/marmousi_full_das/run_switch.py --arm "$arm" --robust "$rb" \
                --start route_b --starter "$st" --optimizer "$opt" --dry-run 2>&1) || fail=1
        plane=$(printf '%s\n' "$out" | grep -o 'Gabor plane:.*' || true)
        if printf '%s\n' "$out" | grep -q "dry-run OK"; then
            printf '  ok   %-8s %-6s %-18s robust=%-8s %s\n' \
                   "$arm" "$opt" "$st" "$rb" "$plane"
        else
            printf '  FAIL %-8s %-6s %-18s robust=%s\n' "$arm" "$opt" "$st" "$rb"
            printf '%s\n' "$out" | tail -4 | sed 's/^/       /'
            fail=1
        fi
    done < <(cells)
    [[ $fail -eq 0 ]] || { echo "dry-run FAILED -- do not submit"; exit 4; }
    echo "all $(cells | wc -l | tr -d ' ') configs valid -- next: --smoke"
    ;;

  --smoke)
    check_fixes
    # the TWO genuinely new code paths, and they are not interchangeable:
    #   tfphase SOLO   -> lambda pinned to 0, tfphase sits in the REFINER slot
    #   switch+tfphase -> lambda TOGGLES, so tfphase is entered and left through
    #                     set_lambda while the controller runs
    # Smoking only the solo arm would leave the controller path unverified,
    # which is exactly the half-covered mistake that hid the l2 arm's
    # AttributeError in the conditioning campaign.
    echo "smoke: tfphase solo, and switch with tfphase as its robust stage"
    run_cell tfphase adam i50_gc_adam_b3 envelope --smoke
    run_cell switch  adam i50_gc_adam_b3 tfphase  --smoke
    echo
    echo "WAIT for both, then confirm they RAN rather than merely exited 0:"
    echo "  sacct -u \$USER -S today --format=JobID%14,JobName%20,State,Elapsed,ExitCode | tail -6"
    echo "  ls results/marmousi_full_das/switch_routeb/*smoke*tfphase*/metrics.json"
    echo "  grep -h 'Gabor plane' output/*.out | tail -2     # row count actually used"
    echo "Then rerun with no arguments to submit all 12."
    ;;

  submit)
    check_fixes
    n=0
    while read -r arm opt st rb; do
        run_cell "$arm" "$opt" "$st" "$rb"; n=$((n+1))
    done < <(cells)
    echo
    echo "submitted $n cells (~8 SU)."
    echo "rank with:  python hpc/marmousi_full_das/rank_switch.py --rung routeb"
    echo "bars to beat: l2 0.846 (i300 non-skip), switch 0.742 (i50 skip)."
    ;;

  *) echo "usage: $0 [--list|--dry-run|--smoke]" >&2; exit 2 ;;
esac
