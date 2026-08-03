#!/usr/bin/env bash
# Noe-style DAS conditioning A/B on the Route B starters -- 32 cells.
#
#   hpc/marmousi_full_das/submit_conditioning_ab.sh --list      # print, run nothing
#   hpc/marmousi_full_das/submit_conditioning_ab.sh --dry-run   # validate all 32 configs, no GPU
#   hpc/marmousi_full_das/submit_conditioning_ab.sh --smoke     # 2 GPU jobs, both bug paths
#   hpc/marmousi_full_das/submit_conditioning_ab.sh             # submit all 32
#
# RUN THEM IN THAT ORDER. --dry-run validates configuration and exits before the
# loop, so it CANNOT catch runtime errors; --smoke is what actually executes the
# conditioning code on a GPU. The first attempt at this campaign lost all 16
# cells to two runtime bugs that a dry-run could never have seen.
#
# THE DESIGN. Conditioning (Noe et al. 2025) is meant to compose with ANY misfit,
# so the A/B needs a conditioned control that is NOT the switch -- otherwise a
# gain cannot be attributed to the conditioning rather than to its interaction
# with the switch. Hence both arms:
#     arm       switch, l2                (l2 = the control)
#     optimizer adam, adamw               (the two that lead the board)
#     starter   i300_gc_adam_b3  (skip 0.521, NON-skip regime)
#               i50_gc_adam_b3   (skip 0.668, SKIP regime)
#     cond      w    --window               (arrival window, 2 s / 4 s)
#               c    --channel-weight       (amplitude-weighted channels)
#               g    --grad-smooth wavelength  (lambda/4 gradient smoothing)
#               wcg  all three
# = 2 x 2 x 2 x 4 = 32 cells. The UNCONDITIONED baseline is the "none" level and
# is ALREADY on the board (l2/switch x adam/adamw x both starters, done), so it
# is deliberately not resubmitted -- rank_switch.py --rung routeb groups on
# (arm, optimizer, conditioning) and will place these beside it.
#
# COST: ~20 min/cell on an H100 = 2 SU/hour, so 32 x 0.33 x 2 ~= 21 SU.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"; cd "$REPO"

MODE="${1:-submit}"
ARMS=(switch l2)
OPTS=(adam adamw)
STARTERS=(i300_gc_adam_b3 i50_gc_adam_b3)
CONDS=(w c g wcg)

cond_flags() {                       # cond key -> run_switch.py flags
    case "$1" in
        w)   echo "--window" ;;
        c)   echo "--channel-weight" ;;
        g)   echo "--grad-smooth wavelength" ;;
        wcg) echo "--window --channel-weight --grad-smooth wavelength" ;;
        *)   echo "unknown conditioning key: $1" >&2; exit 2 ;;
    esac
}

# --------------------------------------------------------------------------- #
# guard: refuse to spend SUs unless BOTH fixes are actually present.
# The previous attempt died on these; a stale checkout would repeat it exactly.
# --------------------------------------------------------------------------- #
check_fixes() {
    local bad=0
    grep -q "int(round(float(fraction)" inversion/das_conditioning.py || {
        echo "*** inversion/das_conditioning.py: wavelength_span still returns a FLOAT." >&2
        echo "    smooth2d needs an int span -> every '_g' cell dies at the first" >&2
        echo "    gradient. Run: git pull   (fix is commit b51d3ef)" >&2; bad=1; }
    grep -q "def __getattr__" inversion/das_conditioning.py || {
        echo "*** inversion/das_conditioning.py: ConditionedMisfit has no __getattr__." >&2
        echo "    It would hide set_lambda/set_stage from the controller -> every" >&2
        echo "    '_w'/'_c' cell dies at the first controller update." >&2
        echo "    Run: git pull   (fix is commit b51d3ef)" >&2; bad=1; }
    [[ $bad -eq 0 ]] || exit 3
    echo "fixes present (wavelength_span->int, ConditionedMisfit.__getattr__)"
}

# --------------------------------------------------------------------------- #
# clear the empty dirs left by the failed attempt. run_switch.py skips a cell
# only when metrics.json EXISTS, so an empty dir is not itself a blocker -- but
# leaving half-written iter_*.npz behind makes a rerun ambiguous to read later.
# Only ever touches conditioned dirs with NO metrics.json.
# --------------------------------------------------------------------------- #
clear_failed() {
    local root="results/marmousi_full_das/switch_routeb" n=0
    [[ -d "$root" ]] || return 0
    for d in "$root"/*_full_w "$root"/*_full_c "$root"/*_full_g "$root"/*_full_wcg; do
        [[ -d "$d" ]] || continue
        if [[ -f "$d/metrics.json" ]]; then
            echo "  KEEPING $d (has results)"
        else
            rm -rf "$d"; n=$((n+1))
        fi
    done
    echo "cleared $n empty conditioned dir(s)"
}

cells() {                            # emit one "arm opt starter cond" per line
    for arm in "${ARMS[@]}"; do
      for opt in "${OPTS[@]}"; do
        for st in "${STARTERS[@]}"; do
          for c in "${CONDS[@]}"; do echo "$arm $opt $st $c"; done
        done
      done
    done
}

run_cell() {                         # arm opt starter cond [extra...]
    local arm="$1" opt="$2" st="$3" c="$4"; shift 4
    hpc/slurm/submit.sh switch gc "$opt" \
        --arm "$arm" --start route_b --starter "$st" --optimizer "$opt" \
        $(cond_flags "$c") "$@"
}

case "$MODE" in
  --list)
    cells | while read -r arm opt st c; do
        echo "hpc/slurm/submit.sh switch gc $opt --arm $arm --start route_b" \
             "--starter $st --optimizer $opt $(cond_flags "$c")"
    done
    echo "--- $(cells | wc -l | tr -d ' ') cells ---"
    ;;

  --dry-run)
    check_fixes
    # validate every config LOCALLY (no GPU, no scheduler). Any failure here is
    # a config error and must be fixed before a single SU is spent.
    fail=0
    while read -r arm opt st c; do
        if python hpc/marmousi_full_das/run_switch.py \
             --arm "$arm" --start route_b --starter "$st" --optimizer "$opt" \
             $(cond_flags "$c") --dry-run >/dev/null 2>&1; then
            printf '  ok   %-7s %-6s %-18s %s\n' "$arm" "$opt" "$st" "$c"
        else
            printf '  FAIL %-7s %-6s %-18s %s\n' "$arm" "$opt" "$st" "$c"
            python hpc/marmousi_full_das/run_switch.py \
              --arm "$arm" --start route_b --starter "$st" --optimizer "$opt" \
              $(cond_flags "$c") --dry-run 2>&1 | sed 's/^/       /' | tail -4
            fail=1
        fi
    done < <(cells)
    [[ $fail -eq 0 ]] || { echo "dry-run FAILED -- do not submit"; exit 4; }
    echo "all $(cells | wc -l | tr -d ' ') configs valid -- next: --smoke"
    ;;

  --smoke)
    check_fixes
    # TWO cells, chosen to cover every distinct code path the 16 failures hit:
    #   wcg exercises ConditionedMisfit (the set_lambda/__getattr__ path) AND
    #   grad_smooth (the smooth2d int-span path) in one run;
    #   and BOTH arms, because every non-ladder arm -- l2 included -- drives the
    #   misfit through BlendedMisfit.set_lambda, so the l2 control cells would
    #   have died on exactly the same bug. Smoking only `switch` would leave
    #   half the campaign unverified.
    echo "smoke: switch+wcg and l2+wcg (2 iters each, ~2 min, <1 SU)"
    run_cell switch adam i50_gc_adam_b3 wcg --smoke
    run_cell l2     adam i50_gc_adam_b3 wcg --smoke
    echo
    echo "WAIT for both to finish, then CHECK THEY REALLY RAN -- a smoke job that"
    echo "exits 0 without writing metrics.json is a failure that looks like success:"
    echo "  sacct -u \$USER -S today --format=JobID%14,JobName%20,State,Elapsed,ExitCode | tail -6"
    echo "  ls results/marmousi_full_das/switch_routeb/*smoke*/metrics.json"
    echo "Then rerun this script with no arguments to submit all 32."
    ;;

  submit)
    check_fixes
    clear_failed
    n=0
    while read -r arm opt st c; do
        run_cell "$arm" "$opt" "$st" "$c"; n=$((n+1))
    done < <(cells)
    echo
    echo "submitted $n cells (~21 SU at ~20 min each)."
    echo "rank with:  python hpc/marmousi_full_das/rank_switch.py --rung routeb"
    echo "the 'cond' column should now show w/c/g/wcg instead of '-'."
    ;;

  *) echo "usage: $0 [--list|--dry-run|--smoke]" >&2; exit 2 ;;
esac
