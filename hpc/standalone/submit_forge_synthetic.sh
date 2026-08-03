#!/usr/bin/env bash
# FORGE SYNTHETIC campaign -- 11 cells on the deadline path to an acoustic result.
#
#   hpc/standalone/submit_forge_synthetic.sh --list      # print, run nothing
#   hpc/standalone/submit_forge_synthetic.sh --dry-run   # validate all 11, no GPU
#   hpc/standalone/submit_forge_synthetic.sh --smoke     # 3 GPU jobs, ~1 SU
#   hpc/standalone/submit_forge_synthetic.sh             # submit all 11
#
# RUN IN THAT ORDER. --dry-run validates configuration and exits before the
# loop; only --smoke actually executes. Assembling and running this pipeline
# found SIX bugs no unit test could catch, two of which produced plausible
# numbers rather than crashing (a silently non-exact E3 gauge, and a NaN skip
# measurement that quietly disabled the switch).
#
# WHY A SYNTHETIC AT ALL, on a deadline: the field has NO GROUND TRUTH, so a
# wrong operator, starter or misfit yields a plausible model with nothing to
# flag it. Here truth is known, so each choice is decided by measurement. And
# it is NOT an inverse crime -- data are ELASTIC, inverted ACOUSTICALLY, with
# noise and the measured 162 m topographic ramp.
#
# FOUR STAGES, each fixing its winner for the next (NOT a factorial):
#   0  sanity      does it invert at all, matched wavelet, single band      2
#   1  WAVELET     l2 / gc / convsi x {matched, mismatched}  -- decides the  6
#                  field refiner (#50). Park ASSUME a 10 Hz Ricker; l2 fits
#                  amplitude AND phase so it must absorb a wrong wavelet into
#                  the VELOCITY MODEL, convsi is source-independent and should
#                  not. Nobody has reported this for FORGE.
#   2  MULTISCALE  cascade x switch, 2x2 -- "does multiscale help" and "does   4
#                  it help GIVEN the switch" are different questions, and
#                  conflating them is how Phase B went wrong on Marmousi
#   3  PARK-MATCH  their setup: gc, 30 iterations, single band              2
#
# COST: ~8 SU/cell at 150 iterations (10 m, nt=2000 = 12.7x a Marmousi cell),
# ~0.8 SU for the 30-iteration Park cells. Total ~40 SU.
# NOTE: this is a BASH script. `$a` splits on whitespace here; the zsh-only
# `${=a}` form is a syntax error under bash and has tripped me up repeatedly.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"; cd "$REPO"
MODE="${1:-submit}"
RUN=hpc/standalone/run_forge_synthetic.py

check_fixes() {
    local bad=0
    grep -q "z_index must be scalar or one per source" forge/proxy_model.py || {
        echo "*** forge_fibers/vibroseis_line predate the topography fixes:" >&2
        echo "    sources would be buried in the AIR layer and skip reads NaN." >&2
        echo "    Run: git pull  (99f68f8)" >&2; bad=1; }
    grep -q "gauge_l = float(2.0 \* dz" forge/proxy_model.py || {
        echo "*** the DAS gauge is still hardcoded to 10 m -- E3 is NOT exact" >&2
        echo "    at dz != 5 m, and it does not crash, it just goes wrong." >&2
        echo "    Run: git pull  (2e833e7)" >&2; bad=1; }
    grep -q "assuming SKIPPED" "$RUN" || {
        echo "*** a NaN skip would silently mean 'no skip' and disable the" >&2
        echo "    switch. Run: git pull  (99f68f8)" >&2; bad=1; }
    [[ $bad -eq 0 ]] || exit 3
    echo "fixes present (per-source depths, gauge = 2*dz, NaN skip fails safe)"
}

# stage | args
cells() {
# One cell per DISTINCT configuration. Several serve more than one stage --
# `switch-gc @150` is simultaneously the sanity check, the matched-wavelet gc
# arm, and the single-scale control for multiscale -- so running it once and
# reading it three times is correct, not a gap. 11 cells, not 14.
cat <<'EOF'
0,1,2|--arm switch --refiner gc --f0-true 10 --iterations 150
0,2|--arm gc --iterations 150
1|--arm switch --refiner l2 --f0-true 10 --iterations 150
1|--arm switch --refiner l2 --f0-true 14 --f0-assumed 10 --iterations 150
1|--arm switch --refiner gc --f0-true 14 --f0-assumed 10 --iterations 150
1|--arm switch --refiner convsi --f0-true 10 --iterations 150
1|--arm switch --refiner convsi --f0-true 14 --f0-assumed 10 --iterations 150
2|--arm gc --bands 5,8,12,full --iterations 150
2|--arm switch --refiner gc --bands 5,8,12,full --iterations 150
3|--arm gc --iterations 30
3|--arm switch --refiner gc --iterations 30
EOF
}
case "$MODE" in
  --list)
    cells | while IFS='|' read -r stage a; do printf '  %-13s %s\n' "$stage" "$a"; done
    echo "--- $(cells | wc -l | tr -d ' ') cells ---"
    ;;
  --dry-run)
    check_fixes; fail=0
    while IFS='|' read -r stage a; do
      if out=$(python $RUN $a --dry-run 2>&1); then
        printf '  ok   %-13s %s\n' "$stage" \
               "$(printf '%s\n' "$out" | grep -o '=== fsyn[^ ]*' | head -1)"
      else
        printf '  FAIL %-13s %s\n' "$stage" "$a"
        printf '%s\n' "$out" | tail -4 | sed 's/^/       /'; fail=1
      fi
    done < <(cells)
    [[ $fail -eq 0 ]] || { echo "dry-run FAILED -- do not submit"; exit 4; }
    # tags must be unique or two cells share a directory (5 collisions so far)
    n=$(cells | wc -l | tr -d ' ')
    u=$(while IFS='|' read -r _ a; do
          python $RUN $a --dry-run 2>/dev/null | grep -o '=== fsyn[^ ]*'
        done < <(cells) | sort -u | wc -l | tr -d ' ')
    [[ "$n" == "$u" ]] || { echo "TAG COLLISION: $n cells -> $u tags"; exit 5; }
    echo "all $n configs valid, all $u tags distinct -- next: --smoke"
    ;;
  --smoke)
    check_fixes
    # one cell per DISTINCT code path: the plain arm, the wavelet-mismatch
    # path, and the multiscale path. Smoking only the default would leave the
    # two paths the campaign actually exists to exercise unverified.
    echo "smoke: 3 cells (~4 iterations each)"
    hpc/slurm/submit.sh field gc adam -- --arm gc --smoke --n-shots 4
    hpc/slurm/submit.sh field gc adam -- --arm switch --refiner convsi \
        --f0-true 14 --f0-assumed 10 --smoke --n-shots 4
    hpc/slurm/submit.sh field gc adam -- --arm switch --refiner gc \
        --bands 5,8,full --smoke --n-shots 4
    echo
    echo "WAIT, then confirm they RAN rather than merely exited 0:"
    echo "  ls results/forge_synthetic/*smoke*/metrics.json"
    echo "  grep -h 'skip=' output/*.out | tail -5    # must NOT be nan"
    ;;
  submit)
    check_fixes; n=0
    while IFS='|' read -r stage a; do
      hpc/slurm/submit.sh field gc adam -- $a; n=$((n+1))
    done < <(cells)
    echo; echo "submitted $n cells (~50 SU)."
    echo "read with: python hpc/standalone/rank_forge_synthetic.py"
    ;;
  *) echo "usage: $0 [--list|--dry-run|--smoke]" >&2; exit 2 ;;
esac
