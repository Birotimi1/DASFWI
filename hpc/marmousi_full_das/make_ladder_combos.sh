#!/usr/bin/env bash
# Build combos_ladder.txt for the Phase-1 cycle-skip test: the full 45-combo
# grid (from combos.txt) crossed with the starting-model rungs you pass.
#
# Run calibrate_rungs.py FIRST and use the rungs it reports as "TRANSITION" —
# rungs that are 0% or 100% skipped teach us nothing and cost 45 jobs each.
#
#   ./hpc/marmousi_full_das/make_ladder_combos.sh s12 s16 s20 s24
#   ./hpc/marmousi_full_das/make_ladder_combos.sh s16 s20 s24 const
#
# s6 is deliberately NOT a valid argument: it is the finished baseline campaign
# (rung 0) and must not be re-run.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/combos.txt"
OUT="$HERE/combos_ladder.txt"
VALID="s12 s16 s20 s24 const"

[ $# -ge 1 ] || { echo "usage: $0 <rung> [rung ...]   (valid: $VALID)" >&2; exit 2; }
for r in "$@"; do
    case " $VALID " in
        *" $r "*) ;;
        *) echo "invalid rung '$r' (valid: $VALID; s6 is the finished baseline)" >&2
           exit 2 ;;
    esac
done

: > "$OUT"
for r in "$@"; do
    while read -r m o; do
        [ -n "${m:-}" ] || continue
        printf '%s %s %s\n' "$m" "$o" "$r" >> "$OUT"
    done < "$SRC"
done

n=$(wc -l < "$OUT" | tr -d ' ')
echo "wrote $OUT: $n jobs = $(wc -l < "$SRC" | tr -d ' ') combos x $# rungs ($*)"
echo "submit with: condor_submit hpc/condor/skip_ladder.sub"
