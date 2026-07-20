#!/usr/bin/env bash
# Status of the elastic A/B campaign: progress, a REAL-problem health check,
# the first completed metrics.json, and the ranking (partial is fine).
#
#   ./hpc/elastic_full_das/status.sh
#
# Run from anywhere; it resolves the repo root itself. Override the results dir
# with DASFWI_RESULTS and the condor log dir with DASFWI_OUT if non-default.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
RES="${DASFWI_RESULTS:-results/elastic_full_das}"
OUT="${DASFWI_OUT:-output}"

echo "==================== elastic A/B campaign status ===================="
n_done=$(find "$RES" -maxdepth 2 -name metrics.json 2>/dev/null | wc -l | tr -d ' ')
echo "completed (metrics.json written): ${n_done} / 90"

echo ""
echo "--- health check (real failures only; ignores library warnings) ---"
# a genuine numeric failure prints 'loss nan/inf' in the .out
bad_out=$(grep -ilE "loss (nan|-nan|inf)" "$OUT"/elastic_*.out 2>/dev/null || true)
if [ -n "$bad_out" ]; then
    echo "  NaN/Inf LOSS in:"; echo "$bad_out" | sed 's/^/    /'
else
    echo "  no NaN/Inf loss in any .out  (good)"
fi
# a genuine crash prints a Python Traceback / CUDA error / OOM in the .err
bad_err=$(grep -lE "Traceback \(most recent|CUDA error|RuntimeError|out of memory" \
          "$OUT"/elastic_*.err 2>/dev/null || true)
if [ -n "$bad_err" ]; then
    echo "  CRASH (traceback/CUDA/OOM) in:"; echo "$bad_err" | sed 's/^/    /'
else
    echo "  no tracebacks / CUDA errors in any .err  (bare 'nan'/'error' grep hits"
    echo "  are just library warnings, e.g. 'pandas' matches 'nan')"
fi

echo ""
echo "--- first completed metrics.json ---"
first=$(find "$RES" -maxdepth 2 -name metrics.json 2>/dev/null | sort | head -1)
if [ -n "$first" ]; then
    echo "$first:"; cat "$first"; echo
else
    echo "  (none finished yet -- metrics.json is written after iteration 300)"
fi

echo ""
echo "--- ranking (partial ok) ---"
if [ "$n_done" -gt 0 ]; then
    python hpc/elastic_full_das/rank_campaign.py
else
    echo "  (no completed runs yet -- re-run this script once jobs finish)"
fi
