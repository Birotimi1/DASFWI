"""Mine the finished Phase-1 gate for the switch design's calibration -- FREE
(reads metrics.json only; run on the login node, no GPU, no SU).

Answers, per the verification review (F4.1):
  1. envelope's own SSIM/MAPE by rung -- is stateless envelope ~= weci at high
     skip (then envelope is the active ingredient and is the robust term), or is
     weci clearly better (its GC-refinement half mattered)?
  2. skip_final by misfit x rung -- do the robust runs actually END below the
     hand-over threshold (off_below)? If weci/envelope@s16 finish at skip < ~0.45
     the hand-over is REACHABLE; if they finish ~0.6 the staged switch would
     never fire and needs a fallback trigger.
  3. the converged-L2 skip floor (l2@s6 skip_final) -- off_below must sit ABOVE
     this, else the controller could never hand over even on a model L2 itself
     is happy with.

    python hpc/marmousi_full_das/mine_gate.py [--results DIR]
"""
import argparse
import json
import os
import sys
from pathlib import Path

# repo root AND the bundled ADFWI mirror (adaptive_misfit imports ADFWI.fwi.misfit).
# Mirrors common.py's path setup; ADFWI_ROOT overrides the bundled copy.
_REPO = Path(__file__).resolve().parents[2]
_ADFWI = Path(os.environ.get("ADFWI_ROOT", _REPO / "ADFWI_local"))
for _p in (str(_ADFWI), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from inversion.adaptive_misfit import SKIP_ON_ABOVE, SKIP_OFF_BELOW

RUNG_DIRS = {"s6": "", "s16": "ladder_s16", "s20": "ladder_s20"}
OPTS = ("sgd", "adagrad", "adam", "adamw", "nadam")
FOCUS = ("l2", "envelope", "weci", "gc", "traveltime")


def load(results, rung, combo):
    d = results / RUNG_DIRS[rung] / combo if RUNG_DIRS[rung] else results / combo
    f = d / "metrics.json"
    if not f.is_file():
        return None
    m = json.loads(f.read_text())
    return m if m.get("complete", True) else None      # partials excluded


def fmt(x, spec=".3f", none="   -"):
    return format(x, spec) if isinstance(x, (int, float)) else none


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    args = ap.parse_args()
    results = Path(args.results) if args.results else \
        Path(__file__).resolve().parents[2] / "results" / "marmousi_full_das"

    print("=" * 78)
    print("GATE MINING -- calibration for the L2<->envelope skip switch")
    print(f"thresholds under test: on_above={SKIP_ON_ABOVE}  off_below={SKIP_OFF_BELOW}")
    print("=" * 78)

    # ---- 1. envelope vs weci vs l2, SSIM by rung (mean + best over optimizers)
    print("\n--- 1. is ENVELOPE the active ingredient of weci? (SSIM by rung) ---")
    print(f"{'misfit':11s}" + "".join(f"  {r:>13s}" for r in RUNG_DIRS))
    for mf in FOCUS:
        row = f"{mf:11s}"
        for rung in RUNG_DIRS:
            vals = [m["ssim"] for o in OPTS
                    if (m := load(results, rung, f"{mf}_{o}")) and m.get("ssim")]
            row += (f"  {sum(vals)/len(vals):5.3f}/{max(vals):5.3f}"
                    if vals else f"  {'-':>11s}")
        print(row + ("   (mean/best)" if mf == FOCUS[0] else ""))

    # ---- 2. skip_final: is the hand-over reachable? --------------------------
    print("\n--- 2. skip_final by misfit x rung (mean over optimizers) ---")
    print("    hand-over is REACHABLE where the robust term ends below "
          f"off_below={SKIP_OFF_BELOW}")
    print(f"{'misfit':11s}" + "".join(f"  {r:>11s}" for r in RUNG_DIRS))
    verdicts = {}
    for mf in FOCUS:
        row = f"{mf:11s}"
        for rung in RUNG_DIRS:
            vals = [m["skip_final"] for o in OPTS
                    if (m := load(results, rung, f"{mf}_{o}"))
                    and isinstance(m.get("skip_final"), (int, float))]
            v = sum(vals) / len(vals) if vals else None
            if mf in ("envelope", "weci") and rung == "s16":
                verdicts[mf] = v
            row += f"  {fmt(v, '11.3f', '          -')}"
        print(row)

    # ---- 3. the converged-L2 skip floor --------------------------------------
    print("\n--- 3. converged-L2 skip floor (l2@s6 skip_final; off_below must sit ABOVE it) ---")
    floor = [m["skip_final"] for o in OPTS
             if (m := load(results, "s6", f"l2_{o}"))
             and isinstance(m.get("skip_final"), (int, float))]
    fl = min(floor) if floor else None
    print(f"    l2@s6 skip_final: min={fmt(fl)} "
          f"(per-opt: {', '.join(fmt(v) for v in floor) if floor else '-'})")

    # ---- verdict --------------------------------------------------------------
    print("\n" + "=" * 78)
    ok = True
    for mf, v in verdicts.items():
        if v is None:
            print(f"VERDICT {mf}@s16: no skip_final recorded -- run the hand-over sweep")
            ok = False
        elif v < SKIP_OFF_BELOW:
            print(f"VERDICT {mf}@s16 ends at skip {v:.3f} < {SKIP_OFF_BELOW}: "
                  "hand-over REACHABLE with the default off_below")
        else:
            print(f"VERDICT {mf}@s16 ends at skip {v:.3f} >= {SKIP_OFF_BELOW}: "
                  "hand-over would NOT fire -- raise off_below toward this value "
                  "(hand-over sweep) or add a plateau trigger")
            ok = False
    if fl is not None and fl >= SKIP_OFF_BELOW:
        print(f"WARNING: off_below={SKIP_OFF_BELOW} <= converged-L2 floor {fl:.3f} "
              "-- the controller could never hand over; raise off_below above the floor")
        ok = False
    print("mining verdict:", "CLEAN -- proceed to the hand-over sweep" if ok
          else "ADJUST THRESHOLDS -- see lines above, then the hand-over sweep")
    print("=" * 78)


if __name__ == "__main__":
    main()
