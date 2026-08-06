"""Read the FORGE ACOUSTIC FIELD campaign -- where there is NO TRUE MODEL.

    python hpc/standalone/rank_forge_field.py
    python hpc/standalone/rank_forge_field.py --validate   # + logs, cross-well

>>> THE FIELD HAS NO TRUTH, SO NOTHING HERE IS "MODEL ERROR". <<<
Every synthetic ranking used MAPE/SSIM against a known model. That is gone. The
ONLY quantity measurable from field data alone is HOW WELL THE DATA IS FIT, so
that is what this ranks on: percent reduction in the misfit, first iteration to
last.

>>> AND A GOOD DATA FIT IS NOT A GOOD MODEL. <<<
An inversion can fit data beautifully by burying a wavelet error, a coupling
error or a 2-D/3-D mismatch in the velocity field. That is why the validation
block exists, and why it is reported SEPARATELY and never ranked on:

  CROSS-WELL     78A-32 and 78B-32 are INDEPENDENT datasets over shared
                 geology. Models inverted separately must agree where they
                 overlap. Needs no truth at all, and Park CANNOT do this --
                 they invert both wells together.
  ZONE DEPTHS    do recovered velocity jumps land on zones I/II/III?
  58-32 SONIC    >>> VALIDATION ONLY. NEVER A RANKING KEY. <<<

That last line is a methodological commitment, not a style preference. Our
mandate is Vp and Vs from DAS strain rate ALONE. The moment a log becomes a
selection criterion we are tuning to it, and the transferability claim -- the
entire point of the project -- is gone. So the log appears BELOW the ranking,
in its own block, and the code never sorts by it. Thresholds were fixed in
inversion/field_acceptance.py before any field result existed.

WHAT THE CAMPAIGN ASKS (groups, from submit_forge_field.sh):
  A vs B   Route B wave-equation xcorr starter  vs  first-break tomography.
           THE TRANSFERABILITY CLAIM: theirs needs 100 hand-picked gathers,
           ours needs none. A beating B on real data is publishable by itself.
  C        gc + traveltime -- the Park-comparable baseline.
  D        78B-32, the second well: cross-validation.
  E        ablations: no window, and the skip switch.
  S        optimizer sweep on the key cell.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(os.environ.get("DASFWI_RESULTS",
                           Path(__file__).resolve().parents[2] / "results"
                           / "standalone_field"))
#: zone I/II/III boundaries at FORGE, read off Park's figure (m below surface).
#: Site-specific by nature; override with --zones for another field.
ZONES_M = (150.0, 700.0)


def rows(root):
    out = []
    for d in sorted(Path(root).glob("*")):
        f = d / "metrics.json"
        if not f.is_file():
            continue
        try:
            m = json.loads(f.read_text())
        except json.JSONDecodeError:
            print(f"  [skip] {d.name}: unreadable metrics.json")
            continue
        m["_dir"] = d
        lo, hi = m.get("loss_first"), m.get("loss_last")
        # percent misfit reduction -- the one number the field can actually
        # supply. Park report 51.7% for INV2 on first-arrival mismatch; this is
        # the full waveform misfit, so it is a RELATED but not identical
        # quantity and is not claimed as a like-for-like comparison.
        m["drop_pct"] = (100.0 * (lo - hi) / lo
                         if lo and hi is not None and np.isfinite(lo)
                         and lo != 0 else float("nan"))
        m["group"] = _group(m)
        out.append(m)
    return out


def _group(m):
    """Which campaign question does this cell answer?"""
    st, ref, well = m.get("starting"), m.get("refiner"), m.get("well", "")
    if well.startswith("78B"):
        return "D 78B"
    if not m.get("window"):
        return "E abl"
    if str(m.get("arm", "")).startswith("switch"):
        return "E abl"
    if ref == "gc" and st == "traveltime":
        return "C park"
    if st == "traveltime":
        return "B tomo"
    if m.get("optimizer") not in (None, "nadam"):
        return "S opt"
    return "A routeB"


def final_vp(d):
    """Final Vp model from iter_vp.npz, or None."""
    f = Path(d) / "iter_vp.npz"
    if not f.is_file():
        return None
    with np.load(f) as z:
        a = z["data"]
    return np.asarray(a[-1] if a.ndim == 3 else a, float)


def table(rs):
    print(f"\n{'group':<9} {'tag':<46} {'it':>4} {'drop%':>7} "
          f"{'loss_first':>11} {'loss_last':>11} {'vp range':>15} {'h':>5}")
    print("-" * 118)
    for m in sorted(rs, key=lambda r: (-r["drop_pct"]
                                       if np.isfinite(r["drop_pct"]) else 1e9)):
        vr = m.get("vp_final_range") or [float("nan")] * 2
        flag = ""
        if m.get("diverged"):
            flag = "  <-- DIVERGED"
        elif not m.get("loss_decreased", True):
            flag = "  <-- loss ROSE"
        elif not m.get("complete", True):
            flag = f"  <-- incomplete ({m.get('iterations_done')})"
        print(f"{m['group']:<9} {m.get('tag','?'):<46} "
              f"{m.get('iterations_done', 0):>4} {m['drop_pct']:>7.1f} "
              f"{m.get('loss_first', float('nan')):>11.3e} "
              f"{m.get('loss_last', float('nan')):>11.3e} "
              f"{vr[0]:>6.0f}-{vr[1]:<8.0f} {m.get('runtime_h', 0):>5.2f}"
              f"{flag}")


def headline(rs):
    """A vs B: the transferability claim, at matched iterations."""
    print("\n=== A vs B: Route B (no picking) vs first-break tomography ===")
    a = {m["iterations"]: m for m in rs if m["group"] == "A routeB"}
    b = {m["iterations"]: m for m in rs if m["group"] == "B tomo"}
    both = sorted(set(a) & set(b))
    if not both:
        print("  (no matched-iteration pair -- cannot compare)")
        return
    for it in both:
        da, db = a[it]["drop_pct"], b[it]["drop_pct"]
        verdict = ("Route B WINS" if da > db else
                   "tomography wins" if db > da else "tie")
        print(f"  {it:>4} iters:  route_b {da:6.1f}%   traveltime {db:6.1f}%"
              f"   -> {verdict} ({da-db:+.1f} pts)")
    print("  Route B needs NO picked first breaks; the tomography starter needs"
          "\n  ~100 hand-picked gathers. Equal performance already favours it.")


def validate(rs, zones, dz, log_path=None):
    print("\n=== VALIDATION (never a ranking key) ===")
    from inversion.field_acceptance import (cross_validate, zone_boundaries,
                                            compare_to_log, ACCEPT)
    # ---- cross-well: independent data, shared geology --------------------- #
    a = next((m for m in rs if m["group"] == "A routeB"
              and m["iterations"] == 150), None)
    b = next((m for m in rs if m["group"] == "D 78B"
              and m["iterations"] == 150), None)
    if a and b:
        va, vb = final_vp(a["_dir"]), final_vp(b["_dir"])
        if va is not None and vb is not None and va.shape == vb.shape:
            cv = cross_validate(va, vb)
            ok = cv["rel_diff_pct"] <= ACCEPT["cross_well_rel_diff_pct"]
            print(f"  cross-well 78A vs 78B : {cv['rel_diff_pct']:.1f}% rel diff "
                  f"(threshold {ACCEPT['cross_well_rel_diff_pct']}%)  "
                  f"{'PASS' if ok else 'FAIL'}")
        else:
            print("  cross-well            : models missing or shape-mismatched")
    else:
        print("  cross-well            : need both 78A and 78B at 150 iters")

    # ---- zone boundaries --------------------------------------------------- #
    best = max((m for m in rs if np.isfinite(m["drop_pct"])),
               key=lambda m: m["drop_pct"], default=None)
    if best is None:
        return
    v = final_vp(best["_dir"])
    if v is None:
        print("  zones / log           : no iter_vp.npz for the best cell")
        return
    col = v[:, v.shape[1] // 2]
    z = np.arange(v.shape[0]) * dz
    got = zone_boundaries(col, z, n_zones=len(zones) + 1)
    print(f"  zone boundaries       : recovered {[f'{g:.0f}' for g in got]} m "
          f"vs expected {list(zones)} m  (tol {ACCEPT['boundary_tol_m']:.0f} m)")

    # ---- 58-32 sonic: VALIDATION ONLY -------------------------------------- #
    try:
        from forge.well_logs import load_58_32
        log = load_58_32(log_path)
        c = compare_to_log(col, z, log["z_m"], log["vp"])
        print(f"  58-32 sonic (Vp)      : rms {c['rms']:.0f} m/s, bias "
              f"{c['bias']:+.0f}, corr {c['corr']:.2f}, n={c['n']}   "
              f"[threshold {ACCEPT['log_rms_ms']:.0f} m/s]")
        print("     ^ VALIDATION ONLY -- never used to select a cell. Tuning to"
              " this log would\n       destroy the transferability claim.")
    except Exception as ex:                                    # noqa: BLE001
        print(f"  58-32 sonic           : unavailable ({type(ex).__name__}: {ex})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--validate", action="store_true",
                    help="also run cross-well / zone / log validation")
    ap.add_argument("--dz", type=float, default=20.0,
                    help="model dz in m; MUST match the run (driver default 20)")
    ap.add_argument("--zones", default=",".join(str(z) for z in ZONES_M))
    ap.add_argument("--log", default=None, help="path to the sonic LAS")
    args = ap.parse_args()

    rs = rows(args.root)
    if not rs:
        print(f"no metrics.json under {args.root}")
        return 1
    print(f"=== FORGE ACOUSTIC FIELD: {len(rs)} cells from {args.root} ===")
    table(rs)
    headline(rs)
    bad = [m for m in rs if m.get("diverged") or not m.get("model_finite", True)]
    if bad:
        print(f"\n  {len(bad)} cell(s) DIVERGED: "
              f"{', '.join(m.get('tag','?') for m in bad)}")
    if args.validate:
        zones = tuple(float(x) for x in args.zones.split(",") if x.strip())
        validate(rs, zones, args.dz, args.log)
    print("\nNOTE: ranking is by DATA FIT, the only field-measurable quantity."
          "\n      A good fit is not a good model -- see the validation block.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
