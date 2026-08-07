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
#: Zone I/II/III boundaries, m below SURFACE, quoted verbatim from Park et al.
#: (TLE 44(4), doi 10.1190/tle44040256.1): "Zone I extends from the surface to
#: approximately 0.35 km depth, zone II from 0.35 km to around 1 km, and zone
#: III beyond 1 km depth." I had 150/700 m read off their figure by eye, which
#: was wrong by 200 and 300 m -- enough to fail a 100 m tolerance on a model
#: that was actually correct. Read the text, not the picture.
#: NOTE Park's Figure 9 shows these at ~0.45 and ~1.1 km because their section
#: INCLUDES the air layer; these values are below-surface, matching our chan_z.
#: Site-specific by nature -- override with --zones at any other field.
ZONES_M = (350.0, 1000.0)
#: Corroboration from a second, independent source: the 78B-32 cuttings log puts
#: the top of the granitoid at 823 m, inside zone II->III as defined above.
GRANITOID_TOP_78B_M = 823.0


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
        # >>> A CORRELATION MISFIT IS NEGATIVE, AND MORE NEGATIVE IS BETTER. <<<
        # gc went -3.669e-02 -> -2.754e+01, a 750x better fit, and the old
        # formula 100*(lo-hi)/lo printed -74963%, ranking the best-behaved cells
        # dead last. Normalise by |lo| so the sign of the loss cannot invert the
        # verdict, and mark sign-varying misfits so they are not compared
        # against norm-type ones -- a 750x correlation gain and an 88% L2
        # reduction are different quantities and ranking them together is
        # meaningless whichever formula is used.
        m["neg_misfit"] = bool(lo is not None and lo < 0)
        if lo and hi is not None and np.isfinite(lo) and np.isfinite(hi) and lo != 0:
            m["drop_pct"] = 100.0 * (lo - hi) / abs(lo)
        else:
            m["drop_pct"] = float("nan")
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


def roughness(vp, dz=10.0):
    """RMS second vertical derivative, normalised -- 1/m, scale-free.

    The critique of the first campaign was "rougher than Park, short-wavelength
    oscillations the data do not resolve". That is a real observation and it was
    made BY EYE. Eyeballing is how I got Park's zone depths wrong by 200-300 m.
    So: measure it, put it in the table, and let the gradient-smoothing A/B be
    decided by a number instead of an impression.
    """
    v = np.asarray(vp, float)
    if v.ndim != 2 or v.shape[0] < 3:
        return float("nan")
    d2 = np.diff(v, n=2, axis=0) / (dz ** 2)
    return float(np.sqrt(np.nanmean(d2 ** 2)) / max(np.nanmean(np.abs(v)), 1e-9))


def table(rs):
    print(f"\n{'group':<9} {'tag':<46} {'it':>4} {'drop%':>7} "
          f"{'rough':>8} {'smooth':>7} {'loss_last':>11} {'vp range':>15} {'h':>5}")
    print("-" * 122)
    for m in sorted(rs, key=lambda r: (bool(r.get("neg_misfit")),
                                       -r["drop_pct"]
                                       if np.isfinite(r["drop_pct"]) else 1e9)):
        vr = m.get("vp_final_range") or [float("nan")] * 2
        flag = ""
        # A correlation misfit is unbounded below, so its "drop%" is not on the
        # same scale as a norm's and the two must not be read as a ranking.
        # Marked rather than hidden: the number is still the right measure of
        # that cell's own progress.
        if m.get("neg_misfit"):
            flag += "  [corr misfit -- drop% NOT comparable to norm cells]"
        if m.get("diverged"):
            flag = "  <-- DIVERGED"
        elif not m.get("loss_decreased", True):
            flag = "  <-- loss ROSE"
        elif not m.get("complete", True):
            flag = f"  <-- incomplete ({m.get('iterations_done')})"
        if "_rough" not in m:
            v = final_vp(m["_dir"])
            m["_rough"] = roughness(v) if v is not None else float("nan")
        gs = "lam/4" if m.get("grad_smooth", "none") != "none" else "none"
        print(f"{m['group']:<9} {m.get('tag','?'):<46} "
              f"{m.get('iterations_done', 0):>4} {m['drop_pct']:>7.1f} "
              f"{m['_rough']:>8.2e} {gs:>7} "
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


def smoothing_ab(rs):
    """Group C (lambda/4 smoothed) against group G (identical, unsmoothed)."""
    print("\n=== gradient smoothing A/B (identical cells but for --grad-smooth) ===")
    pairs = {}
    for m in rs:
        key = (m.get("well"), m.get("refiner"), m.get("starting"),
               m.get("iterations"), bool(m.get("window")))
        pairs.setdefault(key, {})[m.get("grad_smooth", "none") != "none"] = m
    any_pair = False
    for key, d in sorted(pairs.items(), key=lambda kv: str(kv[0])):
        if True in d and False in d:
            any_pair = True
            a, b = d[True], d[False]
            print(f"  {key[1]} {key[3]:>3} it:  roughness "
                  f"smoothed {a['_rough']:.2e}  vs  none {b['_rough']:.2e}  "
                  f"({100*(b['_rough']-a['_rough'])/max(b['_rough'],1e-30):+.0f}% )"
                  f"   |  data fit {a['drop_pct']:.1f}% vs {b['drop_pct']:.1f}%")
    if not any_pair:
        print("  (no matched smoothed/unsmoothed pair found)")
    print("  A smoother model that fits the data WORSE is not automatically "
          "better;\n  read both columns before concluding.")


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
    ap.add_argument("--dz", type=float, default=None,
                    help="model dz in m. Default: READ FROM metrics.json, which "
                         "is authoritative. Only needed for runs predating that "
                         "field.")
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
    smoothing_ab(rs)
    if args.validate:
        zones = tuple(float(x) for x in args.zones.split(",") if x.strip())
        dz = args.dz
        if dz is None:
            dzs = {m.get("dz") for m in rs if m.get("dz")}
            if len(dzs) > 1:
                print(f"  *** MIXED GRIDS in one directory: dz = {sorted(dzs)}. "
                      f"These runs are NOT comparable; separate them before "
                      f"reading anything below.")
            dz = sorted(dzs)[0] if dzs else 20.0
            print(f"  (dz = {dz:g} m, from metrics.json)")
        validate(rs, zones, dz, args.log)
    print("\nNOTE: ranking is by DATA FIT, the only field-measurable quantity."
          "\n      A good fit is not a good model -- see the validation block.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
