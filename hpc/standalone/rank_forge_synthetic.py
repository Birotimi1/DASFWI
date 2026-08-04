"""Read the FORGE synthetic campaign.

>>> SSIM IS DEGENERATE ON THIS PROBLEM -- DO NOT RANK BY IT. <<<
Measured across all 16 cells: SSIM falls monotonically from iteration 0, so
"best SSIM" trivially selects the STARTING MODEL and the metric says "never
invert". A smooth 1-D ramp is structurally similar to a smooth layered truth, so
ANY added detail -- right or wrong -- lowers SSIM. It is reported last, in
brackets, and never sorted on. My first version led with it and produced a table
whose headline read "@it 0" for every cell, which is worse than useless.

RANK ON DEPTH-RESOLVED ERROR, because the campaign's central finding is
depth-dependent: the acoustic inversion CORRUPTS the shallow section (fitting
surface waves it cannot model) while genuinely improving at depth. One number
cannot express that, and any single-number ranking hides it.

The DATA-FIT change is printed alongside, because a low model error with a poor
data fit -- or the reverse -- is the signature of an inversion that has absorbed
an error somewhere invisible. That matters most for `l2` under a mismatched
wavelet: L2 fits amplitude AND phase, so it can bury the wavelet error in the
model and still look good on a single metric.

    python hpc/standalone/rank_forge_synthetic.py
    python hpc/standalone/rank_forge_synthetic.py --mismatched-only
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(os.environ.get("DASFWI_RESULTS", "results")) / "forge_synthetic"
SHALLOW_M = 400.0          # below the ground: where surface waves do the damage


def _rows(root):
    out = []
    for mf in sorted(root.glob("*/metrics.json")):
        if "smoke" in mf.parent.name:
            continue
        m = json.loads(mf.read_text())
        npz = mf.parent / "vp.npz"
        if not npz.is_file():
            continue
        d = np.load(npz)
        vt, vi, vp = d["vp_true"], d["vp_init"], d["vp"]
        nair = int(m.get("n_air_rows", 0))
        ns = nair + int(SHALLOW_M / 10.0)
        e = lambda a, s: float(np.abs(a[s] - vt[s]).mean())
        l0, l1 = m.get("loss_first"), m.get("loss_last")
        # A PERCENTAGE IS MEANINGLESS WHEN THE LOSS CROSSES ZERO. gc (global
        # CORRELATION) starts near +9 and ends near -400: more negative is
        # BETTER, and dividing by |l0|~9 produced "-10097%", which reads as a
        # catastrophic misfit when gc in fact improved. Percentages are reported
        # ONLY when the sign is preserved; otherwise the raw endpoints are
        # shown, because an uninterpretable number is worse than a longer one.
        if l0 in (None, 0) or l1 is None:
            red, cross = float("nan"), False
        elif (l0 > 0) != (l1 > 0):
            red, cross = float("nan"), True            # sign flip -> no %
        else:
            red, cross = 100.0 * (abs(l0) - abs(l1)) / abs(l0), False
        out.append(dict(
            arm=m.get("arm"), win=bool(m.get("window", False)),
            mism=bool(m.get("wavelet_mismatched", False)),
            bands=len(m.get("bands", [None])) > 1, it=m["iterations_done"],
            sh0=e(vi, slice(nair, ns)), sh1=e(vp, slice(nair, ns)),
            dp0=e(vi, slice(ns, None)), dp1=e(vp, slice(ns, None)),
            mape=m.get("mape"), ssim=m.get("ssim"), lossred=red,
            div=m.get("diverged", False), tag=mf.parent.name,
            l0=l0, l1=l1, cross=cross))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    ap.add_argument("--mismatched-only", action="store_true", dest="mism_only")
    args = ap.parse_args()
    rows = _rows(Path(args.results) if args.results else ROOT)
    if not rows:
        return print("no cells found")
    if args.mism_only:
        rows = [r for r in rows if r["mism"]]
    rows.sort(key=lambda r: r["sh1"])          # rank on SHALLOW error

    print(f"{'arm':8s} {'win':4s} {'mis':4s} {'ms':3s} {'it':>4s} "
          f"{'SHALLOW err':>15s} {'DEEP err':>15s} {'MAPE%':>6s} "
          f"{'dFit%':>6s}  (ssim)")
    print("-" * 92)
    for r in rows:
        sa = "->" if r["sh1"] <= r["sh0"] else "^^"       # ^^ = got WORSE
        da = "->" if r["dp1"] <= r["dp0"] else "^^"
        mape = r["mape"] if r["mape"] is not None else float("nan")
        print(f"{str(r['arm']):8s} {'yes' if r['win'] else '-':4s} "
              f"{'yes' if r['mism'] else '-':4s} {'yes' if r['bands'] else '-':3s} "
              f"{r['it']:4d} {r['sh0']:6.0f}{sa}{r['sh1']:6.0f}  "
              f"{r['dp0']:6.0f}{da}{r['dp1']:6.0f}  {mape:6.1f} "
              + (f"{'sign':>6s}" if r["cross"] else f"{r['lossred']:6.1f}")
              + f"  ({r['ssim']:.3f})"
              + ("  *** DIVERGED" if r["div"] else ""))

    print("\n'sign' in dFit = the loss changed sign (gc is a CORRELATION: it "
          "starts near\nzero and goes negative, so a percentage is "
          "uninterpretable). Raw values below.")
    print("\n^^ = error INCREASED.  SSIM is bracketed and never sorted on: it "
          "falls\nmonotonically from iteration 0 here, so ranking by it would "
          "say 'never invert'.")

    print("\nWINDOWING, paired per refiner (an unpaired cell is not a comparison):")
    byref = {}
    for r in rows:
        if r["mism"] and not r["bands"] and r["arm"] in ("l2", "gc", "convsi"):
            byref.setdefault(r["arm"], {})[r["win"]] = r
    paired = False
    for arm, p in sorted(byref.items()):
        if True in p and False in p:
            paired = True
            d = p[True]["sh1"] - p[False]["sh1"]
            print(f"   {arm:8s} shallow {p[False]['sh1']:6.0f} -> "
                  f"{p[True]['sh1']:6.0f}  ({d:+.0f} m/s)  "
                  f"{'HELPS' if d < 0 else 'NO HELP'}")
        else:
            print(f"   {arm:8s} INCOMPLETE -- "
                  f"{'window' if False in p else 'no-window'} cell still running")
    if not paired:
        print("   (no complete pairs yet)")

    solo = [r for r in rows if r["arm"] in ("l2", "gc", "convsi")
            and r["mism"] and not r["bands"]]
    if solo:
        print("\nREFINER under a MISMATCHED wavelet -- this decides the field run:")
        for r in sorted(solo, key=lambda x: x["sh1"]):
            fit = ("loss %.3g -> %.3g (sign change: %% is meaningless)"
                   % (r["l0"], r["l1"]) if r["cross"]
                   else "dFit %+6.1f%%" % r["lossred"])
            print(f"   {r['arm']:8s} {'win' if r['win'] else '   '}  "
                  f"shallow {r['sh1']:6.0f}   deep {r['dp1']:6.0f}   "
                  f"MAPE {r['mape']:5.1f}%   {fit}")
        print("   Judge shallow AND deep AND the data fit TOGETHER. A good model "
              "error with a\n   poor data fit (or the reverse) means the "
              "inversion hid the wavelet error --\n   which is precisely what "
              "l2 can do, because it fits amplitude as well as phase.")


if __name__ == "__main__":
    main()
