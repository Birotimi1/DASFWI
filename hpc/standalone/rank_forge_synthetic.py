"""Read the FORGE synthetic campaign, INCLUDING the optimal stopping point.

The campaign's headline finding is that shallow error grows MONOTONICALLY with
iterations while the loss falls -- the acoustic code inventing near-surface
velocity to explain surface waves it cannot model. The 30-iteration cells beat
the 150-iteration ones. So the final model is NOT the best model, and reporting
only the endpoint would hide that. `iter_vp.npz` keeps the trajectory, so the
best iteration is recoverable after the fact.

    python hpc/standalone/rank_forge_synthetic.py
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from inversion.metrics import model_scores            # noqa: E402

ROOT = Path(os.environ.get("DASFWI_RESULTS", "results")) / "forge_synthetic"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    args = ap.parse_args()
    root = Path(args.results) if args.results else ROOT
    rows = []
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
        sh = slice(nair, nair + 40)                    # ~400 m below ground
        err = lambda a, s: float(np.abs(a[s] - vt[s]).mean())
        # best iteration from the cached trajectory, if present
        best_it, best_ssim = m["iterations_done"], m["ssim"]
        tj = mf.parent / "iter_vp.npz"
        if tj.is_file():
            try:
                traj = np.load(tj)["data"]
                if traj.ndim == 3 and len(traj) > 1:
                    ss = [model_scores(vt, f)["ssim"] for f in traj]
                    k = int(np.nanargmax(ss))
                    best_ssim = float(ss[k])
                    best_it = int(round(k * m["iterations_done"] / max(len(ss) - 1, 1)))
            except Exception:                          # noqa: BLE001
                pass
        rows.append(dict(tag=mf.parent.name, arm=m.get("arm"),
                         window=m.get("window", False),
                         mism=m.get("wavelet_mismatched", False),
                         it=m["iterations_done"], ssim=m["ssim"],
                         best_ssim=best_ssim, best_it=best_it,
                         sh0=err(vi, sh), sh1=err(vp, sh),
                         div=m.get("diverged", False)))
    if not rows:
        return print(f"no cells in {root}")
    rows.sort(key=lambda r: -r["best_ssim"])
    print(f"{'arm':9s} {'win':4s} {'mism':5s} {'SSIM_end':>8s} {'SSIM_best':>9s} "
          f"{'@it':>5s} {'shallow err':>16s}")
    print("-" * 72)
    for r in rows:
        arrow = "WORSE" if r["sh1"] > r["sh0"] else "better"
        print(f"{str(r['arm']):9s} {'yes' if r['window'] else '-':4s} "
              f"{'yes' if r['mism'] else '-':5s} {r['ssim']:8.3f} "
              f"{r['best_ssim']:9.3f} {r['best_it']:5d} "
              f"{r['sh0']:6.0f}->{r['sh1']:6.0f} {arrow:>6s}"
              + ("   *** DIVERGED" if r["div"] else ""))
    print("\nSSIM_best vs SSIM_end: if best << end the run is DEGRADING, and the")
    print("stopping point matters more than the misfit choice.")
    w = [r for r in rows if r["window"]]; nw = [r for r in rows if not r["window"]]
    if w and nw:
        dw = np.mean([r["sh1"] - r["sh0"] for r in w])
        dn = np.mean([r["sh1"] - r["sh0"] for r in nw])
        print(f"\nWINDOWING: mean shallow change {dn:+.0f} m/s without, "
              f"{dw:+.0f} m/s with -> windowing "
              f"{'HELPS' if dw < dn else 'does NOT help'} here.")
        print("(Prediction recorded before any FORGE run: `w` helps here where "
              "it HURT on Marmousi, because a noiseless synthetic has no "
              "surface waves to window out.)")


if __name__ == "__main__":
    main()
