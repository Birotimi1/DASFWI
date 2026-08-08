"""Regenerate field figures from saved models -- NO GPU, NO re-inversion.

    python hpc/standalone/replot_field.py                    # every cell
    python hpc/standalone/replot_field.py --root <dir>

WHY THIS EXISTS
The 18-cell campaign completed, wrote every metrics.json and every
iter_vp.npz, and produced ZERO figures: the plotting block raised KeyError on
bundle["grid"]["src_z_grid"] (the key is bundle["src_z_grid"]) after the
results had been saved. About 60 SU of finished inversions with nothing to look
at.

The models were never lost -- only the rendering was. So this reads what is
already on disk and draws it. Re-running the inversion to recover a PNG would
be spending a GPU allocation on a matplotlib bug.

Park's conventions, so our sections can sit beside theirs: air white, blue
slow -> brown fast on a FIXED scale, depth clipped at the deepest receiver.
"""

# --- import bootstrap: depend on NO environment ----------------------------- #
import sys as _sys
from pathlib import Path as _Path

for _r in _Path(__file__).resolve().parents:
    if (_r / "forge").is_dir() and (_r / "inversion").is_dir():
        for _p in (_r, _r / "ADFWI_local"):
            if (_p / "ADFWI").is_dir() and str(_p) not in _sys.path:
                _sys.path.insert(0, str(_p))
        if str(_r) not in _sys.path:
            _sys.path.insert(0, str(_r))
        break
# ---------------------------------------------------------------------------- #
import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def replot(run_dir, force=False):
    """Draw final.png for one result directory. Returns a status string."""
    d = Path(run_dir)
    mf, vf = d / "metrics.json", d / "iter_vp.npz"
    if not (mf.is_file() and vf.is_file()):
        return "skip (no metrics.json / iter_vp.npz)"
    if (d / "final.png").is_file() and not force:
        return "exists"
    m = json.loads(mf.read_text())
    with np.load(vf) as z:
        vp = np.asarray(z["data"], float)
    if vp.ndim != 3 or vp.shape[0] < 1:
        return f"skip (iter_vp shape {vp.shape})"
    # iter_vp[0] is the model BEFORE the first update -- the starting model as
    # the inversion actually saw it, which is what belongs beside the result.
    vp_init, vp_final = vp[0], vp[-1]

    dz = float(m.get("dz") or 10.0)
    dx = float(m.get("dx") or dz)
    lo, hi = (m.get("vp_bound") or [1000.0, 6000.0])[:2]
    z_max_km = (float(m["chan_z_max"]) / 1000.0
                if m.get("chan_z_max") else None)

    # air mask from the model itself: the starting model carries the air layer
    # at V_AIR, so the boundary is measurable rather than assumed. Runs made
    # before n_air_rows was recorded are handled the same way.
    from inversion.near_surface import V_AIR
    air = vp_init < (V_AIR * 1.5)
    ground = np.array([np.argmax(~air[:, j]) * dz if air[:, j].any() else 0.0
                       for j in range(vp_init.shape[1])])

    from forge.plot_field_result import velocity_panel
    loss = None
    lf = d / "iter_loss.npz"
    if lf.is_file():
        with np.load(lf) as zz:
            loss = np.asarray(zz["data"], float)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, (v, ttl) in zip(axes[:2],
                            [(vp_init, f"initial ({m.get('starting','?')})"),
                             (vp_final, f"inverted {m.get('tag', d.name)}")]):
        im = velocity_panel(ax, v, dx, dz, vmin=lo, vmax=hi, ground=ground,
                            z_max_km=z_max_km, title=f"vp {ttl} [m/s]")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Vp [m/s]")
    if loss is not None and loss.size:
        axes[2].plot(loss, "k.-")
        axes[2].set(title="loss", xlabel="iteration")
    else:
        axes[2].axis("off")
    fig.savefig(d / "final.png", dpi=150)
    plt.close(fig)
    return "OK"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None,
                    help="results directory (default: results/standalone_field)")
    ap.add_argument("--force", action="store_true",
                    help="redraw even where final.png already exists")
    args = ap.parse_args()
    root = Path(args.root or (Path(__file__).resolve().parents[2]
                              / "results" / "standalone_field"))
    if not root.is_dir():
        print(f"no such directory: {root}")
        return 2
    n_ok = 0
    for d in sorted(root.glob("*")):
        if not d.is_dir():
            continue
        st = replot(d, force=args.force)
        if st == "OK":
            n_ok += 1
        print(f"  {st:<38} {d.name}")
    print(f"\n{n_ok} figure(s) written under {root}")
    return 0 if n_ok else 1


if __name__ == "__main__":
    _sys.exit(main())
