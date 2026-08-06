"""Park-style figure of our FORGE result, with the 58-32 sonic beside it.

Reproduces the layout of Park et al.'s VM0/VM3 comparison so the two can be put
side by side without the reader having to reconcile different conventions:
same colour range (1.6-5.6 km/s), same well markers, the sonic log overlaid in
a white inset at 58-32, and the I/II/III zone boundaries as red dashed lines.

GEOMETRY, measured from the SEG-Y headers and the LAS lat/lon rather than read
off their figure:
    58-32  +328 m along-section from the 78A-32 wellhead (68 m off-line)
    78A-32    0 m  (our origin)
    78B-32  -82 m  (22 m off-line)
Park's figure shows 58->78A ~400 m and 78A->78B ~100 m, so the spacings agree.
Our PCA section axis points the opposite way, so `--flip` mirrors x to match
their left-to-right ordering (58-32, then 78A, then 78B).

THE AIR LAYER IS MASKED so the topographic surface reads as a surface, exactly
as in their figure -- plotting 340 m/s air as a velocity would dominate the
colour scale and hide the section.

>>> The sonic is VALIDATION ONLY: never a constraint, never a tuning target.
And it starts at 655 m, so it says NOTHING about the near surface -- which is
where our synthetic showed the error concentrates. Do not read agreement in the
granitoid as agreement everywhere. <<<
"""

# --- import bootstrap: depend on NO environment ----------------------------- #
# `python forge/preflight.py` puts forge/ on sys.path, NOT the repo root, so
# ADFWI/forge/inversion are unimportable unless PYTHONPATH happens to be set.
# It was set in every shell I tested in and NOT in the user's cluster shell, so
# this died there with a bare ModuleNotFoundError. Resolve from __file__
# instead: walk up to the directory holding forge/ and inversion/, and add the
# tracked ADFWI package next to it.
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

#: Park's colour range, km/s -- identical so the figures are comparable by eye
VMIN_KMS, VMAX_KMS = 1.6, 5.6
#: zone boundaries from their density logs (I/II/III), metres below datum
ZONE_M = (450.0, 1100.0)
#: along-section positions, metres from the 78A-32 wellhead (MEASURED)
WELLS = {"58-32": 328.0, "78A-32": 0.0, "78B-32": -82.0}
V_AIR_MAX = 400.0
#: Park's scale runs BLUE (1.6, slow) -> green -> BROWN (5.6, fast). Plain `jet`
#: is blue->red; their brown top end needs an explicit map, and getting this
#: backwards (jet_r) made our figure unreadable beside theirs.
PARK_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "park", ["#2020c0", "#2090f0", "#40d0d0", "#60d060", "#c8c840",
             "#b08030", "#804010"])


def _load(run_dir):
    d = np.load(Path(run_dir) / "vp.npz")
    m = json.loads((Path(run_dir) / "metrics.json").read_text())
    return d, m


def _sonic():
    """(depth_m, vp_ms) for 58-32, or None if the LAS is unavailable."""
    try:
        from forge.well_logs import load_58_32
        L = load_58_32()
        z = L.get("tvd_m", L["z_m"])
        v = L["vp"]
        ok = np.isfinite(z) & np.isfinite(v)
        return z[ok], v[ok]
    except Exception:                                    # noqa: BLE001
        return None


def panel(ax, vp, dx, dz, title, flip=False, sonic=None, show_zones=True):
    """One Park-style velocity panel."""
    nz, nx = vp.shape
    v = np.array(vp, float)
    v[v <= V_AIR_MAX] = np.nan                 # mask air -> topography shows
    x0 = -(nx // 2) * dx
    ext = [x0 / 1000.0, (x0 + nx * dx) / 1000.0, nz * dz / 1000.0, 0.0]
    if flip:
        v = v[:, ::-1]
        ext = [-ext[1], -ext[0], ext[2], ext[3]]
    im = ax.imshow(v / 1000.0, extent=ext, aspect="auto", cmap=PARK_CMAP,
                   vmin=VMIN_KMS, vmax=VMAX_KMS, interpolation="bilinear")
    if show_zones:
        for z in ZONE_M:
            ax.axhline(z / 1000.0, color="red", ls="--", lw=1.0, alpha=0.8)
        edges = [0.0] + [z / 1000.0 for z in ZONE_M] + [ext[2]]
        for lab, a, b in zip(("I", "II", "III"), edges[:-1], edges[1:]):
            ax.text(ext[0] + 0.03, (a + b) / 2, lab, style="italic",
                    fontsize=10, va="center", zorder=6)
    for name, xm in WELLS.items():
        xk = (-xm if flip else xm) / 1000.0
        ax.plot([xk], [0.02], "o", ms=7, mfc="yellow", mec="k", zorder=5,
                clip_on=False)
        # 78A and 78B are only 82 m apart, so stagger the labels vertically
        dy = 9 + (12 if name == "78B-32" else 0)
        ax.annotate(name, (xk, 0.0), xytext=(0, dy), textcoords="offset points",
                    ha="center", fontsize=8, zorder=6)
        ax.plot([xk, xk], [0.05, ext[2] * 0.75], "k-" if name != "58-32"
                else "k:", lw=1.2, zorder=4)
    # --- the sonic, in a white inset at 58-32, exactly as Park draw it -------
    if sonic is not None:
        zl, vl = sonic
        xk = (-WELLS["58-32"] if flip else WELLS["58-32"]) / 1000.0
        halfw = 0.13                     # narrower: it was masking the section
        lo, hi = 2000.0, 7000.0                         # Park's 2.0-7.0 km/s box
        keep = (zl / 1000.0) <= ext[2]
        zl, vl = zl[keep], vl[keep]
        xs = xk + (np.clip(vl, lo, hi) - lo) / (hi - lo) * 2 * halfw - halfw
        ax.add_patch(plt.Rectangle((xk - halfw, zl.min() / 1000.0), 2 * halfw,
                                   (zl.max() - zl.min()) / 1000.0,
                                   fc="white", ec="k", lw=0.6, zorder=7))
        ax.plot(xs, zl / 1000.0, color="purple", lw=0.6, zorder=8)
        # ABOVE the box, not inside it: inside, the text sat on the purple
        # curve; below the axis it collided with "Distance (km)".
        ax.text(xk, zl.min() / 1000.0 - 0.03, "58-32 sonic  2.0-7.0 km/s",
                ha="center", va="bottom", fontsize=6, zorder=9,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax.set_xlabel("Distance (km)"); ax.set_ylabel("Depth (km)")
    ax.set_title(title, loc="right", fontsize=10, fontweight="bold")
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--compare", default=None,
                    help="second run to stack beneath (Park stack VM3 over VM0)")
    ap.add_argument("--initial", action="store_true",
                    help="show OUR starting model as the lower panel, the way "
                         "Park show VM0 -- so the reader sees what the "
                         "inversion actually added")
    ap.add_argument("--dx", type=float, default=10.0)
    ap.add_argument("--dz", type=float, default=10.0)
    ap.add_argument("--flip", action="store_true", default=True)
    ap.add_argument("--no-flip", action="store_false", dest="flip")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d, m = _load(args.run_dir)
    son = _sonic()
    if son is None:
        print("NOTE: no 58-32 LAS found -- plotting without the sonic overlay "
              "(set FORGE_DAS_DIR)", flush=True)

    panels = [(d["vp"], f"OURS  {m.get('arm','?')}/{m.get('optimizer','?')} "
                        f"i{m.get('iterations_done','?')}")]
    if args.initial and "vp_init" in d:
        panels.append((d["vp_init"], "OUR STARTING MODEL"))
    elif args.compare:
        d2, m2 = _load(args.compare)
        panels.append((d2["vp"], f"{m2.get('arm','?')}/{m2.get('optimizer','?')} "
                                 f"i{m2.get('iterations_done','?')}"))

    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 4.2 * len(panels)),
                             constrained_layout=True, squeeze=False)
    for ax, (vp, t) in zip(axes[:, 0], panels):
        im = panel(ax, vp, args.dx, args.dz, t, flip=args.flip, sonic=son)
    fig.colorbar(im, ax=axes[:, 0].tolist(), label="Vp (km/s)", shrink=0.85)
    out = Path(args.out) if args.out else Path(args.run_dir) / "park_style.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
