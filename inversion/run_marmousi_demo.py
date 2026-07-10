"""Downscaled LOCAL Marmousi2 DAS inverse-crime demo.

NOT the spec's full Marmousi2 reproduction (that is HPC work: full model,
tens of shots, hundreds of multiscale iterations). This local demo validates
the same chain on a structurally rich 1.5 x 1.0 km crop of Marmousi2
(x = 10-11.5 km, z = 0.45-1.45 km; vp 1500-3550 m/s) at the project's 5 m
grid: synthetic vertical DAS fiber, 6 surface shots, 2 multiscale bands
(5 Hz then 10 Hz), GC misfit, AdamW. float32 (ADFWI's own published
precision; the float64 correctness gates in tests/ validate the stack).

The 1.25 m native Marmousi2 grid is decimated by exactly 4 -> 5 m, so no
interpolation is involved. Results (npz + pngs) go to results/marmousi_demo/.
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from ADFWI.utils.velocityDemo import load_marmousi_model
from inversion.run_inverse_crime import run_inverse_crime

MARMOUSI_DIR = os.path.join(os.path.dirname(__file__), "..", "..",
                            "Data_downloads", "marmousi2")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results",
                       "marmousi_demo")

X0_M, Z0_M = 10000.0, 450.0        # crop origin in Marmousi2 coordinates
NZ, NX = 201, 301                  # 1.0 km x 1.5 km at 5 m
STEP = 4                           # 1.25 m * 4 = 5 m exactly


def load_crop():
    m = load_marmousi_model(MARMOUSI_DIR)
    ix0, iz0 = int(X0_M / 1.25), int(Z0_M / 1.25)
    sl = (slice(ix0, ix0 + (NX - 1) * STEP + 1, STEP),
          slice(iz0, iz0 + (NZ - 1) * STEP + 1, STEP))
    vp = np.asarray(m["vp"][sl].T, dtype=np.float64)
    rho = np.asarray(m["rho"][sl].T, dtype=np.float64)
    return vp, rho


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # rho: Marmousi2's TRUE density, fixed and shared by the observed-data
    # and inversion models. The first run of this demo derived obs-rho from
    # vp_true but inversion-rho from vp_init (Gardner each), and vp absorbed
    # the resulting amplitude error: GC loss fell while model RMS ROSE.
    vp_true, rho_true = load_crop()
    vp_init = gaussian_filter(vp_true, sigma=12.0)   # smooth starting model
    print(f"vp_true {vp_true.shape}: {vp_true.min():.0f}-{vp_true.max():.0f}",
          flush=True)

    t0 = time.time()
    result = run_inverse_crime(dict(
        vp_true=vp_true,
        vp_init=vp_init,
        nt=2500, dt=4e-4,
        fiber=dict(x_well=750.0, z_top=200.0, n_channels=100),
        shots=dict(x_indices=(20, 70, 120, 180, 240, 280), z_index=2,
                   f0=10.0),
        lr=10.0,
        misfit="gc",
        bands=[(5.0, 12), (10.0, 13)],       # multiscale: 5 Hz then 10 Hz
        dtype=torch.float32,
        vp_bound=(1400.0, 3700.0),
        rho=rho_true,
    ))
    mins = (time.time() - t0) / 60.0

    losses = np.asarray(result["iter_loss"])
    vp_final = result["vp_final"]
    err_init = np.sqrt(np.mean((vp_init - vp_true) ** 2))
    err_final = np.sqrt(np.mean((vp_final - vp_true) ** 2))
    d_true = vp_true - vp_init
    d_inv = vp_final - vp_init
    corr = float(np.corrcoef(d_true.ravel(), d_inv.ravel())[0, 1])

    print(f"runtime {mins:.1f} min", flush=True)
    print("losses:", np.array2string(losses, precision=4), flush=True)
    print(f"model RMS error: init {err_init:.1f} -> final {err_final:.1f} m/s",
          flush=True)
    print(f"update-vs-true-update correlation: {corr:.3f}", flush=True)

    np.savez(os.path.join(OUT_DIR, "marmousi_demo_result.npz"),
             vp_true=vp_true, vp_init=vp_init, vp_final=vp_final,
             losses=losses)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    ext = [0, (NX - 1) * 5 / 1000, (NZ - 1) * 5 / 1000, 0]
    for ax, (data, title) in zip(
            axes.flat[:3],
            [(vp_true, "true"), (vp_init, "initial"), (vp_final, "inverted")]):
        im = ax.imshow(data, extent=ext, cmap="jet",
                       vmin=vp_true.min(), vmax=vp_true.max())
        ax.set(title=f"vp {title} [m/s]", xlabel="x [km]", ylabel="z [km]")
        fig.colorbar(im, ax=ax, shrink=0.8)
    axes.flat[3].plot(losses, "k.-")
    axes.flat[3].set(title="GC loss", xlabel="iteration")
    fig.savefig(os.path.join(OUT_DIR, "marmousi_demo.png"), dpi=150)
    print("saved:", os.path.join(OUT_DIR, "marmousi_demo.png"), flush=True)


if __name__ == "__main__":
    main()
