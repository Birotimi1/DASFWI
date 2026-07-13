"""Starting-model degradation ladder (cycle-skipping robustness experiment).

The autograd/adjoint machinery does NOT relax FWI's dependence on a good
starting model - that comes from the MISFIT and multiscale scheduling. This
experiment quantifies it: invert the SAME Marmousi DAS strain-rate data from
progressively worse starting models, with each cycle-skipping-robust misfit,
and measure how far down the ladder each misfit can still recover the model.

Axes:
  starting models (increasingly poor): mild smooth -> strong smooth ->
    1-D depth gradient (no lateral structure) -> constant mean velocity.
  misfits: l2 (the cycle-skipping baseline that should fail first) vs the
    robust set gc / envelope / sinkhorn / nim / (optionally sdtw, traveltime,
    weci). Optimizer fixed to sgd (gradient-proportional, our best) and
    multiscale bands (low-frequency first) so the ONLY variable that explains
    a cell's success is the misfit's basin of attraction.

Metric per cell: recovery = 1 - RMS(vp_final - vp_true)/RMS(vp_init - vp_true)
  (fraction of the initial model error removed; >0 improves, <0 worsens) plus
  the update correlation. The heatmap of recovery over (misfit x starting
  model) shows each misfit's cycle-skipping tolerance.

Usage:
    python inversion/run_starting_model_ladder.py            # full (HPC)
    python inversion/run_starting_model_ladder.py --quick    # local wiring
    python inversion/run_starting_model_ladder.py --misfits gc,envelope,nim
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from ADFWI.fwi.misfit import Misfit_waveform_L2, Misfit_envelope
from inversion.run_inverse_crime import run_inverse_crime, GCMisfit64
from inversion.safe_misfits import (SinkhornSafe, SdtwSafe, TravelTimeSafe,
                                    make_nim)
from inversion.misfit_schedule import ScheduledMisfit, weci_sigmoid_weights
from inversion.run_marmousi_demo import load_crop, NZ, NX

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results",
                       "starting_model_ladder")

# starting-model ladder (worse going down); each maps vp_true -> vp_init
def _ladder(vp_true):
    nx = vp_true.shape[1]
    zmean = vp_true.mean(axis=1, keepdims=True)          # 1-D depth profile
    return [
        ("smooth_s8",  gaussian_filter(vp_true, 8.0)),
        ("smooth_s16", gaussian_filter(vp_true, 16.0)),
        ("smooth_s32", gaussian_filter(vp_true, 32.0)),
        ("grad_1d",    np.repeat(gaussian_filter(zmean, 16.0), nx, axis=1)),
        ("const",      np.full_like(vp_true, vp_true.mean())),
    ]

ALL_MISFITS = ("l2", "gc", "envelope", "sinkhorn", "nim", "sdtw",
               "traveltime", "weci")
DEFAULT_MISFITS = ("l2", "gc", "envelope", "sinkhorn", "nim")   # fast-ish set


def build_misfit(name, total_iters, dt):
    if name == "l2":
        return Misfit_waveform_L2(dt=dt), True
    if name == "gc":
        return GCMisfit64(dt=1), True
    if name == "envelope":
        return Misfit_envelope(dt=dt, p=1.5), True
    if name == "sinkhorn":
        return SinkhornSafe(dt=0.01, sparse_sampling=2, p=1, blur=1e-2), False
    if name == "nim":
        return make_nim(p=1, dt=dt), True
    if name == "sdtw":
        return SdtwSafe(gamma=1, sparse_sampling=2, dt=dt), True
    if name == "traveltime":
        return TravelTimeSafe(dt=dt, beta=10), True
    if name == "weci":
        return ScheduledMisfit([GCMisfit64(dt=1), Misfit_envelope(dt=dt, p=1.5)],
                               weight_fn=weci_sigmoid_weights(total_iters)), True
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--misfits", default=",".join(DEFAULT_MISFITS))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    misfits = [m for m in args.misfits.split(",") if m]
    for m in misfits:
        if m not in ALL_MISFITS:
            raise SystemExit(f"unknown misfit {m!r}; choices {ALL_MISFITS}")
    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available()
                             else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)

    vp_true, rho_true = load_crop()
    if args.quick:
        vp_true, rho_true = vp_true[:60, :60], rho_true[:60, :60]
    ladder = _ladder(vp_true)
    bands = [(5.0, 1), (10.0, 1)] if args.quick else [(5.0, 15), (10.0, 15)]
    nt = 700 if args.quick else 2500
    total_iters = sum(n for _, n in bands)
    dt = 4e-4
    fiber = dict(x_well=(30.0 if args.quick else 150.0) * 5.0,
                 z_top=100.0, n_channels=10 if args.quick else 100)
    shots = dict(x_indices=(10, 45) if args.quick else (20, 90, 160, 240),
                 z_index=2, f0=10.0)

    grid = {}     # misfit -> {starting_label -> metrics}
    for mname in misfits:
        grid[mname] = {}
        for slabel, vp_init in ladder:
            misfit, normalize = build_misfit(mname, total_iters, dt)
            t0 = time.time()
            res = run_inverse_crime(dict(
                vp_true=vp_true, vp_init=np.asarray(vp_init, np.float64),
                nt=nt, dt=dt, fiber=fiber, shots=shots,
                optimizer="sgd", lr=0.004,
                misfit=misfit, bands=bands, dtype=torch.float32,
                vp_bound=(1400.0, 3700.0), rho=rho_true,
                grad_mask_top=5, waveform_normalize=normalize,
                device=device))
            vpf = res["vp_final"]
            rms_i = float(np.sqrt(((vp_init - vp_true) ** 2).mean()))
            rms_f = float(np.sqrt(((vpf - vp_true) ** 2).mean()))
            dtru, dinv = vp_true - vp_init, vpf - vp_init
            den = np.sqrt((dtru ** 2).sum() * (dinv ** 2).sum())
            grid[mname][slabel] = dict(
                rms_init=rms_i, rms_final=rms_f,
                recovery=float(1.0 - rms_f / rms_i) if rms_i > 0 else 0.0,
                update_corr=float((dtru * dinv).sum() / den) if den > 0 else 0.0,
                runtime_s=round(time.time() - t0, 1))
            c = grid[mname][slabel]
            print(f"{mname:11s} {slabel:11s} recovery {c['recovery']:+.3f} "
                  f"corr {c['update_corr']:+.3f} ({c['runtime_s']:.0f}s)",
                  flush=True)

    with open(os.path.join(OUT_DIR, "ladder.json"), "w") as f:
        json.dump(grid, f, indent=2)

    # heatmap: recovery over (misfit rows x starting-model columns)
    slabels = [s for s, _ in ladder]
    M = np.array([[grid[m][s]["recovery"] for s in slabels] for m in misfits])
    fig, ax = plt.subplots(figsize=(1.6 * len(slabels) + 2, 0.7 * len(misfits) + 2),
                           constrained_layout=True)
    im = ax.imshow(M, cmap="RdYlGn", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(slabels)), slabels, rotation=30, ha="right")
    ax.set_yticks(range(len(misfits)), misfits)
    for i in range(len(misfits)):
        for j in range(len(slabels)):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("recovery = 1 - RMS_final/RMS_init  (green=recovers, red=fails)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(os.path.join(OUT_DIR, "ladder_heatmap.png"), dpi=150)
    print("saved ladder.json + ladder_heatmap.png to", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
