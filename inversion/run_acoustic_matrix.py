"""Acoustic optimizer x misfit test matrix on the Marmousi2 crop.

Runs the SAME acoustic DAS inverse-crime setup (identical crop, fiber, shots,
bands, smoothing, fixed true density) under every combination of:

    optimizers : adamw (Adam with weight_decay=0), sgd (gradient-proportional
                 classic FWI update via GradProcessor norm_grad)
    misfits    : gc        - global correlation (dtype-safe GCMisfit64)
                 weci      - hybrid Envelope->GC sigmoid schedule built from
                             dasfwi's ScheduledMisfit (T6), N = total iters
                 sinkhorn  - Wasserstein sinkhorn divergence (dtype-safe
                             subclass; blur/scaling PROVISIONAL per spec T6)

6 combos, run sequentially; each saves losses + inverted vp + metrics, and a
summary table/figure across the matrix is written at the end.

Usage:
    python inversion/run_acoustic_matrix.py            # full (~4-6 h CPU)
    python inversion/run_acoustic_matrix.py --quick    # 2-iter debug pass

Results -> results/acoustic_matrix/<optimizer>_<misfit>/ + summary.json.
"""

import json
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

from ADFWI.fwi.misfit import (Misfit_global_correlation, Misfit_envelope,
                              Misfit_wasserstein_sinkhorn)
from geomloss import SamplesLoss

from inversion.run_inverse_crime import run_inverse_crime, GCMisfit64
from inversion.misfit_schedule import ScheduledMisfit, weci_sigmoid_weights
from inversion.run_marmousi_demo import load_crop, NZ, NX

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results",
                       "acoustic_matrix")

BANDS_FULL = [(5.0, 12), (10.0, 13)]     # 25 iterations, 5 -> 10 Hz
BANDS_QUICK = [(5.0, 1), (10.0, 1)]      # debug pass
# quick uses the FULL record length: a short window leaves deep channels
# with no arrival (dead traces), which is a different physical regime than
# the full run and produced misleading failures during bring-up
NT_FULL, NT_QUICK = 2500, 2500
DT = 4e-4

# lr semantics differ by optimizer (see run_inverse_crime optimizer note):
# adam moves ~lr m/s per cell per iteration; sgd with norm_grad moves the
# PEAK cell lr*vmax m/s (vmax ~ 3550 here -> 0.004 * 3550 ~ 14 m/s).
# "adam" = torch.optim.Adam exactly as in Liu's ADFWI examples (he uses
# lr=10 over 300 iterations with dense surface receivers; 2.0 here for a
# 25-iteration single-fiber run).
OPTIMIZERS = {"adam": 2.0, "sgd": 0.004}
GRAD_MASK_TOP = 10                        # Liu's examples: mask top 10 rows
FIBER_X_INDEX = 150                       # x_well 750 m / dx 5


# SinkhornSafe lives in inversion/safe_misfits.py (shared by the HPC
# campaign scripts); re-exported here for backward compatibility.
from inversion.safe_misfits import SinkhornSafe  # noqa: F401,E402


def build_misfit(name, total_iters):
    if name == "gc":
        return GCMisfit64(dt=1)
    if name == "weci":
        # hybrid: Envelope-dominated early -> GC-dominated late (T6, E10);
        # component order: index 0 gets the growing sigmoid weight (GC)
        return ScheduledMisfit(
            components=[Misfit_global_correlation(dt=1),
                        Misfit_envelope(dt=1, p=1.5)],
            weight_fn=weci_sigmoid_weights(total_iters))
    if name == "sinkhorn":
        # EXACTLY Liu's working configuration (ADFWI examples/acoustic/
        # 02-misfit-functions-test/01-Marmousi2-Test/
        # 02_inversion_WassersteinSinkhorn.py L119):
        #   Misfit_wasserstein_sinkhorn(dt=0.01, sparse_sampling=2, p=1,
        #                               blur=1e-2)  [scaling left at 0.5]
        # verified on our real gathers; the spec-T6-guessed (p=2, blur=0.1,
        # dt=true-dt) settings NaN'd geomloss's epsilon schedule
        return SinkhornSafe(dt=0.01, sparse_sampling=2, p=1, blur=1e-2)
    raise ValueError(name)


def corridor_metrics(vp_true, vp_init, vp_final):
    """RMS + update correlation, split near/far of the fiber column."""
    dv, dtru = vp_final - vp_init, vp_true - vp_init
    cols = np.abs(np.arange(vp_true.shape[1]) - FIBER_X_INDEX)
    out = {}
    for tag, sel in (("all", cols >= 0), ("near_fiber", cols < 15),
                     ("far", cols >= 15)):
        a, b = dtru[:, sel].ravel(), dv[:, sel].ravel()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        out[tag] = dict(
            rms_init=float(np.sqrt(((vp_init - vp_true)[:, sel] ** 2).mean())),
            rms_final=float(np.sqrt(((vp_final - vp_true)[:, sel] ** 2).mean())),
            update_corr=float((a * b).sum() / denom) if denom > 0 else 0.0)
    return out


def main(quick=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    bands = BANDS_QUICK if quick else BANDS_FULL
    nt = NT_QUICK if quick else NT_FULL
    total_iters = sum(n for _, n in bands)

    vp_true, rho_true = load_crop()
    vp_init = gaussian_filter(vp_true, sigma=12.0)

    summary = {}
    for opt, lr in OPTIMIZERS.items():
        for mis_name in ("gc", "weci", "sinkhorn"):
            tag = f"{opt}_{mis_name}"
            print(f"=== {tag} ===", flush=True)
            t0 = time.time()
            misfit = build_misfit(mis_name, total_iters)
            result = run_inverse_crime(dict(
                vp_true=vp_true, vp_init=vp_init,
                nt=nt, dt=DT,
                fiber=dict(x_well=750.0, z_top=200.0, n_channels=100),
                shots=dict(x_indices=(20, 70, 120, 180, 240, 280),
                           z_index=2, f0=10.0),
                optimizer=opt, lr=lr,
                misfit="gc" if mis_name == "gc" else misfit,
                bands=bands,
                dtype=torch.float32,
                vp_bound=(1400.0, 3700.0),
                rho=rho_true,
                grad_mask_top=GRAD_MASK_TOP,
                # sinkhorn scales globally inside SinkhornSafe; per-trace
                # normalization is numerically unsafe for it (see the class)
                waveform_normalize=(mis_name != "sinkhorn"),
            ))
            mins = (time.time() - t0) / 60.0
            m = corridor_metrics(vp_true, vp_init, result["vp_final"])
            m["runtime_min"] = round(mins, 1)
            m["losses"] = [float(x) for x in result["iter_loss"]]
            summary[tag] = m

            d = os.path.join(OUT_DIR, tag)
            os.makedirs(d, exist_ok=True)
            np.savez(os.path.join(d, "result.npz"),
                     vp_final=result["vp_final"],
                     losses=np.asarray(result["iter_loss"]))
            print(f"{tag}: {mins:.1f} min, "
                  f"RMS {m['all']['rms_init']:.1f} -> {m['all']['rms_final']:.1f}, "
                  f"corr {m['all']['update_corr']:.3f} "
                  f"(near {m['near_fiber']['update_corr']:.3f} / "
                  f"far {m['far']['update_corr']:.3f})", flush=True)

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # matrix figure: inverted vp per combo + loss curves
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), constrained_layout=True)
    ext = [0, (NX - 1) * 5 / 1000, (NZ - 1) * 5 / 1000, 0]
    for r, opt in enumerate(OPTIMIZERS):
        for c, mis_name in enumerate(("gc", "weci", "sinkhorn")):
            tag = f"{opt}_{mis_name}"
            vpf = np.load(os.path.join(OUT_DIR, tag, "result.npz"))["vp_final"]
            im = axes[r, c].imshow(vpf, extent=ext, cmap="jet",
                                   vmin=vp_true.min(), vmax=vp_true.max())
            mm = summary[tag]["all"]
            axes[r, c].set_title(
                f"{tag}  RMS {mm['rms_init']:.0f}->{mm['rms_final']:.0f} "
                f"corr {mm['update_corr']:.2f}")
            fig.colorbar(im, ax=axes[r, c], shrink=0.8)
    fig.savefig(os.path.join(OUT_DIR, "matrix_vp.png"), dpi=150)

    fig2, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for tag, m in summary.items():
        ls = np.asarray(m["losses"])
        ax.plot(ls / max(1e-30, abs(ls[0])), ".-", label=tag)
    ax.set(title="loss (normalized to |iter 0|)", xlabel="iteration")
    ax.legend(fontsize=8)
    fig2.savefig(os.path.join(OUT_DIR, "matrix_losses.png"), dpi=150)
    print("summary + figures saved to", OUT_DIR, flush=True)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
