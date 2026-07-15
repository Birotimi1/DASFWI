"""STANDALONE FORGE field DAS-FWI script (real 78A-32 / 78B-32 strain rate).

WHAT THIS DOES
--------------
Full-waveform inversion of P-velocity from the REAL FORGE walkaway-VSP DAS
strain-rate data, through the exact E3 gauge operator (both wells are
vertical, so E3 is complete - see forge/field_loader.py). No strain->velocity
conversion: the differentiable DAS layer sits between the propagator and the
misfit, and autograd builds the adjoint through it.

Unlike the Marmousi/synthetic scripts there is NO true model, so RMS-vs-truth
metrics are meaningless and omitted; the run starts from a 1-D velocity
gradient (or --starting traveltime) and reports the loss trajectory and the
inverted model. Two field realities to keep in mind: (a) the true source
wavelet is unknown - a placeholder Ricker is used, but --misfit convsi
(source-INDEPENDENT convolved-wavefields, Choi & Alkhalifah 2011) cancels the
unknown source entirely and is the recommended field misfit; (b) the 3-D
walkaway is projected onto a 2-D section (out-of-plane offset dropped) for
ADFWI's 2-D code.

HOW TO RUN (edit the PARAMETERS block, then):
    python hpc/standalone/run_field_das.py --well 78A-32 --misfit gc
    python hpc/standalone/run_field_das.py --smoke --shots 4     # wiring check
    python hpc/standalone/run_field_das.py --well 78B-32 --shots 60

Defaults are a COARSE grid (fast wiring); set --dz 5 --dt 4e-4 --nt 6000 for
a production-resolution run (HPC).
"""

# ============================================================================
# [0] PATHS
# ============================================================================
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_adfwi_default = REPO.parent / "ADFWI"
ADFWI_ROOT = Path(os.environ.get(
    "ADFWI_ROOT",
    _adfwi_default if (_adfwi_default / "ADFWI").is_dir()
    else REPO / "ADFWI_local"))
OUT_ROOT = Path(os.environ.get(
    "DASFWI_RESULTS", REPO / "results" / "standalone_field"))

for _p in (str(ADFWI_ROOT), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import json
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ADFWI.model import AcousticModel
from ADFWI.propagator import AcousticPropagator, GradProcessor
from ADFWI.fwi import AcousticFWI
from ADFWI.fwi.misfit import (Misfit_waveform_L2, Misfit_envelope,
                              Misfit_global_correlation, Misfit_weighted_ECI)

from inversion import config          # single source of truth for techniques
from forge.field_loader import load_forge_field, summarize

# ============================================================================
# [1] PARAMETERS
# ============================================================================
WELL = "78A-32"        # 78A-32 (1010 ch) | 78B-32 (1206 ch)
N_SHOTS = 20           # walkaway shots to load
MISFIT = "gc"          # l2 | envelope | gc | sdtw | sinkhorn | weci
OPTIMIZER = "adam"     # sgd | adagrad | adam | adamw | nadam
ITERATIONS = 200

# --- grid / time (COARSE default = fast wiring; see the module docstring) ----
DZ = DX = 20.0                     # production: 5 m (gauge l = 2*dz = 10 m)
NT_MODEL, DT_MODEL = 1200, 1e-3    # production: ~6000, 4e-4 s (CFL at 5 m)
F0 = 15.0                          # PLACEHOLDER Ricker centre freq [Hz]
NABC = 30
F_ARRIVAL_PAD = 15                 # grid padding nodes

# --- starting model: 1-D vp gradient (no true model for field data) ----------
VP_TOP, VP_BOTTOM = 2000.0, 5500.0     # linear surface->deep [m/s]
VP_BOUND = (1500.0, 6000.0)

# --- inversion machinery -----------------------------------------------------
GRAD_MASK_TOP = 8
SCHEDULER = dict(step_size=100, gamma=0.75)
CACHE_EVERY = 10

# techniques from the single source of truth (inversion/config.py)
MISFITS = config.MISFITS
RUN_SETTINGS = config.MISFIT_SETTINGS
OPTIMIZERS = config.LIU_OPTIMIZERS
WELLS = ("78A-32", "78B-32")


def build_misfit(name, iterations, dt):
    return config.build_misfit(name, dt=dt, iterations=iterations)


def pick_device(arg=None):
    if arg:
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def gradient_start_model(nz, nx, dz):
    """1-D linear vp gradient VP_TOP -> VP_BOTTOM broadcast over x."""
    zcol = np.linspace(VP_TOP, VP_BOTTOM, nz)
    return np.repeat(zcol[:, None], nx, axis=1).astype(np.float64)


# ============================================================================
# main
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--well", default=WELL, choices=WELLS)
    ap.add_argument("--shots", type=int, default=N_SHOTS)
    ap.add_argument("--misfit", default=MISFIT, choices=MISFITS)
    ap.add_argument("--optimizer", default=OPTIMIZER, choices=sorted(OPTIMIZERS))
    ap.add_argument("--iterations", type=int, default=ITERATIONS)
    ap.add_argument("--dz", type=float, default=DZ)
    ap.add_argument("--dt", type=float, default=DT_MODEL)
    ap.add_argument("--nt", type=int, default=NT_MODEL)
    ap.add_argument("--f0", type=float, default=F0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--starting", default="gradient",
                    choices=("gradient", "traveltime"),
                    help="starting model: blind 1-D gradient, or data-driven "
                         "first-break traveltime tomography")
    ap.add_argument("--smoke", action="store_true", help="2-iteration check")
    args = ap.parse_args()

    device = pick_device(args.device)
    iterations = 2 if args.smoke else args.iterations
    tag = (f"field_{args.well}_{args.misfit}_{args.optimizer}"
           + ("_smoke" if args.smoke else ""))
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== FORGE field {tag} on {device}, {iterations} iterations ===",
          flush=True)

    # [3] load real field data (strain rate) + geometry
    bundle = load_forge_field(well=args.well, n_shots=args.shots,
                              dz=args.dz, dx=args.dz, nt_model=args.nt,
                              dt_model=args.dt, f0=args.f0, nabc=NABC,
                              pad_nodes=F_ARRIVAL_PAD, device=device)
    print(summarize(bundle), flush=True)
    g = bundle["grid"]
    nz, nx = g["nz"], g["nx"]

    # [4] starting model. Default is a blind 1-D gradient; "traveltime" builds
    # a data-driven v(z) from first breaks of the nearest-offset shot (VSP
    # check-shot method) -- a far better basin for FWI on an unknown site.
    if args.starting == "traveltime":
        from forge.traveltime_tomography import starting_model_from_gathers
        obs = np.asarray(bundle["obs_data"].data["strain_rate"])   # [S, nt, C]
        sx = np.asarray(bundle["src_x_grid"])
        near = int(np.argmin(np.abs(sx - bundle["well_x_index"])))  # min offset
        offset = abs(float(sx[near] - bundle["well_x_index"])) * g["dx"]
        vp_init, z_prof, v_prof, _ = starting_model_from_gathers(
            obs[near], g["dt"], np.asarray(bundle["channel_z_grid"]),
            x_offset=offset, nz=nz, nx=nx, dz=g["dz"],
            v_bounds=VP_BOUND, min_time_s=2 * g["dt"])
        # a STARTING model must be smooth/long-wavelength: heavy vertical
        # smoothing removes pick noise (and the sharp features that would make
        # some synthetic traces dead -> normalization NaNs). Best results need
        # a NEAR-offset shot; with only far-offset shots the picks are weak.
        from scipy.ndimage import gaussian_filter
        vp_init = gaussian_filter(vp_init, sigma=(max(3.0, 60.0 / g["dz"]), 0))
        vp_init = np.clip(vp_init, *VP_BOUND)
        print(f"traveltime starting model: shot {near} offset {offset:.0f} m, "
              f"v(z) {v_prof.min():.0f}-{v_prof.max():.0f} m/s over "
              f"{z_prof.min():.0f}-{z_prof.max():.0f} m "
              f"(nearest offset {offset:.0f} m; use a near-offset shot for a "
              f"reliable profile)", flush=True)
    else:
        vp_init = gradient_start_model(nz, nx, g["dz"])

    # [5] inversion (Liu's machinery through the T5-patched AcousticFWI)
    rho = np.power(vp_init, 0.25) * 310.0
    model = AcousticModel(0, 0, nx, nz, g["dx"], g["dz"], vp_init, rho,
                          vp_bound=list(VP_BOUND), vp_grad=True,
                          free_surface=True, abc_type="PML",
                          abc_jerjan_alpha=0.007, nabc=g["nabc"],
                          device=device, dtype=torch.float32)
    prop = AcousticPropagator(model, bundle["survey"], device=device,
                              dtype=torch.float32)
    optimizer = OPTIMIZERS[args.optimizer](model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **SCHEDULER)
    grad_mask = np.ones((nz, nx))
    grad_mask[:GRAD_MASK_TOP, :] = 0
    settings = RUN_SETTINGS[args.misfit]

    fwi = AcousticFWI(propagator=prop, model=model,
                      optimizer=optimizer, scheduler=scheduler,
                      loss_fn=build_misfit(args.misfit, iterations, g["dt"]),
                      obs_data=bundle["obs_data"],
                      gradient_processor=GradProcessor(grad_mask=grad_mask),
                      waveform_normalize=settings["normalize"],
                      cache_result=True, cache_result_epoch=CACHE_EVERY,
                      save_fig_epoch=-1,
                      das_layer=bundle["das_layer"], obs_key="strain_rate")
    t0 = time.time()
    fwi.forward(iteration=iterations,
                batch_size=settings["batch_size"],
                checkpoint_segments=settings["checkpoint_segments"])
    hours = (time.time() - t0) / 3600.0

    # [6] outputs (no RMS-vs-truth for field data)
    iter_loss = np.asarray(fwi.iter_loss)
    np.savez(out_dir / "iter_vp.npz", data=np.asarray(fwi.iter_vp))
    np.savez(out_dir / "iter_loss.npz", data=iter_loss)
    vp_final = model.vp.detach().cpu().numpy()
    grad_final = (model.vp.grad.detach().cpu().numpy()
                  if model.vp.grad is not None else np.zeros_like(vp_final))
    metrics = dict(
        tag=tag, well=args.well, n_shots=bundle["n_shots"], device=device,
        iterations=iterations, runtime_h=round(hours, 3),
        loss_first=float(iter_loss[0]), loss_last=float(iter_loss[-1]),
        loss_decreased=bool(iter_loss[-1] < iter_loss[0]),
        losses_finite=bool(np.isfinite(iter_loss).all()),
        grad_finite=bool(np.isfinite(grad_final).all()),
        grad_nonzero=bool(np.abs(grad_final).max() > 0),
        vp_final_range=[float(vp_final.min()), float(vp_final.max())])
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    ext = [0, (nx - 1) * g["dx"] / 1000, (nz - 1) * g["dz"] / 1000, 0]
    for ax, (d, ttl) in zip(axes[:2], [(vp_init, f"initial ({args.starting})"),
                                       (vp_final, f"inverted {tag}")]):
        im = ax.imshow(d, extent=ext, cmap="jet",
                       vmin=VP_BOUND[0], vmax=VP_BOUND[1])
        ax.set(title=f"vp {ttl} [m/s]", xlabel="x [km]", ylabel="z [km]")
        # mark the well
        ax.axvline(bundle["well_x_index"] * g["dx"] / 1000, color="w", ls="--", lw=1)
        fig.colorbar(im, ax=ax, shrink=0.8)
    axes[2].plot(iter_loss, "k.-")
    axes[2].set(title="loss", xlabel="iteration")
    fig.savefig(out_dir / "final.png", dpi=150)
    print("saved results to", out_dir, flush=True)


if __name__ == "__main__":
    main()
