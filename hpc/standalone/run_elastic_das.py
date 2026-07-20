"""STANDALONE elastic (Vp + Vs + density) DAS-FWI script (novice-friendly).

WHAT THIS DOES
--------------
Joint Vp/Vs/density full-waveform inversion where the observed data are DAS
STRAIN RATES on vertical fibers. All three parameters are updated
simultaneously; each initial model is a 180 m x 180 m Gaussian smooth of the
corresponding TRUE Marmousi2 field (get_smooth_marmousi_model, kernel=4 on the
45 m grid = 180 m). NOTE: density is weakly constrained in FWI (strong
velocity/impedance trade-off, more so with limited-aperture DAS) - expect Vp
and Vs to recover well and rho to be the noisiest of the three; that is a known
FWI limitation, not a bug in this setup.
For a straight VERTICAL fiber the E3 endpoint
difference  eps_zz_rate = [v_z(z + l/2) - v_z(z - l/2)] / l  is exact in
elastic media too, so the same differentiable operator consumes the elastic
propagator's v_x / v_z records - again with NO strain-to-velocity
conversion anywhere; autograd builds the elastic adjoint through the layer.

Model/survey settings follow Liu's iso-elastic Marmousi2 example
(examples/elastic/Iso-elastic-Marmousi2-shotTop-recTop): 78 x 200 grid at
45 m, f0 = 3 Hz integrated Ricker, nt = 2500 x 3 ms, free surface,
nabc = 50, density INVERTED jointly (true = Marmousi2 rho, initial = 180 m
smooth), water layer pinned to truth and masked from updates, fd_order = 4,
checkpoint_segments = 4, StepLR(100, 0.75), 300 iterations.

The inversion loop lives IN THIS FILE (ADFWI's ElasticFWI is read-only for
this project and has no DAS path); it reproduces Liu's loop structure and
adds two constraints this project established experimentally:
  1. Poisson stability: vs <= vp / 1.5 enforced after every update
     (independent vp/vs steps otherwise drift below vp/vs = sqrt(2),
     negative Poisson's ratio, and the scheme diverges);
  2. optional gradient-proportional updates for "sgd" (Liu's GradProcessor
     norm_grad semantics: peak cell moves lr_scale * vmax per iteration).

HOW TO RUN (edit PARAMETERS, then):
    python hpc/standalone/run_elastic_das.py
    python hpc/standalone/run_elastic_das.py --misfit gc --optimizer adam
    python hpc/standalone/run_elastic_das.py --smoke
"""

# ============================================================================
# [0] PATHS (identical scheme to run_acoustic_das.py)
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
MARMOUSI_DIR = Path(os.environ.get(
    "MARMOUSI_DIR", REPO.parent / "Data_downloads" / "marmousi2"))
OUT_ROOT = Path(os.environ.get(
    "DASFWI_RESULTS", REPO / "results" / "standalone_elastic"))

for _p in (str(ADFWI_ROOT), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import json
import time

import numpy as np
import torch
from scipy import integrate

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ADFWI.model import IsotropicElasticModel
from ADFWI.survey import Source, Receiver, Survey
from ADFWI.propagator import ElasticPropagator
from ADFWI.utils import wavelet
from ADFWI.utils.velocityDemo import (load_marmousi_model,
                                      resample_marmousi_model,
                                      get_smooth_marmousi_model)
from ADFWI.fwi.misfit import (Misfit_waveform_L2, Misfit_envelope,
                              Misfit_global_correlation, Misfit_weighted_ECI)

from das.geometry import FiberGeometry, merge_fibers
from das.das_layer import DASObservationLayer
from inversion import config          # single source of truth for techniques
from inversion.safe_misfits import apply_misfit   # NIM dispatch in the loop
from inversion.preconditioner import illumination_weight   # depth precond

# ============================================================================
# [1] PARAMETERS
# ============================================================================
MISFIT = "gc"          # l2 | envelope | gc | sdtw | sinkhorn | weci
OPTIMIZER = "adam"     # sgd | adagrad | adam | adamw | nadam
ITERATIONS = 300

# --- grid / time / source: Liu's iso-elastic Marmousi2 header ---------------
NZ, NX = 78, 200
DX = DZ = 45.0
NT, DT = 2500, 0.003               # 7.5 s records
F0 = 3.0                           # [Hz], integrated Ricker
NABC, FREE_SURFACE = 50, True
X0_SECTION = 5000.0
SRC_EVERY, SRC_Z = 5, 2
WATER_ROWS = 10                    # top rows: init pinned to truth + no grad
FD_ORDER = 4
CHECKPOINT_SEGMENTS = 4

# --- DAS acquisition ---------------------------------------------------------
GAUGE_L = 2 * DZ                   # 90 m gauge on the 45 m grid
FIBER_X_INDICES = (25, 75, 125, 175)
FIBER_Z_TOP_INDEX = 12             # 540 m: below the water layer
FIBER_N_CHANNELS = 62              # nodes 12..73 -> z = 540..3285 m

# --- update rules -------------------------------------------------------------
MIN_VP_VS = 1.5                    # Poisson stability clamp (DO NOT REMOVE)
SGD_LR_SCALE = 0.01                # sgd: peak update = SGD_LR_SCALE * vmax
SCHEDULER = dict(step_size=100, gamma=0.75)
CACHE_EVERY = 10

# techniques from the single source of truth (inversion/config.py). RUN_SETTINGS
# only needs batch_size/normalize here (the elastic loop uses the module
# CHECKPOINT_SEGMENTS constant); the shared dict's extra key is harmless.
MISFITS = config.MISFITS
RUN_SETTINGS = config.MISFIT_SETTINGS
OPTIMIZERS = config.LIU_OPTIMIZERS


def build_misfit(name, iterations):
    return config.build_misfit(name, dt=DT, iterations=iterations)


def pick_device(arg=None):
    if arg:
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ============================================================================
# [2] MODEL SOURCE - Marmousi2 elastic benchmark (Liu's prep, incl. pinning
# the water rows of the initial model to the true values). For synthetic /
# field cases follow the CASE B / CASE C notes in run_acoustic_das.py.
# ============================================================================
def load_models():
    """True and 180 m-smoothed-initial vp, vs, rho on Liu's elastic section.
    kernel=4 on the 45 m grid = a 180 m x 180 m Gaussian window; the same
    smoother produces the density initial (get_smooth_marmousi_model smooths
    vp, vs AND rho). Water rows of every initial are pinned to truth."""
    marmousi = load_marmousi_model(in_dir=str(MARMOUSI_DIR))
    x = np.linspace(X0_SECTION, X0_SECTION + DX * NX, NX)
    z = np.linspace(0, DZ * NZ, NZ)
    true_model = resample_marmousi_model(x, z, marmousi)
    smooth = get_smooth_marmousi_model(true_model, gaussian_kernel=4,
                                       mask_extra_detph=2, rcv_depth=8)
    true = {k: np.asarray(true_model[k].T, np.float64) for k in ("vp", "vs", "rho")}
    init = {k: np.asarray(smooth[k].T, np.float64) for k in ("vp", "vs", "rho")}
    for k in ("vp", "vs", "rho"):
        init[k][:WATER_ROWS] = true[k][:WATER_ROWS]     # Liu pins the water rows
    return (true["vp"], true["vs"], true["rho"],
            init["vp"], init["vs"], init["rho"])


def build_model(vp, vs, rho, bounds, grad, device):
    """grad=True makes vp, vs AND rho inversion parameters. bounds is
    (vp_bound, vs_bound, rho_bound). auto_update_rho stays False so density is
    a genuine free parameter, not re-derived from vp."""
    vp_b, vs_b, rho_b = bounds
    return IsotropicElasticModel(
        0, 0, NX, NZ, DX, DZ, vp=vp, vs=vs, rho=rho,
        vp_bound=vp_b, vs_bound=vs_b, rho_bound=rho_b,
        vp_grad=grad, vs_grad=grad, rho_grad=grad,
        free_surface=FREE_SURFACE, abc_type="PML",
        abc_jerjan_alpha=0.007, nabc=NABC, auto_update_rho=False,
        device=device, dtype=torch.float32)


def build_acquisition(device):
    src_x = np.arange(2, NX - 2, SRC_EVERY)
    _, src_v = wavelet(NT, DT, F0, amp0=1)
    src_v = integrate.cumtrapz(src_v, axis=-1, initial=0)
    source = Source(nt=NT, dt=DT, f0=F0)
    for ix in src_x:
        source.add_source(src_x=int(ix), src_z=SRC_Z, src_wavelet=src_v,
                          src_type="mt",
                          src_mt=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    geometry = merge_fibers([
        FiberGeometry(x_well=ix * DX, z_top=FIBER_Z_TOP_INDEX * DZ,
                      n_channels=FIBER_N_CHANNELS, l=GAUGE_L,
                      dx=DX, dz=DZ, snap_to_nodes=False)
        for ix in FIBER_X_INDICES])
    rcv = Receiver(nt=NT, dt=DT)
    rcv_z = np.array([kz for (kz, _kx) in geometry.rcv_pos])
    rcv_x = np.array([kx for (_kz, kx) in geometry.rcv_pos])
    rcv.add_receivers(rcv_x, rcv_z, "vz")
    layer = DASObservationLayer(geometry,
                                output="strain_rate").to(torch.float32).to(device)
    return Survey(source, rcv), layer


def normalize_traces(d):
    """Per-trace max-abs normalization (AcousticFWI._normalize equivalent)."""
    mask = torch.sum(torch.abs(d), axis=1, keepdim=True) == 0
    mx = torch.max(torch.abs(d), axis=1, keepdim=True).values
    return d / mx.masked_fill(mask, 1)


# ============================================================================
# main
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--misfit", default=MISFIT, choices=MISFITS)
    ap.add_argument("--optimizer", default=OPTIMIZER,
                    choices=sorted(OPTIMIZERS))
    ap.add_argument("--iterations", type=int, default=ITERATIONS)
    ap.add_argument("--device", default=None)
    ap.add_argument("--precond", choices=["illum", "off"], default="illum",
                    help="illum = divide gradient by source illumination "
                         "(diagonal-Hessian preconditioner, lifts deep cells); "
                         "off = baseline (water mask only)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = pick_device(args.device)
    iterations = 2 if args.smoke else args.iterations
    tag = (f"das_{args.misfit}_{args.optimizer}_"
           + ("illum" if args.precond == "illum" else "noillum")
           + ("_smoke" if args.smoke else ""))
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== elastic {tag} on {device}, {iterations} iterations ===",
          flush=True)

    vp_true, vs_true, rho_true, vp_init, vs_init, rho_init = load_models()
    survey, layer = build_acquisition(device)
    n_shots = survey.source.num
    settings = RUN_SETTINGS[args.misfit]
    batch = settings["batch_size"] or n_shots

    # [4] observed data: inverse crime from the TRUE elastic model (variable rho)
    true_model = build_model(vp_true, vs_true, rho_true, (None, None, None),
                             grad=False, device=device)
    prop = ElasticPropagator(true_model, survey, device=device,
                             dtype=torch.float32)
    with torch.no_grad():
        rec = prop.forward(model=true_model, fd_order=FD_ORDER,
                           checkpoint_segments=CHECKPOINT_SEGMENTS)
        obs = layer(rec["vx"], rec["vz"]).cpu()
    print(f"observed {tuple(obs.shape)}, max|.| {float(obs.abs().max()):.3e}",
          flush=True)
    assert torch.isfinite(obs).all()

    # [5] inversion loop (Liu's structure + Poisson clamp; see module doc).
    # vp, vs, rho are inverted jointly; each has its own [min,max] bound.
    bounds = ([float(vp_true.min()), float(vp_true.max())],
              [float(vs_true.min()), float(vs_true.max())],
              [float(rho_true.min()), float(rho_true.max())])
    model = build_model(vp_init, vs_init, rho_init, bounds, grad=True,
                        device=device)
    prop = ElasticPropagator(model, survey, device=device,
                             dtype=torch.float32)
    optimizer = OPTIMIZERS[args.optimizer]([model.vp, model.vs, model.rho])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **SCHEDULER)
    misfit = build_misfit(args.misfit, iterations)

    grad_mask = torch.ones((NZ, NX), device=device)
    grad_mask[:WATER_ROWS, :] = 0

    losses, iter_vp, iter_vs, iter_rho = [], [], [], []
    t0 = time.time()
    for it in range(iterations):
        optimizer.zero_grad()
        loss_iter = 0.0
        illum = None                                         # source illumination
        for b0 in range(0, n_shots, batch):
            shot_index = np.arange(b0, min(b0 + batch, n_shots))
            rec = prop.forward(model=model, shot_index=shot_index,
                               fd_order=FD_ORDER,
                               checkpoint_segments=CHECKPOINT_SEGMENTS)
            syn = layer(rec["vx"], rec["vz"]).cpu()
            o = obs[shot_index]
            if settings["normalize"]:
                syn, o = normalize_traces(syn), normalize_traces(o)
            loss = apply_misfit(misfit, syn, o)   # dispatches NIM's .apply
            loss.backward()
            loss_iter += float(loss)
            if args.precond == "illum":                      # accumulate diag(H)
                fw = (rec["forward_wavefield_vx"]
                      + rec["forward_wavefield_vz"]).detach()
                illum = fw if illum is None else illum + fw
        with torch.no_grad():
            weight = (illumination_weight(illum) if args.precond == "illum"
                      and illum is not None else None)
            for par in (model.vp, model.vs, model.rho):
                par.grad *= grad_mask                        # Liu's mask
                if weight is not None:                       # lift deep cells
                    par.grad *= weight
                if args.optimizer == "sgd":                  # norm_grad
                    peak = par.grad.abs().max().clamp_min(1e-30)
                    par.grad *= float(par.detach().max()) / peak
        optimizer.step()
        scheduler.step()
        model.forward()                                      # clip to bounds
        with torch.no_grad():                                # Poisson clamp
            model.vs.data = torch.minimum(model.vs.data,
                                          model.vp.data / MIN_VP_VS)
        losses.append(loss_iter)
        if it % CACHE_EVERY == 0 or it == iterations - 1:
            iter_vp.append(model.vp.detach().cpu().numpy().copy())
            iter_vs.append(model.vs.detach().cpu().numpy().copy())
            iter_rho.append(model.rho.detach().cpu().numpy().copy())
        print(f"iter {it}: loss {loss_iter:.6f} "
              f"({(time.time()-t0)/(it+1):.0f}s/iter)", flush=True)

    hours = (time.time() - t0) / 3600.0

    # [6] outputs
    np.savez(out_dir / "iter_vp.npz", data=np.asarray(iter_vp))
    np.savez(out_dir / "iter_vs.npz", data=np.asarray(iter_vs))
    np.savez(out_dir / "iter_rho.npz", data=np.asarray(iter_rho))
    np.savez(out_dir / "iter_loss.npz", data=np.asarray(losses))
    vp_final = model.vp.detach().cpu().numpy()
    vs_final = model.vs.detach().cpu().numpy()
    rho_final = model.rho.detach().cpu().numpy()

    metrics = dict(tag=tag, device=device, iterations=iterations,
                   runtime_h=round(hours, 3),
                   losses_finite=bool(np.isfinite(losses).all()))
    triplet = (("vp", vp_true, vp_init, vp_final),
               ("vs", vs_true, vs_init, vs_final),
               ("rho", rho_true, rho_init, rho_final))
    for nm, tru, ini, fin in triplet:
        dt_, di = tru - ini, fin - ini
        denom = np.sqrt((dt_ ** 2).sum() * (di ** 2).sum())
        metrics[f"rms_init_{nm}"] = float(np.sqrt(((ini - tru) ** 2).mean()))
        metrics[f"rms_final_{nm}"] = float(np.sqrt(((fin - tru) ** 2).mean()))
        metrics[f"update_corr_{nm}"] = (float((dt_ * di).sum() / denom)
                                        if denom else 0.0)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)

    # 3 rows (vp/vs/rho) x 3 cols (true/initial/inverted); each row on its own
    # colour scale (rho ~2000-2600 kg/m^3, velocities in m/s).
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), constrained_layout=True)
    ext = [0, (NX - 1) * DX / 1000, (NZ - 1) * DZ / 1000, 0]
    units = {"vp": "m/s", "vs": "m/s", "rho": "kg/m^3"}
    for r, (nm, tru, ini, fin) in enumerate(triplet):
        for c, (d, ttl) in enumerate([(tru, "true"), (ini, "initial"),
                                      (fin, "inverted")]):
            im = axes[r, c].imshow(d, extent=ext, cmap="jet",
                                   vmin=tru.min(), vmax=tru.max())
            axes[r, c].set(title=f"{nm} {ttl} [{units[nm]}]", xlabel="x [km]",
                           ylabel="z [km]")
            fig.colorbar(im, ax=axes[r, c], shrink=0.8)
    fig.savefig(out_dir / "final.png", dpi=150)
    print("saved results to", out_dir, flush=True)


if __name__ == "__main__":
    main()
