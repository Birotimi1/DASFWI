"""STANDALONE acoustic DAS-FWI script (novice-friendly, Liu-example style).

WHAT THIS DOES
--------------
Full-waveform inversion of P-velocity where the observed data are DAS
STRAIN RATES on vertical fibers - measured natively by the E3 gauge
operator  eps_zz_rate = [v_z(z + l/2) - v_z(z - l/2)] / l  applied to the
propagator's velocity records. There is NO conversion of strain rate to
particle velocity anywhere: the differentiable operator sits between the
wave propagator and the misfit, and autograd builds the adjoint through it.

Everything else follows Liu's ADFWI acoustic Marmousi2 examples
(examples/acoustic/02-misfit-functions-test / 03-optimizer-test).

HOW TO RUN (edit the PARAMETERS block, then):
    python hpc/standalone/run_acoustic_das.py
    python hpc/standalone/run_acoustic_das.py --misfit gc --optimizer adam
    python hpc/standalone/run_acoustic_das.py --smoke        # 2-iter check

The command line only OVERRIDES the defaults set in the PARAMETERS block,
so a job script can stay one line. Results go to OUT_DIR (see below).
"""

# ============================================================================
# [0] PATHS - work on any machine; override by env var if the layout differs
# ============================================================================
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]           # the DASFWI repo root
_adfwi_default = REPO.parent / "ADFWI"               # ../ADFWI if present,
ADFWI_ROOT = Path(os.environ.get(                    # else the built-in
    "ADFWI_ROOT",                                    # ADFWI_local mirror
    _adfwi_default if (_adfwi_default / "ADFWI").is_dir()
    else REPO / "ADFWI_local"))
MARMOUSI_DIR = Path(os.environ.get(
    "MARMOUSI_DIR", REPO.parent / "Data_downloads" / "marmousi2"))
OUT_ROOT = Path(os.environ.get(
    "DASFWI_RESULTS", REPO / "results" / "standalone_acoustic"))

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

# ADFWI modules (Liu's package; T5-patched AcousticFWI accepts das_layer)
from ADFWI.model import AcousticModel
from ADFWI.survey import Source, Receiver, Survey, SeismicData
from ADFWI.propagator import AcousticPropagator, GradProcessor
from ADFWI.utils import wavelet
from ADFWI.utils.velocityDemo import (load_marmousi_model,
                                      resample_marmousi_model,
                                      get_smooth_marmousi_model)
from ADFWI.fwi import AcousticFWI
from ADFWI.fwi.misfit import (Misfit_waveform_L2, Misfit_envelope,
                              Misfit_global_correlation, Misfit_weighted_ECI)

# dasfwi modules (this repo): the DAS operator + hardened misfits
from das.geometry import FiberGeometry, merge_fibers
from das.das_layer import DASObservationLayer
from inversion.safe_misfits import (SinkhornSafe, SdtwSafe, TravelTimeSafe,
                                    make_nim,
                                    ConvolvedWavefieldMisfit)

# ============================================================================
# [1] PARAMETERS - everything a user should ever need to touch
# ============================================================================
# --- inversion choices (also overridable from the command line) -------------
MISFIT = "gc"          # l2 | envelope | gc | sdtw | sinkhorn | weci
OPTIMIZER = "adam"     # sgd | adagrad | adam | adamw | nadam  (Liu settings)
ITERATIONS = 300
USE_DAS = True         # True  = observe DAS strain rate on vertical fibers
                       # False = Liu's conventional surface pressure
                       #         receivers (the A/B control experiment)

# --- grid / time / source: Liu's acoustic Marmousi2 header ------------------
NZ, NX = 88, 200
DX = DZ = 40.0                     # [m]
NT, DT = 1600, 0.003               # 4.8 s records
F0 = 5.0                           # [Hz] Ricker, INTEGRATED (Liu's choice)
NABC, FREE_SURFACE = 30, True
X0_SECTION = 5000.0                # Marmousi section starts at x = 5 km
SRC_EVERY, SRC_Z = 5, 1            # a source every 5th x-node at z-index 1

# --- DAS acquisition (used when USE_DAS) ------------------------------------
GAUGE_L = 2 * DZ                   # 80 m gauge: endpoints exactly on nodes
FIBER_X_INDICES = (25, 75, 125, 175)   # 4 wells at x = 1, 3, 5, 7 km
FIBER_Z_TOP_INDEX = 12                 # first channel at 480 m (below water)
FIBER_N_CHANNELS = 74                  # channels on nodes 12..85 (~3.4 km)

# --- inversion machinery (Liu) ----------------------------------------------
GRAD_MASK_TOP = 10                 # zero gradients in the top 10 rows
SCHEDULER = dict(step_size=100, gamma=0.75)
CACHE_EVERY = 10                   # cache vp snapshot every N iterations

# ============================================================================
# [2] MODEL SOURCE - pick ONE block. Default: Marmousi2 benchmark.
# ============================================================================
def load_model_pair():
    """Return (vp_true, vp_init) as [NZ, NX] float arrays.

    CASE A - MARMOUSI2 BENCHMARK (active): Liu's standard section + his
    gaussian_kernel=6 smooth as the starting model. Needs the SEGY files
    in MARMOUSI_DIR (they download automatically on first use).
    """
    marmousi = load_marmousi_model(in_dir=str(MARMOUSI_DIR))
    x = np.linspace(X0_SECTION, X0_SECTION + DX * NX, NX)
    z = np.linspace(0, DZ * NZ, NZ)
    true_model = resample_marmousi_model(x, z, marmousi)
    smooth = get_smooth_marmousi_model(true_model, gaussian_kernel=6)
    return (np.asarray(true_model["vp"].T, np.float64),
            np.asarray(smooth["vp"].T, np.float64))

    # CASE B - SYNTHETIC (e.g. FORGE proxy): build/replace with any pair of
    # [NZ, NX] arrays; adjust the grid constants above to your model.
    #   from forge.proxy_model import forge_proxy_vp
    #   vp_true = forge_proxy_vp(NZ, NX, dz=DZ)
    #   vp_init = scipy.ndimage.gaussian_filter(vp_true, sigma=...)
    #   return vp_true, vp_init

    # CASE C - FIELD DATA: there is no vp_true; load your starting model and
    # return it twice (metrics vs "true" then mean nothing - ignore them),
    # and REPLACE the observed-data block in [4] with loading your processed
    # field strain-rate gathers (forge/io_preprocess.py chain) into
    # SeismicData via record_data({"strain_rate": gathers}).
    #   vp_init = np.load("my_starting_model.npy")
    #   return vp_init.copy(), vp_init


# ============================================================================
# helpers (nothing below needs editing for routine runs)
# ============================================================================
MISFITS = ("l2", "envelope", "gc", "sdtw", "sinkhorn", "weci",
           "traveltime", "nim", "convsi")

def build_misfit(name, iterations):
    """Liu's misfit constructions; hardened variants where upstream has
    portability bugs (see inversion/safe_misfits.py). traveltime and nim are
    the cycle-skipping-robust additions (traveltime = cross-correlation time
    shift; nim = normalized integration = Wasserstein-1 at p=1)."""
    if name == "l2":
        return Misfit_waveform_L2(dt=DT)
    if name == "envelope":
        return Misfit_envelope(dt=DT, p=1.5)
    if name == "gc":
        return Misfit_global_correlation(dt=DT)
    if name == "sdtw":
        return SdtwSafe(gamma=1, sparse_sampling=2, dt=DT)
    if name == "sinkhorn":
        return SinkhornSafe(dt=0.01, sparse_sampling=2, p=1, blur=1e-2)
    if name == "weci":
        return Misfit_weighted_ECI(p=1.5, dt=1, max_iter=iterations,
                                   instaneous_phase=False)
    if name == "traveltime":
        return TravelTimeSafe(dt=DT, beta=10)
    if name == "nim":
        return make_nim(p=1, trans_type="linear", theta=1.0, dt=DT)
    if name == "convsi":
        # source-independent convolved-wavefields misfit (Choi & Alkhalifah 2011); cancels the unknown source wavelet
        return ConvolvedWavefieldMisfit(dt=DT)
    raise ValueError(f"unknown misfit {name!r}")

# per-misfit run settings (Liu's batch/checkpoint choices; sinkhorn scales
# itself globally, so per-trace normalization is off for it)
RUN_SETTINGS = {
    "l2":         dict(batch_size=None, checkpoint_segments=1, normalize=True),
    "envelope":   dict(batch_size=None, checkpoint_segments=1, normalize=True),
    "gc":         dict(batch_size=None, checkpoint_segments=1, normalize=True),
    "sdtw":       dict(batch_size=5,    checkpoint_segments=2, normalize=True),
    "sinkhorn":   dict(batch_size=2,    checkpoint_segments=2, normalize=False),
    "weci":       dict(batch_size=None, checkpoint_segments=1, normalize=True),
    # traveltime normalizes internally and is O(shots*receivers) slow -> batch
    "traveltime": dict(batch_size=5,    checkpoint_segments=2, normalize=True),
    "nim":        dict(batch_size=None, checkpoint_segments=1, normalize=True),
    "convsi":     dict(batch_size=2,    checkpoint_segments=2, normalize=False),
}

OPTIMIZERS = {   # Liu's exact constructors (03-optimizer-test examples)
    "sgd":     lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9),
    "adagrad": lambda p: torch.optim.Adagrad(p, lr=10, lr_decay=0,
                                             weight_decay=0),
    "adam":    lambda p: torch.optim.Adam(p, lr=10),
    "adamw":   lambda p: torch.optim.AdamW(p, lr=10, betas=(0.9, 0.999),
                                           weight_decay=1e-6),
    "nadam":   lambda p: torch.optim.NAdam(p, lr=10, betas=(0.9, 0.999),
                                           weight_decay=0,
                                           momentum_decay=4e-3),
}

def pick_device(arg=None):
    if arg:
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def build_model(vp, vp_bound, vp_grad, device):
    rho = np.power(np.asarray(vp), 0.25) * 310.0     # Liu's density
    return AcousticModel(0, 0, NX, NZ, DX, DZ, vp, rho,
                         vp_bound=vp_bound, vp_grad=vp_grad,
                         free_surface=FREE_SURFACE, abc_type="PML",
                         abc_jerjan_alpha=0.007, nabc=NABC,
                         device=device, dtype=torch.float32)

def build_source():
    src_x = np.arange(2, NX - 1, SRC_EVERY)
    _, src_v = wavelet(NT, DT, F0, amp0=1)
    src_v = integrate.cumtrapz(src_v, axis=-1, initial=0)   # Liu integrates
    source = Source(nt=NT, dt=DT, f0=F0)
    for ix in src_x:
        source.add_source(src_x=int(ix), src_z=SRC_Z, src_wavelet=src_v,
                          src_type="mt",
                          src_mt=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    return source

def build_acquisition(device):
    """Returns (survey, das_layer, obs_key). DAS: receivers are the fibers'
    deduplicated gauge-endpoint nodes recording v_z; conventional: Liu's
    full surface line recording pressure."""
    source = build_source()
    rcv = Receiver(nt=NT, dt=DT)
    if USE_DAS:
        geometry = merge_fibers([
            FiberGeometry(x_well=ix * DX, z_top=FIBER_Z_TOP_INDEX * DZ,
                          n_channels=FIBER_N_CHANNELS, l=GAUGE_L,
                          dx=DX, dz=DZ, snap_to_nodes=False)
            for ix in FIBER_X_INDICES])
        rcv_z = np.array([kz for (kz, _kx) in geometry.rcv_pos])
        rcv_x = np.array([kx for (_kz, kx) in geometry.rcv_pos])
        rcv.add_receivers(rcv_x, rcv_z, "vz")
        layer = DASObservationLayer(geometry, output="strain_rate")
        layer = layer.to(torch.float32).to(device)
        return Survey(source, rcv), layer, "strain_rate"
    rcv.add_receivers(np.arange(0, NX), np.full(NX, 1), "pr")   # Liu's line
    return Survey(source, rcv), None, "p"


# ============================================================================
# main
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--misfit", default=MISFIT, choices=MISFITS)
    ap.add_argument("--optimizer", default=OPTIMIZER,
                    choices=sorted(OPTIMIZERS))
    ap.add_argument("--iterations", type=int, default=ITERATIONS)
    ap.add_argument("--conventional", action="store_true",
                    help="pressure receivers instead of DAS (A/B control)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="2-iteration wiring check")
    args = ap.parse_args()

    global USE_DAS
    if args.conventional:
        USE_DAS = False
    device = pick_device(args.device)
    iterations = 2 if args.smoke else args.iterations
    tag = (f"{'das' if USE_DAS else 'conv'}_{args.misfit}_{args.optimizer}"
           + ("_smoke" if args.smoke else ""))
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== acoustic {tag} on {device}, {iterations} iterations ===",
          flush=True)

    # [3] model + acquisition
    vp_true, vp_init = load_model_pair()
    survey, das_layer, obs_key = build_acquisition(device)

    # [4] observed data: inverse crime from the TRUE model (replace this
    # block with SeismicData loading for field data - see CASE C above)
    true_model = build_model(vp_true, vp_bound=None, vp_grad=False,
                             device=device)
    prop_true = AcousticPropagator(true_model, survey, device=device,
                                   dtype=torch.float32)
    with torch.no_grad():
        rec = prop_true.forward(checkpoint_segments=4)
        observed = (das_layer(rec["u"], rec["w"]) if USE_DAS
                    else rec["p"]).cpu()
    obs_data = SeismicData(survey)
    obs_data.record_data({obs_key: observed})
    print(f"observed {tuple(observed.shape)}, max|.| "
          f"{float(observed.abs().max()):.3e}", flush=True)

    # [5] inversion (Liu's machinery through the T5-patched AcousticFWI)
    model = build_model(vp_init,
                        vp_bound=[float(vp_true.min()), float(vp_true.max())],
                        vp_grad=True, device=device)
    prop = AcousticPropagator(model, survey, device=device,
                              dtype=torch.float32)
    optimizer = OPTIMIZERS[args.optimizer](model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **SCHEDULER)
    grad_mask = np.ones((NZ, NX))
    grad_mask[:GRAD_MASK_TOP, :] = 0
    settings = RUN_SETTINGS[args.misfit]

    fwi = AcousticFWI(propagator=prop, model=model,
                      optimizer=optimizer, scheduler=scheduler,
                      loss_fn=build_misfit(args.misfit, iterations),
                      obs_data=obs_data,
                      gradient_processor=GradProcessor(grad_mask=grad_mask),
                      waveform_normalize=settings["normalize"],
                      cache_result=True, cache_result_epoch=CACHE_EVERY,
                      save_fig_epoch=-1,
                      das_layer=das_layer, obs_key=obs_key)
    t0 = time.time()
    fwi.forward(iteration=iterations,
                batch_size=settings["batch_size"],
                checkpoint_segments=settings["checkpoint_segments"])
    hours = (time.time() - t0) / 3600.0

    # [6] outputs
    iter_loss = np.asarray(fwi.iter_loss)
    np.savez(out_dir / "iter_vp.npz", data=np.asarray(fwi.iter_vp))
    np.savez(out_dir / "iter_loss.npz", data=iter_loss)
    vp_final = model.vp.detach().cpu().numpy()
    d_true, d_inv = vp_true - vp_init, vp_final - vp_init
    denom = np.sqrt((d_true ** 2).sum() * (d_inv ** 2).sum())
    metrics = dict(
        tag=tag, device=device, iterations=iterations,
        runtime_h=round(hours, 3),
        rms_init=float(np.sqrt(((vp_init - vp_true) ** 2).mean())),
        rms_final=float(np.sqrt(((vp_final - vp_true) ** 2).mean())),
        update_corr=float((d_true * d_inv).sum() / denom) if denom else 0.0,
        losses_finite=bool(np.isfinite(iter_loss).all()))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    ext = [0, (NX - 1) * DX / 1000, (NZ - 1) * DZ / 1000, 0]
    for ax, (d, ttl) in zip(axes.flat[:3], [(vp_true, "true"),
                                            (vp_init, "initial"),
                                            (vp_final, f"inverted {tag}")]):
        im = ax.imshow(d, extent=ext, cmap="jet",
                       vmin=vp_true.min(), vmax=vp_true.max())
        ax.set(title=f"vp {ttl} [m/s]", xlabel="x [km]", ylabel="z [km]")
        fig.colorbar(im, ax=ax, shrink=0.8)
    axes.flat[3].plot(iter_loss, "k.-")
    axes.flat[3].set(title="loss", xlabel="iteration")
    fig.savefig(out_dir / "final.png", dpi=150)
    print("saved results to", out_dir, flush=True)


if __name__ == "__main__":
    main()
