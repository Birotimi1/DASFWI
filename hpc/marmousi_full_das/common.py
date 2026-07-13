"""Shared setup for the full-Marmousi2 DAS misfit x optimizer campaign.

The scientific question: can FWI recover the (full, Liu-standard) Marmousi2
section when the observable is vertical-fiber DAS STRAIN RATE through the E3
gauge operator, instead of Liu's surface pressure records?

Everything except the receiver side is VERBATIM Liu's ADFWI Marmousi2
examples (examples/acoustic/02-misfit-functions-test / 03-optimizer-test /
04-regularization-techniques-test, all sharing one header):

    grid          nz, nx = 88, 200 at dx = dz = 40 m  (x0 = 5000 m section)
    time          nt, dt = 1600, 0.003 s;  f0 = 5 Hz
    source        INTEGRATED Ricker (cumtrapz), mt type, one per 5th x-node
                  at z-index 1 (40 shots)
    model         free_surface=True, PML, nabc=30, abc_jerjan_alpha=0.007
    density       rho = 310 * vp^0.25 from the model's OWN vp, with
                  auto_update_rho left at its default True (rho re-derived
                  from the current vp each iteration - Liu's setup)
    init model    get_smooth_marmousi_model(true, gaussian_kernel=6)
    vp bounds     [vp_true.min(), vp_true.max()]
    inversion     300 iterations, StepLR(step_size=100, gamma=0.75),
                  grad mask zeroing the top 10 rows
    per-optimizer settings: see OPTIMIZERS (copied from Liu's files)
    per-misfit settings:    see build_misfit / MISFIT_RUN_SETTINGS

DAS receiver side (ours):
    N_FIBERS vertical fibers below the water layer, gauge length l = 2*dz
    (endpoints exactly on grid nodes - the E3 operator's exactness
    requirement; the 10 m FORGE gauge scales to 80 m on Liu's 40 m grid),
    strain rate = DASObservationLayer(u, w), obs_key="strain_rate" through
    the T5-patched AcousticFWI.

Paths (HPC-portable, override by environment):
    ADFWI_ROOT      ADFWI repo root. Default: ../ADFWI next to this repo if
                    present, else the repo's own ADFWI_local mirror.
    MARMOUSI_DIR    Marmousi2 SEGY dir (LOCAL ONLY, not in git). Default:
                    ../Data_downloads/marmousi2 next to this repo.
    DASFWI_RESULTS  output root. Default: <repo>/results/marmousi_full_das.
"""

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
    "DASFWI_RESULTS", REPO / "results" / "marmousi_full_das"))

for _p in (str(ADFWI_ROOT), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from scipy import integrate

from ADFWI.model import AcousticModel
from ADFWI.survey import Source, Receiver, Survey, SeismicData
from ADFWI.propagator import AcousticPropagator, GradProcessor
from ADFWI.utils import wavelet
from ADFWI.utils.velocityDemo import (load_marmousi_model,
                                      resample_marmousi_model,
                                      get_smooth_marmousi_model)
from ADFWI.fwi.misfit import (Misfit_waveform_L2, Misfit_envelope,
                              Misfit_global_correlation, Misfit_sdtw,
                              Misfit_weighted_ECI)
from ADFWI.fwi.regularization import (regularization_Tikhonov_1order,
                                      regularization_Tikhonov_2order,
                                      regularization_TV_1order,
                                      regularization_TV_2order)

from das.geometry import FiberGeometry, merge_fibers
from das.das_layer import DASObservationLayer
from inversion.safe_misfits import (SinkhornSafe, SdtwSafe, TravelTimeSafe,
                                    make_nim)


# ---------------------------------------------------------------------------
# Liu-verbatim constants
# ---------------------------------------------------------------------------
OX, OZ = 0, 0
NZ, NX = 88, 200
DX = DZ = 40.0
NT, DT = 1600, 0.003
NABC = 30
F0 = 5.0
FREE_SURFACE = True
X0_SECTION = 5000.0            # Liu: x = linspace(5000, 5000 + dx*nx, nx)
ITERATIONS = 300
GRAD_MASK_TOP = 10             # rows (~ the water layer)

# DAS acquisition (ours)
GAUGE_L = 2 * DZ               # 80 m: endpoints exactly one node from center
N_FIBERS = 4
FIBER_X_INDICES = (25, 75, 125, 175)   # x = 1, 3, 5, 7 km within the section
FIBER_Z_TOP_INDEX = 12                 # 480 m: below the ~450 m water layer
FIBER_N_CHANNELS = 74                  # nodes 12..85 -> z = 480..3400 m

OBS_FILE = "obs_data_das.npz"

# Liu's optimizer configurations, copied from
# examples/acoustic/03-optimizer-test/01-Marmousi2-Test/02_inversion_*.py
OPTIMIZERS = {
    "sgd":     lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9),
    "adagrad": lambda params: torch.optim.Adagrad(params, lr=10, lr_decay=0,
                                                  weight_decay=0),
    "adam":    lambda params: torch.optim.Adam(params, lr=10),
    "adamw":   lambda params: torch.optim.AdamW(params, lr=10,
                                                betas=(0.9, 0.999),
                                                weight_decay=1e-6),
    "nadam":   lambda params: torch.optim.NAdam(params, lr=10,
                                                betas=(0.9, 0.999),
                                                weight_decay=0,
                                                momentum_decay=4e-3),
}

MISFITS = ("l2", "envelope", "gc", "sdtw", "sinkhorn", "weci",
           "traveltime", "nim")

# per-misfit run settings (batch_size, checkpoint_segments,
# waveform_normalize), copied from Liu's misfit-test scripts; sinkhorn runs
# unnormalized because SinkhornSafe scales globally per shot (per-trace
# max-normalization backward overflows float32 on numerically-dead fiber
# traces - see SinkhornSafe)
MISFIT_RUN_SETTINGS = {
    "l2":         dict(batch_size=None, checkpoint_segments=1, normalize=True),
    "envelope":   dict(batch_size=None, checkpoint_segments=1, normalize=True),
    "gc":         dict(batch_size=None, checkpoint_segments=1, normalize=True),
    "sdtw":       dict(batch_size=5,    checkpoint_segments=2, normalize=True),
    "sinkhorn":   dict(batch_size=2,    checkpoint_segments=2, normalize=False),
    "weci":       dict(batch_size=None, checkpoint_segments=1, normalize=True),
    # traveltime is O(shots*receivers) conv1d -> batch to cap memory (still slow)
    "traveltime": dict(batch_size=5,    checkpoint_segments=2, normalize=True),
    "nim":        dict(batch_size=None, checkpoint_segments=1, normalize=True),
}


def pick_device(arg=None):
    if arg:
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_misfit(name, iterations=ITERATIONS):
    """Liu's misfit constructions, verbatim from his example files."""
    if name == "l2":
        return Misfit_waveform_L2(dt=DT)
    if name == "envelope":
        return Misfit_envelope(dt=DT, p=1.5)
    if name == "gc":
        return Misfit_global_correlation(dt=DT)
    if name == "sdtw":
        # Liu: Misfit_sdtw(gamma=1, sparse_sampling=2, dt=dt); SdtwSafe =
        # same math with a portable use_cuda test (see the class)
        return SdtwSafe(gamma=1, sparse_sampling=2, dt=DT)
    if name == "sinkhorn":
        # Liu: Misfit_wasserstein_sinkhorn(dt=0.01, sparse_sampling=2, p=1,
        # blur=1e-2); SinkhornSafe = same math, portable dtypes + fiber
        # numerics guards
        return SinkhornSafe(dt=0.01, sparse_sampling=2, p=1, blur=1e-2)
    if name == "weci":
        # counter == iteration ONLY because batch_size=None for weci
        return Misfit_weighted_ECI(p=1.5, dt=1, max_iter=iterations,
                                   instaneous_phase=False)
    if name == "traveltime":
        # cross-correlation traveltime (cycle-skipping robust); slow
        return TravelTimeSafe(dt=DT, beta=10)
    if name == "nim":
        # normalized integration = Wasserstein-1 at p=1 (cycle-skipping robust)
        return make_nim(p=1, trans_type="linear", theta=1.0, dt=DT)
    raise ValueError(f"unknown misfit {name!r}")


def build_regularization(name, device, dtype):
    """Liu's regularization constructions (04-regularization examples).
    Returns (regularization_fn, weights_x, weights_z)."""
    if name in (None, "none"):
        return None, [0, 0], [0, 0]
    kw = dict(step_size=50, gamma=0.9, device=device, dtype=dtype)
    if name == "tikhonov1":
        fn = regularization_Tikhonov_1order(NX, NZ, DX, DZ, **kw)
    elif name == "tikhonov2":
        fn = regularization_Tikhonov_2order(NX, NZ, DX, DZ, **kw)
    elif name == "tv1":
        fn = regularization_TV_1order(NX, NZ, DX, DZ, 1e-7, 1e-7, **kw)
    elif name == "tv2":
        fn = regularization_TV_2order(NX, NZ, DX, DZ, **kw)
    else:
        raise ValueError(f"unknown regularization {name!r}")
    return fn, [1e-7, 0], [1e-7, 0]


def load_models():
    """True and smooth-initial vp on Liu's standard section."""
    marmousi = load_marmousi_model(in_dir=str(MARMOUSI_DIR))
    x = np.linspace(X0_SECTION, X0_SECTION + DX * NX, NX)
    z = np.linspace(0, DZ * NZ, NZ)
    true_model = resample_marmousi_model(x, z, marmousi)
    smooth_model = get_smooth_marmousi_model(true_model, gaussian_kernel=6)
    vp_true = true_model["vp"].T.copy()
    vp_init = smooth_model["vp"].T.copy()
    return np.asarray(vp_true, np.float64), np.asarray(vp_init, np.float64)


def build_model(vp, vp_bound, vp_grad, device, dtype=torch.float32):
    rho = np.power(np.asarray(vp), 0.25) * 310.0   # Liu's density
    return AcousticModel(OX, OZ, NX, NZ, DX, DZ, vp, rho,
                         vp_bound=vp_bound, vp_grad=vp_grad,
                         free_surface=FREE_SURFACE,
                         abc_type="PML", abc_jerjan_alpha=0.007,
                         nabc=NABC, device=device, dtype=dtype)


def build_geometry():
    fibers = [FiberGeometry(x_well=ix * DX,
                            z_top=FIBER_Z_TOP_INDEX * DZ,
                            n_channels=FIBER_N_CHANNELS,
                            l=GAUGE_L, dx=DX, dz=DZ, snap_to_nodes=False)
              for ix in FIBER_X_INDICES]
    return merge_fibers(fibers)


def build_source():
    """Liu's source line: integrated Ricker, every 5th x-node at z=1."""
    src_x = np.arange(2, NX - 1, 5)
    _, src_v = wavelet(NT, DT, F0, amp0=1)
    src_v = integrate.cumtrapz(src_v, axis=-1, initial=0)   # Liu integrates
    source = Source(nt=NT, dt=DT, f0=F0)
    for ix in src_x:
        source.add_source(src_x=int(ix), src_z=1, src_wavelet=src_v,
                          src_type="mt",
                          src_mt=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    return source


def build_survey(geometry):
    source = build_source()
    rcv = Receiver(nt=NT, dt=DT)
    rcv_z = np.array([kz for (kz, _kx) in geometry.rcv_pos])
    rcv_x = np.array([kx for (_kz, kx) in geometry.rcv_pos])
    rcv.add_receivers(rcv_x, rcv_z, "vz")     # x FIRST (survey/receiver.py)
    return Survey(source, rcv)


def build_gradient_processor():
    grad_mask = np.ones((NZ, NX))
    grad_mask[:GRAD_MASK_TOP, :] = 0
    return GradProcessor(grad_mask=grad_mask)
