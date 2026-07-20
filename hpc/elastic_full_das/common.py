"""Shared setup for the full-Marmousi2 ELASTIC (Vp/Vs) DAS campaign.

Scientific question: can DAS strain-rate FWI recover Vp and Vs on Liu's
iso-elastic Marmousi2 section, across the full misfit x optimizer grid, and does
illumination (diagonal-Hessian) preconditioning improve deep recovery?

Everything follows Liu's iso-elastic Marmousi2 example
(examples/elastic/Iso-elastic-Marmousi2-shotTop-recTop) except the receiver
side (vertical-fiber DAS strain rate through the E3 operator, no strain-to-
velocity conversion):

    grid          nz, nx = 78, 200 at dx = dz = 45 m  (x0 = 5000 m section)
    time          nt, dt = 2500, 0.003 s;  f0 = 3 Hz integrated Ricker
    source        integrated Ricker, mt type, every 5th x-node at z-index 2
    model         free_surface=True, PML, nabc=50, fd_order=4
    parameters    Vp + Vs inverted; DENSITY held CONSTANT at 2450 kg/m^3 and
                  shared by the observed + inversion models. (A 3-parameter run
                  that also inverted rho diverged in density and dragged Vp/Vs
                  down, so density is fixed here per finding #5.)
    init models   180 m x 180 m Gaussian smooth of the true Vp/Vs
                  (get_smooth_marmousi_model kernel=4 on the 45 m grid),
                  water rows pinned to truth
    constraints   vs <= vp / 1.5 Poisson clamp after every step; top 10 rows
                  (water) masked from updates
    inversion     300 iterations, StepLR(step_size=100, gamma=0.75)

DAS receiver side: 4 vertical fibers, 90 m gauge (l = 2*dz), strain rate via
DASObservationLayer. The illumination A/B (see run_one.py --precond) is on the
Vp/Vs gradients.

Paths (HPC-portable, override by environment):
    ADFWI_ROOT      default ../ADFWI next to the repo, else ADFWI_local.
    MARMOUSI_DIR    Marmousi2 SEGY dir. Default ../Data_downloads/marmousi2.
    DASFWI_RESULTS  output root. Default <repo>/results/elastic_full_das.
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
    "DASFWI_RESULTS", REPO / "results" / "elastic_full_das"))

for _p in (str(ADFWI_ROOT), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from scipy import integrate

from ADFWI.model import IsotropicElasticModel
from ADFWI.survey import Source, Receiver, Survey
from ADFWI.propagator import ElasticPropagator
from ADFWI.utils import wavelet
from ADFWI.utils.velocityDemo import (load_marmousi_model,
                                      resample_marmousi_model,
                                      get_smooth_marmousi_model)

from das.geometry import FiberGeometry, merge_fibers
from das.das_layer import DASObservationLayer
from inversion import config                          # techniques source of truth
from inversion.safe_misfits import apply_misfit       # NIM dispatch in the loop


# ---------------------------------------------------------------------------
# Liu iso-elastic Marmousi2 constants (+ our DAS receiver side)
# ---------------------------------------------------------------------------
NZ, NX = 78, 200
DX = DZ = 45.0
NT, DT = 2500, 0.003
F0 = 3.0
NABC, FREE_SURFACE = 50, True
X0_SECTION = 5000.0
SRC_EVERY, SRC_Z = 5, 2
FD_ORDER = 4
CHECKPOINT_SEGMENTS = 4
WATER_ROWS = 10
RHO_CONST = 2450.0                 # density held constant (not inverted)
SMOOTH_KERNEL = 4                  # 4 nodes * 45 m = 180 m Gaussian window
ITERATIONS = 300
SCHEDULER = dict(step_size=100, gamma=0.75)
CACHE_EVERY = 10

# update rules
MIN_VP_VS = 1.5                    # Poisson stability clamp (DO NOT REMOVE)
SGD_LR_SCALE = 0.01               # (unused here; norm_grad scales to par.max())

# DAS acquisition
GAUGE_L = 2 * DZ                   # 90 m gauge on the 45 m grid
FIBER_X_INDICES = (25, 75, 125, 175)
FIBER_Z_TOP_INDEX = 12             # 540 m: below the water layer
FIBER_N_CHANNELS = 62              # nodes 12..73 -> z = 540..3285 m

OBS_FILE = "obs_data_das.npz"

# techniques from the single source of truth (inversion/config.py)
OPTIMIZERS = config.LIU_OPTIMIZERS
MISFITS = config.MISFITS
MISFIT_RUN_SETTINGS = config.MISFIT_SETTINGS


def pick_device(arg=None):
    if arg:
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_misfit(name, iterations=ITERATIONS):
    return config.build_misfit(name, dt=DT, iterations=iterations)


def load_models():
    """True and 180 m-smoothed-initial vp, vs on Liu's elastic section (density
    is held constant at RHO_CONST, not inverted). Water rows of the initials
    pinned to truth."""
    marmousi = load_marmousi_model(in_dir=str(MARMOUSI_DIR))
    x = np.linspace(X0_SECTION, X0_SECTION + DX * NX, NX)
    z = np.linspace(0, DZ * NZ, NZ)
    true_model = resample_marmousi_model(x, z, marmousi)
    smooth = get_smooth_marmousi_model(true_model, gaussian_kernel=SMOOTH_KERNEL,
                                       mask_extra_detph=2, rcv_depth=8)
    vp_true = np.asarray(true_model["vp"].T, np.float64)
    vs_true = np.asarray(true_model["vs"].T, np.float64)
    vp_init = np.asarray(smooth["vp"].T, np.float64)
    vs_init = np.asarray(smooth["vs"].T, np.float64)
    vp_init[:WATER_ROWS] = vp_true[:WATER_ROWS]
    vs_init[:WATER_ROWS] = vs_true[:WATER_ROWS]
    return vp_true, vs_true, vp_init, vs_init


def build_model(vp, vs, bounds, grad, device, dtype=torch.float32):
    """grad=True makes vp, vs inversion parameters. Density is held CONSTANT at
    RHO_CONST (rho_grad=False, auto_update_rho=False) -- a joint density
    inversion diverged under this DAS acquisition. bounds = (vp_bound, vs_bound)."""
    vp_b, vs_b = bounds
    return IsotropicElasticModel(
        0, 0, NX, NZ, DX, DZ, vp=vp, vs=vs,
        rho=np.ones_like(vp) * RHO_CONST,
        vp_bound=vp_b, vs_bound=vs_b,
        vp_grad=grad, vs_grad=grad, rho_grad=False,
        free_surface=FREE_SURFACE, abc_type="PML",
        abc_jerjan_alpha=0.007, nabc=NABC, auto_update_rho=False,
        device=device, dtype=dtype)


def build_acquisition(device, dtype=torch.float32):
    """Liu's source line + our 4 vertical DAS fibers; returns (survey, layer)."""
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
                                output="strain_rate").to(dtype).to(device)
    return Survey(source, rcv), layer, geometry


def normalize_traces(d):
    """Per-trace max-abs normalization (AcousticFWI._normalize equivalent)."""
    mask = torch.sum(torch.abs(d), axis=1, keepdim=True) == 0
    mx = torch.max(torch.abs(d), axis=1, keepdim=True).values
    return d / mx.masked_fill(mask, 1)
