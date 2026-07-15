"""Single source of truth for DASFWI inversion techniques.

Every runner (standalone acoustic/elastic/field, the campaign, the matrix, the
ladder) imports its misfits, optimizers, regularizers and per-misfit run
settings from HERE, instead of redefining them. One place to add a technique;
one place to see the whole toolkit.

The four technique axes:
  MISFITS       - 9 objective functions (waveform / envelope / correlation /
                  optimal-transport / traveltime / hybrid / source-independent)
  OPTIMIZERS    - 5 (Liu's exact constructors) + configurable-lr variants
  REGULARIZERS  - Tikhonov / TV (1st & 2nd order) + none
  (source independence is the `convsi` misfit; starting-model construction is
   forge.traveltime_tomography; both compose with any of the above.)

InversionConfig captures a full technique stack; deployment_score ranks a
finished run. run_technique_matrix.py sweeps combinations and ranks them.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from ADFWI.fwi.misfit import (Misfit_waveform_L2, Misfit_envelope,
                              Misfit_global_correlation, Misfit_weighted_ECI)
from ADFWI.fwi.regularization import (regularization_Tikhonov_1order,
                                      regularization_Tikhonov_2order,
                                      regularization_TV_1order,
                                      regularization_TV_2order)
from inversion.safe_misfits import (SinkhornSafe, SdtwSafe, TravelTimeSafe,
                                    make_nim, ConvolvedWavefieldMisfit,
                                    GCMisfit64)

# ---------------------------------------------------------------------------
# 1. MISFITS
# ---------------------------------------------------------------------------
MISFITS = ("l2", "envelope", "gc", "sdtw", "sinkhorn", "weci",
           "traveltime", "nim", "convsi")

#: misfits that need FFT / pysdtw and so run on CUDA or CPU but NOT Apple MPS
CUDA_OR_CPU_ONLY = ("envelope", "weci", "sdtw", "convsi")

#: one-line description of each misfit's role (for reports / the matrix)
MISFIT_ROLE = {
    "l2":         "waveform L2 (baseline; most nonlinear / cycle-skips first)",
    "envelope":   "instantaneous-amplitude envelope (robust to phase errors)",
    "gc":         "global correlation (normalized; amplitude-insensitive)",
    "sdtw":       "soft-DTW (time-warp tolerant)",
    "sinkhorn":   "Wasserstein-Sinkhorn OT (convex in time shift; best far-field)",
    "weci":       "envelope->GC sigmoid hybrid (needs long runs)",
    "traveltime": "cross-correlation time shift (kinematic; amplitude-free)",
    "nim":        "normalized integration = Wasserstein-1 (cycle-skip robust)",
    "convsi":     "convolved-wavefields SOURCE-INDEPENDENT (unknown wavelet)",
}


def build_misfit(name, dt, iterations, use_gc64=False):
    """Construct a misfit by name. `dt` is the propagator time step; `iterations`
    is the planned count (WECI needs it). `use_gc64=True` gives the dtype-safe
    float64 global-correlation subclass (for float64 tests); otherwise stock GC.

    Liu-fixed misfit parameters are baked in (sinkhorn dt=0.01/p=1/blur=1e-2,
    weci dt=1/p=1.5). GC uses the propagator dt (a benign loss scaling)."""
    if name == "l2":
        return Misfit_waveform_L2(dt=dt)
    if name == "envelope":
        return Misfit_envelope(dt=dt, p=1.5)
    if name == "gc":
        return GCMisfit64(dt=dt) if use_gc64 else Misfit_global_correlation(dt=dt)
    if name == "sdtw":
        return SdtwSafe(gamma=1, sparse_sampling=2, dt=dt)
    if name == "sinkhorn":
        return SinkhornSafe(dt=0.01, sparse_sampling=2, p=1, blur=1e-2)
    if name == "weci":
        return Misfit_weighted_ECI(p=1.5, dt=1, max_iter=iterations,
                                   instaneous_phase=False)
    if name == "traveltime":
        return TravelTimeSafe(dt=dt, beta=10)
    if name == "nim":
        return make_nim(p=1, trans_type="linear", theta=1.0, dt=dt)
    if name == "convsi":
        return ConvolvedWavefieldMisfit(dt=dt)
    raise ValueError(f"unknown misfit {name!r}; choices {MISFITS}")


#: per-misfit run settings: shot mini-batch, checkpoint segments, and whether
#: AcousticFWI should per-trace max-normalize (OFF for the misfits that scale
#: themselves globally: sinkhorn and the source-independent convsi).
MISFIT_SETTINGS = {
    "l2":         dict(batch_size=None, checkpoint_segments=1, normalize=True),
    "envelope":   dict(batch_size=None, checkpoint_segments=1, normalize=True),
    "gc":         dict(batch_size=None, checkpoint_segments=1, normalize=True),
    "sdtw":       dict(batch_size=5,    checkpoint_segments=2, normalize=True),
    "sinkhorn":   dict(batch_size=2,    checkpoint_segments=2, normalize=False),
    "weci":       dict(batch_size=None, checkpoint_segments=1, normalize=True),
    "traveltime": dict(batch_size=5,    checkpoint_segments=2, normalize=True),
    "nim":        dict(batch_size=None, checkpoint_segments=1, normalize=True),
    "convsi":     dict(batch_size=2,    checkpoint_segments=2, normalize=False),
}

# ---------------------------------------------------------------------------
# 2. OPTIMIZERS
# ---------------------------------------------------------------------------
OPTIMIZER_NAMES = ("sgd", "adagrad", "adam", "adamw", "nadam")

#: Liu's exact optimizer constructors (03-optimizer-test examples); fixed lr.
#: Use these for the Liu-faithful campaign / standalone runs.
LIU_OPTIMIZERS = {
    "sgd":     lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9),
    "adagrad": lambda p: torch.optim.Adagrad(p, lr=10, lr_decay=0,
                                             weight_decay=0),
    "adam":    lambda p: torch.optim.Adam(p, lr=10),
    "adamw":   lambda p: torch.optim.AdamW(p, lr=10, betas=(0.9, 0.999),
                                           weight_decay=1e-6),
    "nadam":   lambda p: torch.optim.NAdam(p, lr=10, betas=(0.9, 0.999),
                                           weight_decay=0, momentum_decay=4e-3),
}


def build_optimizer(name, params, lr=None):
    """Optimizer by name. lr=None uses Liu's fixed lr; pass lr to override
    (our experiment scripts tune lr, e.g. sgd 0.004 for gradient-proportional
    updates). `sgd` keeps momentum 0.9; the Adam family keeps Liu's betas."""
    if lr is None:
        return LIU_OPTIMIZERS[name](params)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr, lr_decay=0, weight_decay=0)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999),
                                 weight_decay=1e-6)
    if name == "nadam":
        return torch.optim.NAdam(params, lr=lr, betas=(0.9, 0.999),
                                 weight_decay=0, momentum_decay=4e-3)
    raise ValueError(f"unknown optimizer {name!r}; choices {OPTIMIZER_NAMES}")

# ---------------------------------------------------------------------------
# 3. REGULARIZERS
# ---------------------------------------------------------------------------
REGULARIZERS = ("none", "tikhonov1", "tikhonov2", "tv1", "tv2")


def build_regularization(name, nx, nz, dx, dz, device, dtype):
    """Liu's regularization constructions (04-regularization examples),
    generalized to any grid. Returns (regularization_fn, weights_x, weights_z)
    for AcousticFWI (weights are [vp, rho]; 1e-7 on vp only, per Liu)."""
    if name in (None, "none"):
        return None, [0, 0], [0, 0]
    kw = dict(step_size=50, gamma=0.9, device=device, dtype=dtype)
    if name == "tikhonov1":
        fn = regularization_Tikhonov_1order(nx, nz, dx, dz, **kw)
    elif name == "tikhonov2":
        fn = regularization_Tikhonov_2order(nx, nz, dx, dz, **kw)
    elif name == "tv1":
        fn = regularization_TV_1order(nx, nz, dx, dz, 1e-7, 1e-7, **kw)
    elif name == "tv2":
        fn = regularization_TV_2order(nx, nz, dx, dz, **kw)
    else:
        raise ValueError(f"unknown regularization {name!r}; choices {REGULARIZERS}")
    return fn, [1e-7, 0], [1e-7, 0]

# ---------------------------------------------------------------------------
# 4. A FULL TECHNIQUE STACK + scoring
# ---------------------------------------------------------------------------
@dataclass
class InversionConfig:
    """A complete DASFWI technique stack. One object = one deployable recipe."""
    misfit: str = "gc"
    optimizer: str = "sgd"
    regularization: str = "none"
    starting_model: str = "smooth"        # smooth | gradient | traveltime | ...
    lr: Optional[float] = None            # None -> Liu's fixed lr
    bands: Sequence = field(default_factory=lambda: [(5.0, 15), (10.0, 15)])
    source_independent: bool = False      # convenience; True forces misfit=convsi

    def resolved_misfit(self):
        return "convsi" if self.source_independent else self.misfit

    def normalize(self):
        return MISFIT_SETTINGS[self.resolved_misfit()]["normalize"]

    def tag(self):
        t = f"{self.resolved_misfit()}_{self.optimizer}"
        if self.regularization != "none":
            t += f"_{self.regularization}"
        return t


def deployment_score(vp_true, vp_init, vp_final, iter_loss=None):
    """Rank a finished inversion for deployment. Returns a dict of metrics and a
    single scalar `score` (higher = better) combining error reduction and update
    fidelity. For field data (no vp_true) pass vp_true=None -> only loss metrics."""
    out = {}
    if iter_loss is not None and len(iter_loss):
        ll = np.asarray(iter_loss, float)
        out["loss_first"], out["loss_last"] = float(ll[0]), float(ll[-1])
        out["loss_reduction"] = float(1 - ll[-1] / ll[0]) if ll[0] != 0 else 0.0
        out["losses_finite"] = bool(np.isfinite(ll).all())
    if vp_true is None:
        out["score"] = out.get("loss_reduction", 0.0)
        return out
    vt, vi, vf = np.asarray(vp_true), np.asarray(vp_init), np.asarray(vp_final)
    rms_i = float(np.sqrt(((vi - vt) ** 2).mean()))
    rms_f = float(np.sqrt(((vf - vt) ** 2).mean()))
    dtru, dinv = vt - vi, vf - vi
    den = np.sqrt((dtru ** 2).sum() * (dinv ** 2).sum())
    out.update(rms_init=rms_i, rms_final=rms_f,
               recovery=float(1 - rms_f / rms_i) if rms_i > 0 else 0.0,
               update_corr=float((dtru * dinv).sum() / den) if den > 0 else 0.0)
    # deployment score: reward error reduction AND correct update direction
    out["score"] = round(0.5 * out["recovery"] + 0.5 * out["update_corr"], 4)
    return out
