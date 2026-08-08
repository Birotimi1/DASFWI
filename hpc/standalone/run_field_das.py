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
from inversion.adaptive_misfit import BlendedMisfit, SkipSwitch
from inversion.skip_diagnostic import skip_fraction
from inversion import near_surface as ns
from inversion.das_conditioning import ConditionedMisfit
from forge.field_loader import load_forge_field, summarize

# ============================================================================
# [1] PARAMETERS
# ============================================================================
WELL = "78A-32"        # 78A-32 (1010 ch) | 78B-32 (1206 ch)
# >>> THE ILLUMINATION DEFICIT -- probably why our sections look nothing like
# >>> Park's, and it is not a tuning detail. <<<
# 318 shots exist. We were using 20 (6%). Park used 192 gathers, both wells.
#
#   ours,  20 shots x 103 ch =  2,060 traces vs 40,640 model cells -> 0.05 /cell
#   Park, 192 shots x 220 ch = 42,240 traces                       -> 1.04 /cell
#
# A 2-D FWI at 0.05 traces per cell is underdetermined by ~20:1. Every
# source-receiver pair carves its own sensitivity streak, and with too few
# shots those streaks never overlap enough to average out -- which is exactly
# the vertical fingering in our models. No amount of gradient smoothing fixes
# a data deficit, and the smoothing A/B agreed: it moved roughness by 3%.
#
# AND IT IS FREE. Cost ~ shots x iterations, so 100 shots x 30 iterations costs
# the same 3000 shot-gradients as 20 x 150 -- and the FORGE synthetic already
# measured 30 iterations as BETTER than 150 (shallow error grows monotonically),
# which is also what Park used. We were spending our budget on iterations that
# made the model worse instead of on data.
N_SHOTS = 100          # walkaway shots to load (318 available)
#: shots per gradient step. A full 100-shot wavefield is ~40 GiB; batching keeps
#: it near 8 GiB and makes the gradient stochastic, which adam/nadam handle and
#: line-search methods do not (lbfgs/nlcg are refused by this driver anyway).
SHOT_BATCH = 20
MISFIT = "gc"          # l2 | envelope | gc | sdtw | sinkhorn | weci
OPTIMIZER = "adam"     # sgd | adagrad | adam | adamw | nadam
ITERATIONS = 200

# --- grid / time (COARSE default = fast wiring; see the module docstring) ----
# >>> PARK-COMPARABLE, and affordable only because the geometry was fixed. <<<
# Park et al. (TLE 44(4)): "the spatial grid interval is 10 m, and the time
# sampling is 0.001 s over a total recording time of 2 s ... 30 iterations ...
# a Ricker wavelet with a peak frequency of 10 Hz and a maximum frequency of
# 20 Hz". Matching them makes our result a like-for-like comparison.
#
# 10 m was previously unaffordable because the receivers were mis-mapped to
# 2492-3522 m depth, forcing nz=192 (3840 m). At their true depths (0-1013 m in
# 78A-32, 0-1209 m in 78B-32) the model is nz=127, so the FINER grid costs less
# than the old coarse one did.
#
# dt: Park quote 1 ms, but our CFL limit at dz=10, vmax=6000 is 0.75 ms
# (safety 0.45; the 17:1 air/rock contrast plus PML puts us below the nominal
# 0.606). Their scheme is not ours, so we take OUR stability limit -- Park's
# 1 ms blew up here, and only at the full 2 s record, which is why an early
# short test wrongly cleared it.
DZ = DX = 10.0
NT_MODEL, DT_MODEL = 2667, 7.5e-4  # 2.0 s at the CFL limit for dz=10, 6 km/s
F0 = 10.0                          # Park's Ricker peak frequency
NABC = 30
F_ARRIVAL_PAD = 15                 # grid padding nodes

# --- starting model: 1-D vp gradient (no true model for field data) ----------
VP_TOP, VP_BOTTOM = 2000.0, 5500.0     # linear surface->deep [m/s]
#: Park INV2 uses 1.0-5.9 km/s. Our old (1500, 6000) forbade the slow alluvium
#: of zone I outright AND made an air layer impossible (it clamps 340 -> 1500).
VP_BOUND = ns.VP_BOUND_FIELD
#: arms, mirroring hpc/marmousi_full_das/run_switch.py so results are comparable
ARMS = ("switch", "fixedk", "l2", "gc", "convsi", "tfphase", "envelope")
SOLO_ARMS = ("l2", "gc", "convsi", "tfphase")      # lambda pinned to 0
REFINERS = ("l2", "gc", "convsi", "tfphase")
ROBUSTS = ("envelope", "tfphase")

# --- inversion machinery -----------------------------------------------------
GRAD_MASK_TOP = 8
SCHEDULER = dict(step_size=100, gamma=0.75)
CACHE_EVERY = 10

# techniques from the single source of truth (inversion/config.py)
MISFITS = config.MISFITS
RUN_SETTINGS = config.MISFIT_SETTINGS
OPTIMIZERS = config.LIU_OPTIMIZERS
WELLS = ("78A-32", "78B-32")


def parse_bands(spec):
    """'5,8,full' -> [5.0, 8.0, None]  (None = unfiltered)."""
    out = [None if t.strip().lower() in ("full", "none", "0") else float(t)
           for t in str(spec).split(",") if t.strip()]
    if not out:
        raise ValueError(f"no bands parsed from {spec!r}")
    return out


def allocate_iters(iters, n_bands, mode="final-heavy"):
    """Iterations PER BAND. 'equal' gives the TOP band only 1/n of the budget
    while a single-scale control gets all of it -- and the score is decided at
    the top band. That is how multiscale was measured as harmful on Marmousi."""
    if mode == "equal" or n_bands == 1:
        return [iters] * n_bands
    total = iters * n_bands
    final = total // 2
    rest, rem = divmod(total - final, n_bands - 1)
    out = [rest] * (n_bands - 1) + [final]
    out[-2] += rem
    return out


def build_misfit(name, iterations, dt, f_eff=None):
    if name == "tfphase" and f_eff:
        return config.build_misfit(name, dt=dt, iterations=iterations,
                                   f_min=0.4 * f_eff, f_max=f_eff)
    return config.build_misfit(name, dt=dt, iterations=iterations)


def pick_device(arg=None):
    # Delegates so the SLURM guard lives in ONE place: holding an H100 and
    # falling through to CPU bills the full walltime for a run that will not
    # finish. See inversion/device.py.
    from inversion.device import pick_device as _pd
    return _pd(arg)


def _raw_loss_fields(loss_fn):
    """First/last RAW misfit PER TERM, so first-vs-last is like-for-like.

    Comparing across terms measures their scale difference, not progress: the
    switch arms reported the SAME drop% at 30 and at 150 iterations because
    raw[0] was an envelope value and raw[-1] a convsi one.
    """
    out = {}
    for key, attr in (("lo", "raw_lo"), ("hi", "raw_hi")):
        hist = getattr(loss_fn, attr, None)        # ConditionedMisfit delegates
        good = [v for v in (hist or []) if np.isfinite(v)]
        if good:
            out[f"raw_{key}_first"] = float(good[0])
            out[f"raw_{key}_last"] = float(good[-1])
            out[f"raw_{key}_n"] = len(good)
    if not out:
        return {}
    # score on the term that ran the LONGEST -- for a switch that handed over,
    # that is the refiner it spent most of the run in.
    best = max(("lo", "hi"), key=lambda k: out.get(f"raw_{k}_n", 0))
    out.update(raw_loss_first=out[f"raw_{best}_first"],
               raw_loss_last=out[f"raw_{best}_last"],
               raw_loss_term=best, raw_loss_finite=True)
    return out


def gradient_start_model(nz, nx, dz):
    """1-D linear vp gradient VP_TOP -> VP_BOTTOM broadcast over x."""
    zcol = np.linspace(VP_TOP, VP_BOTTOM, nz)
    return np.repeat(zcol[:, None], nx, axis=1).astype(np.float64)


# ============================================================================
# main
# ============================================================================
def _route_b_starter(bundle, g, nz, nx, device, iters, optimizer_name):
    """Short gc inversion from a 1-D gradient -> a long-wavelength starting model.

    `gc` (global correlation) is used because it WON our Marmousi starter matrix
    (skip 0.544 -> 0.439) and because it is amplitude-insensitive, which matters
    on field data. The result is heavily smoothed by the caller: a starter that
    is not long-wavelength is not a starter.
    """
    vp0 = gradient_start_model(nz, nx, g["dz"])
    rho0 = np.power(vp0, 0.25) * 310.0
    m0 = AcousticModel(0, 0, nx, nz, g["dx"], g["dz"], vp0, rho0,
                       vp_bound=list(VP_BOUND), vp_grad=True, free_surface=True,
                       abc_type="PML", abc_jerjan_alpha=0.007, nabc=g["nabc"],
                       device=device, dtype=torch.float32)
    p0 = AcousticPropagator(m0, bundle["survey"], device=device,
                            dtype=torch.float32)
    opt0 = OPTIMIZERS[optimizer_name](m0.parameters())
    # AcousticFWI.forward calls scheduler.step() UNCONDITIONALLY, so
    # scheduler=None raises AttributeError on the very first iteration. Caught
    # by the end-to-end integration test, not by any unit test -- the starter
    # would have crashed the moment it ran on the cluster.
    sch0 = torch.optim.lr_scheduler.StepLR(opt0, step_size=10 ** 9, gamma=1.0)
    st = RUN_SETTINGS["gc"]
    f0 = AcousticFWI(propagator=p0, model=m0, optimizer=opt0, scheduler=sch0,
                     loss_fn=build_misfit("gc", iters, g["dt"]),
                     obs_data=bundle["obs_data"],
                     gradient_processor=GradProcessor(),
                     waveform_normalize=st["normalize"], cache_result=False,
                     save_fig_epoch=-1, das_layer=bundle["das_layer"],
                     obs_key="strain_rate")
    f0.forward(iteration=iters, batch_size=st.get("batch_size") or SHOT_BATCH,
               checkpoint_segments=st["checkpoint_segments"])
    return m0.vp.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # NOT a fixed list: a hardcoded choices=() makes the driver unusable at any
    # other site. The loader discovers wells from the data directory, so accept
    # any name and let it fail loudly if the directory is absent.
    ap.add_argument("--well", default=WELL,
                    help="well subdirectory under $FORGE_DAS_DIR (any site)")
    ap.add_argument("--shots", type=int, default=N_SHOTS)
    ap.add_argument("--misfit", default=MISFIT, choices=MISFITS)
    ap.add_argument("--optimizer", default=OPTIMIZER, choices=sorted(OPTIMIZERS))
    ap.add_argument("--iterations", type=int, default=ITERATIONS)
    ap.add_argument("--batch-shots", type=int, default=SHOT_BATCH,
                    dest="batch_shots",
                    help="shots per gradient step; caps GPU memory")
    ap.add_argument("--dz", type=float, default=DZ)
    ap.add_argument("--dt", type=float, default=DT_MODEL)
    ap.add_argument("--nt", type=int, default=NT_MODEL)
    ap.add_argument("--f0", type=float, default=F0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--lag-check", action="store_true", dest="lag_check",
                    help="run ONE forward from the starting model, report the "
                         "obs-vs-syn arrival lag distribution, and exit. "
                         "Answers WHY skip saturates at 1.000, which the "
                         "fraction itself cannot.")
    ap.add_argument("--starting", default="gradient",
                    choices=("gradient", "traveltime", "route_b"),
                    help="starting model: blind 1-D gradient; first-break "
                         "traveltime tomography (Park's approach, needs picks); "
                         "or route_b = a SHORT gc inversion then heavy smoothing "
                         "-- wave-equation, NO PICKING, which is the whole "
                         "transferability claim")
    ap.add_argument("--starter-iters", type=int, default=50, dest="starter_iters",
                    help="route_b: iterations of the gc pre-inversion")
    ap.add_argument("--starter-smooth", type=float, default=6.0,
                    dest="starter_smooth",
                    help="route_b: post-hoc Gaussian sigma (nodes). A STARTER "
                         "must stay long-wavelength or it is not a starter.")
    # ---- the adaptive switch (was entirely absent from this driver) --------
    ap.add_argument("--arm", default=None, choices=ARMS,
                    help="switch = envelope->refiner timed by MEASURED skip; "
                         "a solo misfit name runs that misfit alone. Default: "
                         "the --misfit value, i.e. the old behaviour.")
    ap.add_argument("--refiner", default="gc", choices=REFINERS,
                    help="resolution term (lambda=0). NOTE l2 fits amplitude AND "
                         "phase, so at FORGE it inherits the assumed-wavelet "
                         "error; convsi is source-independent.")
    ap.add_argument("--robust", default="envelope", choices=ROBUSTS,
                    help="cycle-skip-tolerant term (lambda=1)")
    ap.add_argument("--chunk", type=int, default=25,
                    help="iterations per controller update + checkpoint")
    ap.add_argument("--fixed-k", type=int, default=100, dest="fixed_k",
                    help="fixedk arm: iterations before handing over")
    # ---- multiscale (also absent) -----------------------------------------
    ap.add_argument("--bands", default=None,
                    help="comma-separated low-pass cut-offs, 'full' for "
                         "unfiltered, e.g. 5,8,12,full. FORGE spans ~2.7 "
                         "octaves so a cascade is real here, unlike Marmousi")
    ap.add_argument("--iter-alloc", default="final-heavy", dest="iter_alloc",
                    choices=("equal", "final-heavy"),
                    help="'equal' starves the top band, which is what made "
                         "multiscale look harmful on Marmousi")
    # ---- near surface ------------------------------------------------------
    ap.add_argument("--z-air", type=float, default=0.0, dest="z_air",
                    help="air-layer thickness (m). MEASURE THE RELIEF FIRST: "
                         "if it is < lambda/4 the flat datum is fine and this "
                         "only costs grid. 0 = no air layer.")
    ap.add_argument("--window", action="store_true",
                    help="ARRIVAL WINDOWING. Park report STRONG SURFACE WAVES "
                         "in the near-offset FORGE gathers, which an ACOUSTIC "
                         "code cannot model and therefore explains by inventing "
                         "near-surface velocity. MEASURED on the FORGE "
                         "synthetic: windowing improves the shallow model for "
                         "every refiner (convsi -54, gc -26, l2 -1 m/s). It "
                         "HURT on Marmousi only because noiseless inverse-crime "
                         "data has no surface waves to remove.")
    ap.add_argument("--window-pre", type=float, default=0.15, dest="window_pre")
    ap.add_argument("--window-post", type=float, default=0.5, dest="window_post",
                    help="s after the peak. Surface waves are SLOW (~0.9*Vs) so "
                         "they arrive well after the P break; a short "
                         "post-window is what removes them.")
    ap.add_argument("--channel-weight", action="store_true", dest="channel_weight",
                    help="REFUSED with a nonlinear robust term -- it weights the "
                         "DATA not each channel's CONTRIBUTION, and through "
                         "envelope^1.5 suppresses a weak channel 220x harder "
                         "than asked (collapsed the switch 0.742->0.26).")
    ap.add_argument("--topo-air", action="store_true", dest="topo_air",
                    help="air layer FOLLOWING the measured topography (the "
                         "FORGE surface is a 162 m ramp, so a uniform slab is "
                         "wrong). Overrides --z-air.")
    ap.add_argument("--grad-smooth", default="none", dest="grad_smooth",
                    choices=("none", "wavelength"),
                    help="wavelength = ANISOTROPIC lambda/4 smoothing, 4:1 H:V "
                         "(Park use 2:1 and 4:1); isotropic discards the "
                         "vertical resolution a VSP exists to provide")
    ap.add_argument("--smoke", action="store_true", help="2-iteration check")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="validate the configuration and EXIT before loading "
                         "any data. Catches argparse errors, refused "
                         "optimizer/conditioning combinations and tag "
                         "collisions without a GPU or the SEG-Y -- this driver "
                         "had no dry-run at all, so a 13-cell campaign could "
                         "not be checked before submission.")
    ap.add_argument("--qc", default="on", choices=("on", "off", "strict"),
                    help="DAS waveform-shape QC before inverting. 'strict' "
                         "ABORTS if coupling is acting as a per-channel filter, "
                         "because no misfit choice fixes that (see "
                         "inversion/das_qc.py)")
    args = ap.parse_args()

    if args.optimizer in ("lbfgs", "nlcg"):
        raise SystemExit(
            f"*** {args.optimizer} is a SETTLED NEGATIVE: both line-search "
            "optimizers DIVERGED to NaN on every Route B cell (4/4, ~54 SU "
            "burned). They need an accurate directional derivative, but with "
            "shot batching the gradient is STOCHASTIC, so the Wolfe/Armijo "
            "conditions are evaluated on noise. Adam/SGD do not line-search "
            "and are unaffected. Use adam, adamw, nadam or sgd.")
    device = pick_device(args.device)
    iterations = 2 if args.smoke else args.iterations
    arm = args.arm or args.misfit
    if arm in SOLO_ARMS:
        args.refiner = arm                     # solo arm IS its own refiner
    bands = parse_bands(args.bands) if args.bands else [None]
    iters_by_band = allocate_iters(max(1, iterations // len(bands)),
                                   len(bands), args.iter_alloc)
    # EVERY knob that changes the experiment goes in the tag. This class of bug
    # (two configurations writing to one directory) has appeared five times.
    pair = "" if args.refiner in ("l2", arm) else f"-{args.refiner}"
    rb = "" if args.robust == "envelope" or arm in SOLO_ARMS else f"+{args.robust}"
    # ITERATIONS IN THE TAG. Without it the 30- and 150-iteration cells render
    # the same directory and the second silently overwrites the first --
    # destroying precisely the early-vs-late comparison the campaign exists to
    # make, since the synthetic showed shallow error grows with iterations.
    # Seventh occurrence of this bug class; caught by the campaign's own
    # tag-uniqueness check rather than by review.
    tag = ("field_" + args.well + "_" + arm + pair + rb + "_" + args.optimizer
           + f"_i{iterations}"
           + "_" + args.starting
           + ("_b" + args.bands.replace(",", "-") if args.bands else "")
           + ("_fh" if args.bands and args.iter_alloc == "final-heavy" else "")
           + ("_air" if args.z_air > 0 else "")
           + ("_w" if args.window else "")
           + ("_c" if args.channel_weight else "")
           + ("_g" if args.grad_smooth != "none" else "")
           + ("_smoke" if args.smoke else ""))
    out_dir = OUT_ROOT / tag
    print(f"=== FORGE field {tag} on {device}, {iterations} iterations ===",
          flush=True)

    if args.dry_run:
        # A DRY RUN MUST NOT TOUCH THE FILESYSTEM. mkdir used to sit above this
        # return, so validating the campaign created one empty directory per
        # cell -- 53 of them here, and a `--dry-run` is run far more often than
        # a real job. They also make `ls results/` look like completed work and
        # break any "does this cell have results yet" check. PSC had just warned
        # about inode exhaustion when I noticed.
        print(f"    dry-run OK: {tag}", flush=True)
        print(f"      arm={arm} refiner={args.refiner} robust={args.robust} "
              f"start={args.starting} opt={args.optimizer} iters={iterations} "
              f"bands={bands} window={args.window} topo_air={args.topo_air}",
              flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # [3] load real field data (strain rate) + geometry
    bundle = load_forge_field(well=args.well, n_shots=args.shots,
                              dz=args.dz, dx=args.dz, nt_model=args.nt,
                              dt_model=args.dt, f0=args.f0, nabc=NABC,
                              pad_nodes=F_ARRIVAL_PAD, device=device)
    print(summarize(bundle), flush=True)
    g = bundle["grid"]
    nz, nx = g["nz"], g["nx"]

    # [3b] DAS QC -- AMPLITUDE distortion or SHAPE distortion? This runs on
    # EVERY field site, not just FORGE, because the answer changes the strategy:
    # amplitude-only is already handled by normalisation + gc, whereas a
    # per-channel FILTER has to be deconvolved out and no misfit choice helps.
    # It needs only the observed gathers, so it transfers anywhere.
    if args.qc != "off":
        from inversion.das_qc import qc_das, format_report, recommended_settings
        obs_qc = np.asarray(bundle["obs_data"].data["strain_rate"])   # [S,nt,C]
        qc = qc_das(obs_qc, g["dt"], channel_spacing=g["dz"],
                    v_min=VP_BOUND[0])
        print(format_report(qc), flush=True)
        (out_dir / "das_qc.json").write_text(json.dumps(
            {"well": args.well, **qc}, indent=2))
        if qc["shape_distortion"]:
            msg = ("DAS QC: coupling is acting as a per-channel FILTER "
                   "(shape distortion). Per-trace normalisation and phase "
                   "misfits do NOT correct this -- deconvolve the channel "
                   "transfer functions or model coupling (Celli et al. 2024).")
            if args.qc == "strict":
                raise SystemExit("ABORT -- " + msg)
            print("!!! WARNING -- " + msg, flush=True)
        elif qc["inconclusive"]:
            print("!!! DAS QC inconclusive -- see das_qc.json; proceeding with "
                  "the conservative defaults.", flush=True)
        print(f"    QC recommends: {recommended_settings(qc)['note']}",
              flush=True)

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
    elif args.starting == "route_b":
        # ROUTE B: a short gc inversion from the blind 1-D gradient, then heavy
        # smoothing. Wave-equation, NO PICKING -- Park manually pick first
        # arrivals on 100 CSGs, and not needing that is the transferability
        # claim. Kept INSIDE this driver rather than as a separate script so it
        # inherits the loader, the QC, the near-surface setup and the tag rules.
        from scipy.ndimage import gaussian_filter
        vp_init = _route_b_starter(bundle, g, nz, nx, device,
                                   args.starter_iters, args.optimizer)
        vp_init = gaussian_filter(vp_init, sigma=args.starter_smooth)
        vp_init = np.clip(vp_init, *VP_BOUND)
        print(f"    route_b starter: {args.starter_iters} gc iterations, "
              f"sigma={args.starter_smooth} nodes, "
              f"vp {vp_init.min():.0f}-{vp_init.max():.0f} m/s", flush=True)
    else:
        vp_init = gradient_start_model(nz, nx, g["dz"])

    # ---- near surface: air layer + bounds + anisotropic smoothing ----------
    # AIR LAYER FOLLOWING THE TOPOGRAPHY, not a uniform slab. MEASURED at FORGE:
    # the surface is a ramp dropping 161.6 m over ~2960 m (corr(x,z)=+0.994), so
    # at one end the ground IS the datum and at the other it is 162 m below. A
    # flat datum fabricates a free-surface ghost 2h/v = 215 ms late = 8.6
    # half-cycles at 20 Hz -- an invented arrival, far past cycle skipping.
    src_z_m = np.asarray(bundle["src_z_grid"], float) * g["dz"]
    if args.topo_air:
        ground = ns.surface_profile(np.asarray(bundle["src_x_grid"], float) * g["dx"],
                                    src_z_m, nx, g["dx"])
        vp_init = ns.with_air_layer_topo(vp_init, ground, g["dz"])
        water_mask = ns.air_mask_topo(nz, nx, ground, g["dz"])
        n_air = int(water_mask.sum(axis=0).max())
        print(f"    topographic air layer: ground {ground.min():.0f}-"
              f"{ground.max():.0f} m below datum, {n_air} rows at the deepest "
              f"column", flush=True)
    else:
        n_air = ns.air_cells(args.z_air, g["dz"])
        if n_air:
            vp_init = ns.with_air_layer(vp_init, n_air)
        water_mask = ns.air_mask(nz, nx, n_air) if n_air else None
    print("    " + ns.describe(nz, nx, g["dz"],
                               (float(n_air) * g["dz"]) if n_air else args.z_air,
                               VP_BOUND[0], args.f0, g["dx"], src_z=src_z_m),
          flush=True)

    # [5] inversion (Liu's machinery through the T5-patched AcousticFWI)
    rho = np.power(vp_init, 0.25) * 310.0
    model = AcousticModel(0, 0, nx, nz, g["dx"], g["dz"], vp_init, rho,
                          vp_bound=list(VP_BOUND), vp_grad=True,
                          # water_layer_mask makes clip_params keep the
                          # UNCLAMPED value there, so 340 m/s air survives a
                          # 1000 m/s lower bound. Without it the bound destroys
                          # the air layer on the first update.
                          water_layer_mask=water_mask,
                          free_surface=True, abc_type="PML",
                          abc_jerjan_alpha=0.007, nabc=g["nabc"],
                          device=device, dtype=torch.float32)
    prop = AcousticPropagator(model, bundle["survey"], device=device,
                              dtype=torch.float32)
    optimizer = OPTIMIZERS[args.optimizer](model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **SCHEDULER)
    grad_mask = np.ones((nz, nx))
    grad_mask[:max(GRAD_MASK_TOP, n_air), :] = 0        # never update the air
    # >>> AND NEVER UPDATE BELOW THE ILLUMINATION. <<<
    # The fibre ends at ~1.1 km; the grid runs a little deeper because the
    # wavefield needs room. Cells below the deepest receiver are constrained by
    # NO data, so an unmasked gradient is free to invent structure there -- and
    # it does, which is a large part of why our sections look nothing like
    # Park's. Park stop at the fibre. Tapered rather than cut, so the mask does
    # not itself imprint a horizontal edge on the model.
    z_deep = float(np.max(bundle["channel_z_grid"]))
    k_deep = int(round(z_deep / g["dz"]))
    taper = max(1, int(round(0.5 * VP_BOUND[1] / args.f0 / g["dz"])))  # lambda/2
    for k in range(k_deep, nz):
        grad_mask[k, :] = max(0.0, 1.0 - (k - k_deep) / taper)
    print(f"    illumination mask: full weight to {z_deep:.0f} m "
          f"(row {k_deep}), tapered to zero over {taper} rows "
          f"({taper*g['dz']:.0f} m); {nz-k_deep} of {nz} rows are below the "
          f"deepest channel", flush=True)
    settings = RUN_SETTINGS[args.refiner if arm in SOLO_ARMS else args.refiner]

    f_eff = args.f0
    # THE SWITCH. Solo arms pin lambda; `switch` moves it on the MEASURED skip
    # fraction, which needs only syn vs obs -- no true model -- so it works on
    # field data exactly as it does on synthetics.
    loss_fn = BlendedMisfit(build_misfit(args.refiner, iterations, g["dt"], f_eff),
                            build_misfit(args.robust, iterations, g["dt"], f_eff),
                            lam=1.0, normalize=True)
    if args.channel_weight and args.robust == "envelope" and arm not in SOLO_ARMS:
        raise SystemExit(
            "*** --channel-weight with an envelope robust term is REFUSED: it "
            "weights the DATA, so through envelope^1.5 a weak channel is "
            "suppressed 220x harder than intended. It collapsed the switch from "
            "0.742 to 0.26 on Marmousi.")
    if args.window or args.channel_weight:
        loss_fn = ConditionedMisfit(loss_fn, dt=g["dt"], window=args.window,
                                    weight=args.channel_weight,
                                    window_pre=args.window_pre,
                                    window_post=args.window_post)
        print(f"    conditioning: window={args.window} "
              f"({args.window_pre}s/{args.window_post}s) "
              f"channel_weight={args.channel_weight}", flush=True)
    ctrl = SkipSwitch() if arm == "switch" else None
    obs_arr = np.asarray(bundle["obs_data"].data["strain_rate"])

    # ---------------------------------------------------------------------- #
    # --lag-check: WHY is skip stuck at 1.000?
    # ---------------------------------------------------------------------- #
    # The 13-cell campaign reported skip = 1.000 at iterations 0, 25, ... 125 --
    # every trace misaligned by more than T/2 = 50 ms, and 150 iterations never
    # moved it. `convsi` then drove the misfit to 1e-7 anyway, because it is
    # SOURCE-INDEPENDENT: it estimates a matching filter, so a systematic time
    # shift is free to absorb. The model railed to the 340/6000 bounds and
    # learned nothing, while the loss curve looked like a triumph.
    #
    # skip is a FRACTION, so it saturates at 1.0 and cannot say WHY. The lag
    # distribution can:
    #
    #   tight spread about a NON-ZERO median  -> a constant time offset. A
    #       trigger delay or an unapplied SEG-Y delay header. No velocity model
    #       can fix it, and it costs nothing to correct.
    #   wide spread                           -> the starting model really is
    #       far off, and the answer is the envelope/switch path, NOT a waveform
    #       misfit from a skipped start.
    #
    # It runs here, AFTER the real bundle and the real starting model, through
    # the same propagator the inversion uses -- checking a different path than
    # the one that runs is how every other bug on this project survived.
    if args.lag_check:
        from inversion.field_acceptance import arrival_lags
        with torch.no_grad():
            rec = prop.forward(checkpoint_segments=settings["checkpoint_segments"])
            syn0 = bundle["das_layer"](rec["u"], rec["w"]).cpu().numpy()
        half_T = 1.0 / (2.0 * args.f0)
        max_lag_s = 0.5
        L = arrival_lags(syn0, obs_arr, g["dt"], max_lag_s=max_lag_s)
        good = L[np.isfinite(L)]
        if good.size == 0:
            print("*** LAG CHECK: no usable traces -- every trace was rejected "
                  "as dead or zero-amplitude. That is the bug.", flush=True)
            return 3
        med = float(np.median(good))
        iqr = float(np.percentile(good, 75) - np.percentile(good, 25))
        mad = float(np.median(np.abs(good - med)))
        beyond = float(np.mean(np.abs(good) > half_T))
        print(f"\n=== LAG CHECK ({good.size} of {L.size} traces usable) ===\n"
              f"    median lag   {med*1e3:+8.1f} ms   (syn minus obs)\n"
              f"    IQR          {iqr*1e3:8.1f} ms\n"
              f"    MAD          {mad*1e3:8.1f} ms\n"
              f"    |lag| > T/2  {beyond*100:7.1f} %   (T/2 = {half_T*1e3:.0f} ms"
              f" at f0={args.f0:g} Hz)\n"
              f"    range        {good.min()*1e3:+.1f} .. {good.max()*1e3:+.1f} ms",
              flush=True)
        # A lag pinned at the search limit is a CENSORED measurement, not a
        # measurement. Quoting IQR/MAD from a railed distribution understates
        # the true spread and reads as though it were exact.
        railed = float(np.mean(np.abs(good) > 0.98 * max_lag_s))
        if railed > 0.02:
            print(f"    *** {railed*100:.0f}% of lags are AT the +-{max_lag_s*1e3:.0f} ms "
                  f"search limit: the spread above is a LOWER BOUND, not the "
                  f"true value. Raise max_lag_s to measure it.", flush=True)
        # Whole traces are correlated, and field DAS carries surface waves an
        # ACOUSTIC model cannot produce, so some of this spread is events the
        # physics was never going to match. --window is the mitigation and is
        # NOT applied here; read the spread as pessimistic.
        # A CONSTANT offset means the spread is small compared with the shift
        # itself AND small compared with the half period we must land inside.
        constant = abs(med) > half_T and mad < 0.5 * half_T
        if constant:
            print(f"    VERDICT: CONSTANT OFFSET of {med*1e3:+.0f} ms. The "
                  f"spread (MAD {mad*1e3:.0f} ms) is small next to the shift, so "
                  f"this is a TIMING problem, not a velocity one.\n"
                  f"    Check the SEG-Y delay header, the trigger/zero time, and "
                  f"the source wavelet delay. No inversion can remove it.",
                  flush=True)
        elif beyond > 0.5:
            print(f"    VERDICT: WIDE SPREAD, {beyond*100:.0f}% beyond T/2. The "
                  f"starting model is genuinely far off.\n"
                  f"    A waveform misfit from here cannot work -- use the "
                  f"envelope/switch path, or a lower f0 to widen T/2.",
                  flush=True)
        else:
            print(f"    VERDICT: mostly aligned ({beyond*100:.0f}% beyond T/2). "
                  f"Skip=1.000 in the campaign is then NOT explained by the "
                  f"lags, and the skip diagnostic itself is suspect.",
                  flush=True)
        return 0

    gp_kw = {}
    if args.grad_smooth == "wavelength":
        sx, sz = ns.anisotropic_span(VP_BOUND[0], args.f0, g["dx"], g["dz"])
        gp_kw["grad_smooth"] = sz          # ADFWI's own span is isotropic; the
        print(f"    grad smoothing {sx}x{sz} cells "  # anisotropic pass is applied
              f"({sx*g['dx']:.0f}x{sz*g['dz']:.0f} m) -- ADFWI applies the "
              f"vertical span; H:V handled by near_surface", flush=True)

    fwi = AcousticFWI(propagator=prop, model=model,
                      optimizer=optimizer, scheduler=scheduler,
                      loss_fn=loss_fn,
                      obs_data=bundle["obs_data"],
                      gradient_processor=GradProcessor(grad_mask=grad_mask,
                                                       grad_mute=n_air,
                                                       marine_or_land=("marine"
                                                           if n_air else "land"),
                                                       **gp_kw),
                      waveform_normalize=settings["normalize"],
                      cache_result=True, cache_result_epoch=CACHE_EVERY,
                      save_fig_epoch=-1,
                      das_layer=bundle["das_layer"], obs_key="strain_rate")
    # Chunked so a walltime kill keeps the latest model + curves. forward()
    # accumulates iter_vp/iter_loss and takes start_iter, and the scheduler steps
    # once per iteration regardless, so chunking is trajectory-identical to one
    # call. FIELD runs are long and have no truth to fall back on -- losing one
    # to the walltime would be a total loss.

    def _save(done, hours, complete):
        iter_loss = np.asarray(fwi.iter_loss)
        np.savez(out_dir / "iter_vp.npz", data=np.asarray(fwi.iter_vp))
        np.savez(out_dir / "iter_loss.npz", data=iter_loss)
        vp_final = model.vp.detach().cpu().numpy()
        grad_final = (model.vp.grad.detach().cpu().numpy()
                      if model.vp.grad is not None else np.zeros_like(vp_final))
        m = dict(
            tag=tag, well=args.well, n_shots=bundle["n_shots"], device=device,
            optimizer=args.optimizer,
            # >>> RECORD THE GRID AND THE GEOMETRY. <<<
            # Without these, results from different grids sit in one directory
            # indistinguishable: 21 cells accumulated across the pre- and
            # post-geometry-fix runs and nothing in metrics.json could tell them
            # apart. The ranker also had to be TOLD dz with a flag, defaulted to
            # the old 20 m, and reported zone boundaries at twice their depth.
            # A result that does not carry its own configuration is not
            # interpretable later.
            dz=g["dz"], dx=g["dx"], nz=int(nz), nx=int(nx),
            nt=int(g["nt"]), dt=float(g["dt"]), f0=float(args.f0),
            chan_z_min=float(np.min(bundle["channel_z_grid"])),
            chan_z_max=float(np.max(bundle["channel_z_grid"])),
            n_channels=int(len(bundle["channel_z_grid"])),
            # `diverged` was never set here, so the ranker could not filter a
            # dead cell. Field runs have no truth, so the ONLY divergence
            # signals are the loss and the model itself.
            diverged=bool(not np.isfinite(iter_loss).all()
                          or not np.isfinite(vp_final).all()
                          or np.abs(vp_final).max() > 1e5),
            model_finite=bool(np.isfinite(vp_final).all()),
            iterations=iterations, iterations_done=int(done),
            complete=bool(complete), runtime_h=round(hours, 3),
            loss_first=float(iter_loss[0]) if len(iter_loss) else None,
            loss_last=float(iter_loss[-1]) if len(iter_loss) else None,
            # RAW misfit, un-normalised. The switch arms report loss values like
            # 1.9e18 because BlendedMisfit returns E/||grad E|| -- meaningful for
            # weighting the two terms, meaningless as a progress measure, and
            # divided by a DIFFERENT EMA scale each iteration. Percent reduction
            # must be computed from these instead.
            **_raw_loss_fields(loss_fn),
            loss_decreased=bool(len(iter_loss) > 1 and iter_loss[-1] < iter_loss[0]),
            losses_finite=bool(np.isfinite(iter_loss).all()),
            grad_finite=bool(np.isfinite(grad_final).all()),
            grad_nonzero=bool(np.abs(grad_final).max() > 0),
            vp_final_range=[float(vp_final.min()), float(vp_final.max())],
            arm=arm, refiner=args.refiner, robust=args.robust,
            starting=args.starting, bands=[("full" if b is None else b)
                                           for b in bands],
            iters_per_band=list(iters_by_band), iter_alloc=args.iter_alloc,
            z_air=args.z_air, n_air_rows=int(n_air),
            window=bool(args.window), channel_weight=bool(args.channel_weight),
            vp_bound=list(VP_BOUND), grad_smooth=args.grad_smooth,
            handovers=(ctrl.handbacks if ctrl is not None else None),
            trajectory=traj)
        (out_dir / "metrics.json").write_text(json.dumps(m, indent=2))
        return vp_final, iter_loss, m

    def measure_skip(f_band):
        """Skip fraction on the RAW data. Needs only syn vs obs -- NO true model
        -- which is why the switch transfers to field data unchanged."""
        with torch.no_grad():
            rec = prop.forward(checkpoint_segments=settings["checkpoint_segments"])
            syn = bundle["das_layer"](rec["u"], rec["w"]).cpu()
        return float(skip_fraction(syn, obs_arr, g["dt"], f_band)["skip_fraction"])

    t0 = time.time()
    done = 0
    traj = []
    for bi, f_band in enumerate(bands):
        fb = args.f0 if f_band is None else min(f_band, args.f0)
        band_iters = iters_by_band[bi]
        # A FRESH controller per band: raising the band raises f_max, so skip
        # jumps at every boundary and the controller must be free to re-enter
        # the robust stage -- one persistent switch's ratchet would block it.
        if arm == "switch":
            ctrl = SkipSwitch()
        print(f"--- band {bi+1}/{len(bands)} [{band_iters} iters]: "
              f"cutoff={'full' if f_band is None else f'{f_band} Hz'} "
              f"(f_eff={fb:.2f}, T/2={1000/(2*fb):.0f} ms) ---", flush=True)
        in_band = 0
        while in_band < band_iters:
            sk = measure_skip(fb)
            if not np.isfinite(sk):
                # The smoke showed `skip=nan -> lambda=0` on REAL data, i.e. the
                # controller handed straight to the refiner at an unknown skip
                # level -- the worst available move and the exact thing it
                # exists to prevent. Guarded in the synthetic driver but NOT
                # here. Fail SAFE: treat an unmeasurable skip as fully skipped.
                print("  *** skip is NaN -- assuming SKIPPED (fail-safe)",
                      flush=True)
                sk = 1.0
            if arm == "switch":
                lam = ctrl.update(sk)
            elif arm == "fixedk":
                lam = 1.0 if done < args.fixed_k else 0.0
            else:
                lam = 0.0 if arm in SOLO_ARMS else 1.0
            loss_fn.set_lambda(lam)
            mode = args.robust.upper() if lam >= 0.5 else args.refiner.upper()
            print(f"  iter {done:3d}: skip={sk:.3f} -> lambda={lam:.0f} ({mode})",
                  flush=True)
            traj.append(dict(iter=done, band=(None if f_band is None else f_band),
                             skip=sk, lam=float(lam)))
            n = min(args.chunk, band_iters - in_band)
            fwi.forward(iteration=n, start_iter=done,
                        batch_size=(settings["batch_size"] or args.batch_shots),
                        checkpoint_segments=settings["checkpoint_segments"],
                        cutoff_freq=f_band)
            done += n; in_band += n
            _save(done, (time.time() - t0) / 3600.0,
                  complete=(done >= sum(iters_by_band)))
            print(f"  checkpoint {done}/{sum(iters_by_band)}", flush=True)
    iterations = sum(iters_by_band)
    hours = (time.time() - t0) / 3600.0

    # [6] outputs (no RMS-vs-truth for field data)
    vp_final, iter_loss, metrics = _save(iterations, hours, complete=True)
    print(json.dumps(metrics, indent=2), flush=True)

    from forge.plot_field_result import velocity_panel
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    # clip the panel at the deepest RECEIVER: below it nothing constrains the
    # model, so plotting further shows the starting model dressed as a result.
    z_max_km = float(np.max(bundle["channel_z_grid"])) / 1000.0
    ground_m = (ns.surface_profile(bundle["src_x_grid"] * g["dx"],
                                   bundle["grid"]["src_z_grid"] * g["dz"],
                                   nx, g["dx"])
                if args.topo_air else None)
    for ax, (d, ttl) in zip(axes[:2], [(vp_init, f"initial ({args.starting})"),
                                       (vp_final, f"inverted {tag}")]):
        im = velocity_panel(ax, d, g["dx"], g["dz"],
                            vmin=VP_BOUND[0], vmax=VP_BOUND[1],
                            ground=ground_m, z_max_km=z_max_km,
                            well_x_km=bundle["well_x_index"] * g["dx"] / 1000,
                            title=f"vp {ttl} [m/s]")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Vp [m/s]")
    axes[2].plot(iter_loss, "k.-")
    axes[2].set(title="loss", xlabel="iteration")
    fig.savefig(out_dir / "final.png", dpi=150)
    print("saved results to", out_dir, flush=True)


if __name__ == "__main__":
    main()
