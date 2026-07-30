"""PHASE B: multiscale DAS-FWI with the frequency-adaptive L2 -> ROBUST objective.

Climbs a low->high frequency cascade (AcousticFWI's `cutoff_freq`, which
low-passes BOTH synthetic and observed identically and differentiably) while the
blend weight lambda ramps from L2 (resolution, safe at low frequency) to the
robust term (needed once cycle skipping bites).

!! The robust term is now `envelope`, NOT sinkhorn. The Phase-1 gate (2026-07-29)
REFUTED the OT hypothesis: sinkhorn is never above L2 and craters with it under
skip, while the phase-insensitive envelope family wins from rung s16 on. Keep
`--hi sinkhorn` available only as a control arm for the record.
!! Do the SINGLE-SCALE Phase A test (run_switch.py at s16) BEFORE this driver:
multiscale changes both the skip measurement (band-limited lag, larger T/2) and
the physics (frequency continuation), so the gate's thresholds must be
re-verified per band first.

Three ARMS, so the acceptance test is a single flag:
    --objective adaptive     the L2 -> envelope blend      (the proposal)
    --objective l2           fixed L2 at every band        (control 1)
    --objective envelope     fixed robust at every band    (control 2)
Any other registry misfit name also works as a control (except weci, which is
stateful and rejected as a blend term -- see adaptive_misfit._reject_stateful).

ACCEPTANCE: adaptive >= the better fixed arm in final SSIM (tol ~0.01), with
final-band MAPE at L2 grade, and never worse than L2 below the flip band.

    python hpc/marmousi_full_das/run_adaptive.py --objective adaptive \
        --start-rung s20 --bands 3.0,4.5,6.25,full --iters 60

The lambda schedule's (f_lo, f_hi) MUST come from the Phase-1 flip curve; the
defaults here are placeholders, not physics. Pass --flip-lo/--flip-hi.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import (OUT_ROOT, OBS_FILE, NZ, NX, DX, DZ, DT, F0, NT,
                    MISFITS, OPTIMIZERS, MISFIT_RUN_SETTINGS,
                    START_RUNGS, DEFAULT_RUNG,
                    pick_device, load_models, build_model, build_geometry,
                    build_survey, build_misfit, build_regularization,
                    build_gradient_processor, DASObservationLayer,
                    SeismicData, AcousticPropagator)

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ADFWI.fwi import AcousticFWI
from inversion.adaptive_misfit import BlendedMisfit, LambdaSchedule
from inversion.metrics import model_scores
from inversion.skip_diagnostic import skip_fraction, ricker_f90

#: placeholder flip point - REPLACE from the Phase-1 flip curve
DEFAULT_FLIP_LO, DEFAULT_FLIP_HI = 3.0, 8.0
DEFAULT_BANDS = "3.0,4.5,6.25,full"


def parse_bands(spec):
    """'3.0,4.5,full' -> [3.0, 4.5, None]  (None = unfiltered, full band)."""
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        out.append(None if tok in ("full", "none", "0") else float(tok))
    if not out:
        raise ValueError(f"no bands parsed from {spec!r}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="adaptive",
                    help="'adaptive' or any registry misfit name (control arm)")
    ap.add_argument("--optimizer", default="adam", choices=sorted(OPTIMIZERS))
    ap.add_argument("--bands", default=DEFAULT_BANDS,
                    help="comma-separated low-pass cut-offs, 'full' for unfiltered")
    ap.add_argument("--iters", type=int, default=75, help="iterations PER BAND")
    ap.add_argument("--start-rung", default=DEFAULT_RUNG, choices=START_RUNGS,
                    dest="start_rung")
    ap.add_argument("--lo", default="l2", help="low-frequency term of the blend")
    ap.add_argument("--hi", default="envelope",
                    help="robust term of the blend (gate: envelope, NOT sinkhorn)")
    ap.add_argument("--flip-lo", type=float, default=DEFAULT_FLIP_LO,
                    dest="flip_lo", help="frequency where lambda starts to rise")
    ap.add_argument("--flip-hi", type=float, default=DEFAULT_FLIP_HI,
                    dest="flip_hi", help="frequency where lambda reaches 1")
    ap.add_argument("--regularization", default="none")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--smoke", action="store_true", help="2 iters/band")
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = torch.float32
    bands = parse_bands(args.bands)
    iters = 2 if args.smoke else args.iters
    adaptive = args.objective.lower() == "adaptive"
    if not adaptive and args.objective not in MISFITS:
        raise SystemExit(f"--objective must be 'adaptive' or one of {MISFITS}")

    tag = args.tag or (f"adaptive_{args.lo}-{args.hi}" if adaptive
                       else f"fixed_{args.objective}")
    tag = f"{tag}_{args.optimizer}_{args.start_rung}" + ("_smoke" if args.smoke else "")
    out_dir = OUT_ROOT / "adaptive" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    f90 = ricker_f90(F0, DT, NT, integrated=True)
    print(f"=== {tag} on {device} | bands={bands} x {iters} iters | "
          f"source f90={f90:.2f} Hz ===", flush=True)

    vp_true, vp_init = load_models(args.start_rung)
    geometry = build_geometry()
    survey = build_survey(geometry)
    layer = DASObservationLayer(geometry, output="strain_rate").to(dtype).to(device)
    obs_data = SeismicData(survey)
    obs_data.load(str(OUT_ROOT / OBS_FILE))
    obs_arr = torch.as_tensor(obs_data.data["strain_rate"])

    model = build_model(vp_init, vp_bound=[float(vp_true.min()), float(vp_true.max())],
                        vp_grad=True, device=device, dtype=dtype)
    prop = AcousticPropagator(model, survey, device=device, dtype=dtype)
    optimizer = OPTIMIZERS[args.optimizer](model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.75)

    total_iters = iters * len(bands)
    if adaptive:
        loss_fn = BlendedMisfit(build_misfit(args.lo, iterations=total_iters),
                                build_misfit(args.hi, iterations=total_iters),
                                lam=0.0)
        settings = MISFIT_RUN_SETTINGS[args.hi]     # the costlier term sets batching
        schedule = LambdaSchedule(args.flip_lo, args.flip_hi)
        print(f"    schedule {schedule}", flush=True)
    else:
        loss_fn = build_misfit(args.objective, iterations=total_iters)
        settings = MISFIT_RUN_SETTINGS[args.objective]
        schedule = None

    reg_fn, wx, wz = build_regularization(args.regularization, device, dtype)
    fwi = AcousticFWI(propagator=prop, model=model, optimizer=optimizer,
                      scheduler=scheduler, loss_fn=loss_fn, obs_data=obs_data,
                      gradient_processor=build_gradient_processor(),
                      regularization_fn=reg_fn, regularization_weights_x=wx,
                      regularization_weights_z=wz,
                      waveform_normalize=settings["normalize"],
                      cache_result=True, cache_result_epoch=10, save_fig_epoch=-1,
                      das_layer=layer, obs_key="strain_rate")

    def measure_skip(f_eff):
        with torch.no_grad():
            rec = prop.forward(checkpoint_segments=settings["checkpoint_segments"])
            syn = layer(rec["u"], rec["w"]).cpu()
        return skip_fraction(syn, obs_arr, DT, f_eff)

    # ---- the cascade -------------------------------------------------------
    band_log, start = [], 0
    t0 = time.time()
    for bi, f_band in enumerate(bands):
        f_eff = f90 if f_band is None else min(f_band, f90)
        lam = schedule.lam(f_eff) if adaptive else None
        if adaptive:
            loss_fn.set_lambda(lam)
        try:
            sk = measure_skip(f_eff)["skip_fraction"]
        except Exception:                                    # noqa: BLE001
            sk = float("nan")
        print(f"--- band {bi+1}/{len(bands)}: cutoff="
              f"{'full' if f_band is None else f'{f_band} Hz'} "
              f"(f_eff={f_eff:.2f}) lambda={'-' if lam is None else f'{lam:.3f}'} "
              f"skip@start={sk:.3f} ---", flush=True)
        fwi.forward(iteration=iters, batch_size=settings["batch_size"],
                    checkpoint_segments=settings["checkpoint_segments"],
                    start_iter=start, cutoff_freq=f_band)
        start += iters
        band_log.append(dict(band=bi + 1, cutoff=f_band, f_eff=f_eff,
                             lam=lam, skip_at_start=sk,
                             scales=(loss_fn.scales if adaptive else None)))
    hours = (time.time() - t0) / 3600.0

    # ---- outputs -----------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    iter_vp = np.asarray(fwi.iter_vp)
    iter_loss = np.asarray(fwi.iter_loss)
    np.savez(out_dir / "iter_vp.npz", data=iter_vp)
    np.savez(out_dir / "iter_loss.npz", data=iter_loss)
    vp_final = model.vp.detach().cpu().numpy()
    sc = model_scores(vp_true, vp_final)
    try:
        skip_end = measure_skip(f90)["skip_fraction"]
    except Exception:                                        # noqa: BLE001
        skip_end = None

    metrics = dict(
        tag=tag, objective=args.objective, optimizer=args.optimizer,
        start_rung=args.start_rung, bands=[("full" if b is None else b) for b in bands],
        iters_per_band=iters, adaptive=adaptive,
        flip_lo=args.flip_lo if adaptive else None,
        flip_hi=args.flip_hi if adaptive else None,
        device=device, runtime_h=round(hours, 3),
        rms_init=float(np.sqrt(((vp_init - vp_true) ** 2).mean())),
        rms_final=float(np.sqrt(((vp_final - vp_true) ** 2).mean())),
        ssim=sc["ssim"], mape=sc["mape"], skip_final=skip_end,
        loss_first=float(iter_loss[0]), loss_last=float(iter_loss[-1]),
        losses_finite=bool(np.isfinite(iter_loss).all()),
        band_log=band_log,
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps({k: v for k, v in metrics.items() if k != "band_log"},
                     indent=2, default=str), flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    ext = [0, (NX - 1) * DX / 1000, (NZ - 1) * DZ / 1000, 0]
    for ax, (d, ttl) in zip(axes.flat[:3], [(vp_true, "true"), (vp_init, "initial"),
                                            (vp_final, f"inverted ({tag})")]):
        im = ax.imshow(d, extent=ext, cmap="jet", vmin=vp_true.min(), vmax=vp_true.max())
        ax.set(title=f"vp {ttl} [m/s]", xlabel="x [km]", ylabel="z [km]")
        fig.colorbar(im, ax=ax, shrink=0.8)
    axes.flat[3].plot(iter_loss, "k.-", ms=3)
    for bi in range(1, len(bands)):
        axes.flat[3].axvline(bi * iters, color="r", ls="--", lw=.8)
    axes.flat[3].set(title="loss (red = band change)", xlabel="iteration")
    fig.savefig(out_dir / "final.png", dpi=150)
    print("saved results to", out_dir, flush=True)


if __name__ == "__main__":
    main()
