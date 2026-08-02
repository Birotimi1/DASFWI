"""PHASE A: the STAGED envelope->L2 misfit switch (Fable-verified design).

One cell of the switch experiment on the acoustic Marmousi campaign setup,
SINGLE-SCALE FULL BAND (the exact conditions the Phase-1 gate calibrated):

    python hpc/marmousi_full_das/run_switch.py --arm switch --start-rung s16 \
        --optimizer adam [--iterations 300] [--chunk 25] [--fixed-k 150]

Arms:
  switch    envelope while the MEASURED skip fraction is high, hand over to L2
            when it falls below off_below. Controller = SkipSwitch (init from
            first measurement, EMA smoothing, dwell, hand-back ratchet with
            logged re-entries). Both stages run through BlendedMisfit(l2,
            envelope) at binary lambda: the short-circuit never evaluates the
            unused term, and the detached grad-norm normalization keeps the
            adjoint-source norm ~unit across the hand-over so Adam/adagrad
            moment states see no rescale shock. (For sgd the normalization
            rescales the effective step size vs the raw-loss gate controls --
            rank sgd against the other arms here, not against gate cells.)
  fixedk    the DUMB CONTROL the reviewer demanded: envelope for --fixed-k
            iterations, then L2 -- no diagnostic at all. If this matches the
            switch arm, the skip machinery isn't earning its complexity.
  l2 | envelope   pure single-misfit arms THROUGH THE SAME BlendedMisfit
            normalization (lambda pinned 0 / 1) -- only needed if you want
            normalization-matched controls; the raw controls already exist in
            the gate results (l2_*, envelope_*, weci_* at every rung).

Skip is measured every --chunk iterations from ONE extra no-grad forward on the
RAW full-band data -- exactly the gate's calibration measurement (~1-2%
overhead). Checkpoints (iter_vp/iter_loss/metrics.json + the full skip/lambda
trajectory) are written every chunk, so a walltime kill loses nothing.
"""
import argparse
import re
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import (OUT_ROOT, OBS_FILE, ITERATIONS, NZ, NX, DX, DZ, DT, F0, NT,
                    OPTIMIZERS, MISFIT_RUN_SETTINGS, START_RUNGS,
                    pick_device, load_models, build_model, build_geometry,
                    build_survey, build_misfit, build_gradient_processor,
                    DASObservationLayer, SeismicData, AcousticPropagator)

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ADFWI.fwi import AcousticFWI
from inversion.metrics import model_scores
from inversion.skip_diagnostic import skip_fraction, ricker_f90
from inversion.das_conditioning import ConditionedMisfit, wavelength_span
from inversion.adaptive_misfit import (BlendedMisfit, SkipSwitch,
                                       StagedMisfit, StageLadder,
                                       SKIP_ON_ABOVE, SKIP_OFF_BELOW,
                                       LADDER_THRESHOLDS)

ARMS = ("switch", "fixedk", "ladder", "l2", "envelope")
LADDER_STAGES = ("envelope", "gc", "l2")     # robust -> gentle -> sharp refiner


def _starter_file(name=None):
    """Path to a Route B starter. Each convergence level lives in its own
    subdirectory (starter/i20, starter/i80, ...); with no name, pick the most
    converged one available so a single-starter setup needs no flag."""
    root = OUT_ROOT / "starter"
    if name:
        return root / name / "vp_start.npz"
    cands = sorted(root.glob("*/vp_start.npz")) if root.is_dir() else []
    if not cands:
        return root / "i80" / "vp_start.npz"          # canonical, for the message
    def _n(p):
        m = re.match(r"i(\d+)$", p.parent.name)
        return int(m.group(1)) if m else -1
    return max(cands, key=_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    # The gate + mining showed the PAIR matters: weci = envelope->gc staged
    # (Weci.py composes Misfit_envelope and Misfit_global_correlation) scores
    # 0.451 at s16 while envelope alone is 0.240 and gc alone 0.210. l2 alone
    # (0.326) refines better than gc alone, so --refiner l2 is the proposal;
    # --refiner gc reproduces weci's exact pair under SKIP-DRIVEN timing, which
    # isolates timing (vs weci's hardcoded iteration-150 sigmoid) as the variable.
    ap.add_argument("--refiner", default="l2", choices=("l2", "gc"),
                    help="the resolution term handed over TO (lambda=0)")
    ap.add_argument("--robust", default="envelope", choices=("envelope",),
                    help="the cycle-skip-tolerant term (lambda=1); stateless only")
    ap.add_argument("--optimizer", default="adam", choices=sorted(OPTIMIZERS))
    ap.add_argument("--starter", default=None,
                    help="which Route B starter to use, e.g. i20 (partly "
                         "converged -> more skipping) or i80. Default: the only "
                         "one present, else the most converged.")
    ap.add_argument("--start", default="rung", choices=("rung", "route_b"),
                    help="route_b = the wave-equation cross-correlation starter "
                         "(THE PLAN: what we will have at FORGE); rung = the "
                         "smoothed-truth ladder (controlled proof only)")
    ap.add_argument("--band", type=float, default=None,
                    help="low-pass cut-off (Hz). Selects the REGIME from a fixed "
                         "start: low band = non-skip, high/None = full band = "
                         "skip. Read the starter's skip-vs-band table to choose.")
    ap.add_argument("--start-rung", default="s16", choices=START_RUNGS,
                    dest="start_rung",
                    help="s16 is the primary test point (64%% skip: L2 collapsed,"
                         " envelope-family recovers); s20 secondary; s6 sanity")
    ap.add_argument("--iterations", type=int, default=ITERATIONS)
    ap.add_argument("--chunk", type=int, default=25,
                    help="iterations per controller update + checkpoint")
    ap.add_argument("--fixed-k", type=int, default=150, dest="fixed_k",
                    help="fixedk arm: envelope iterations before the L2 leg")
    ap.add_argument("--on-above", type=float, default=SKIP_ON_ABOVE, dest="on_above")
    ap.add_argument("--off-below", type=float, default=SKIP_OFF_BELOW, dest="off_below")
    ap.add_argument("--ladder-thresholds", default=None, dest="ladder_thresholds",
                    help="ladder arm: descending skip thresholds, one per "
                         f"hand-over (default {','.join(map(str, LADDER_THRESHOLDS))} "
                         f"for stages {'->'.join(LADDER_STAGES)})")
    ap.add_argument("--dwell", type=int, default=1,
                    help="minimum controller updates per mode (chunks)")
    # --- Noe et al. 2025 conditioning (opt-in, so results stay comparable) ---
    ap.add_argument("--window", action="store_true",
                    help="keep only --window-pre/--window-post around each "
                         "trace's peak (Noe use 2 s / 4 s). Also drops the LATE "
                         "arrivals that carry the most phase error.")
    ap.add_argument("--window-pre", type=float, default=2.0, dest="window_pre")
    ap.add_argument("--window-post", type=float, default=4.0, dest="window_post")
    ap.add_argument("--channel-weight", action="store_true", dest="channel_weight",
                    help="weight channels by observed amplitude, so broadside-"
                         "insensitive / poorly coupled ones do not dominate")
    ap.add_argument("--grad-smooth", choices=("none", "wavelength"),
                    default="none", dest="grad_smooth",
                    help="wavelength = smooth the gradient at ~lambda/4 "
                         "(frequency-aware) instead of leaving it unsmoothed")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="validate the plan and exit (no GPU, no data). RUN FIRST.")
    ap.add_argument("--smoke", action="store_true", help="2 chunks of 2 iters")
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = torch.float32
    iterations = 4 if args.smoke else args.iterations
    chunk = 2 if args.smoke else max(1, args.chunk)
    # encode the pair in the tag unless it is the default envelope->l2, so the
    # envelope->gc variant cannot overwrite the proposal's results
    pair = "" if args.refiner == "l2" else f"-{args.refiner}"
    _b = "full" if args.band is None else f"{args.band:g}Hz"
    # the STARTER NAME must be in the tag: i20 and i80 are different experiments
    # (that is the whole point of the convergence axis) and would otherwise
    # overwrite each other -- the same collision already hit --timing and --bands
    _st = ("" if args.start == "rung"
           else "_" + _starter_file(args.starter).parent.name)
    _cond = ("" + ("w" if args.window else "") + ("c" if args.channel_weight else "")
             + ("g" if args.grad_smooth != "none" else ""))
    tag = f"{args.arm}{pair}_{args.optimizer}{_st}_{_b}" + (f"_{_cond}" if _cond else "")
    if args.smoke:
        tag = "smoke_" + tag
    _grp = "routeb" if args.start == "route_b" else args.start_rung
    out_dir = OUT_ROOT / f"switch_{_grp}" / tag
    problems = []
    if not (OUT_ROOT / OBS_FILE).is_file():
        (print(f"    NOTE: no observed data at {OUT_ROOT / OBS_FILE} "
               "(fine for --dry-run; run genobs before the real job)")
         if args.dry_run else
         problems.append(f"no observed data at {OUT_ROOT / OBS_FILE} (run genobs)"))
    if args.start == "route_b":
        _f = _starter_file(args.starter)
        if not _f.is_file():
            problems.append(f"--start route_b but no starter at {_f} -- run "
                            "hpc/marmousi_full_das/run_traveltime_starter.py "
                            "(PLAN STEP 1) first")
        else:
            try:
                _shp = tuple(np.load(_f)["vp_start"].shape)
                if _shp != (NZ, NX):
                    problems.append(f"starter is {_shp}, this grid is {(NZ, NX)}"
                                    " -- rebuild it on this grid")
            except Exception as e:                       # noqa: BLE001
                problems.append(f"cannot read {_f}: {type(e).__name__}: {e}")
    if args.band is not None and args.band <= 0:
        problems.append(f"--band {args.band} must be positive (omit it for full band)")
    if chunk > iterations:
        problems.append(f"--chunk {chunk} > --iterations {iterations}: the "
                        "controller would never update")
    if (out_dir / "metrics.json").is_file():
        print(f"    NOTE: {out_dir} already has results -- this OVERWRITES them",
              flush=True)
    for pb in problems:
        print(f"    *** {pb}", flush=True)
    if problems:
        raise SystemExit("preflight FAILED -- nothing was run")
    if args.dry_run:
        _f90 = ricker_f90(F0, DT, NT, integrated=True)
        _fe = _f90 if args.band is None else min(args.band, _f90)
        print(f"    dry-run OK: arm={args.arm} {args.robust}->{args.refiner} "
              f"start={args.start}"
              + (f"({args.start_rung})" if args.start == "rung" else "")
              + f" band={'full' if args.band is None else str(args.band)+'Hz'} "
              f"-> f_eff={_fe:.2f} Hz, T/2={1000/(2*_fe):.0f} ms | "
              f"{iterations} iters, chunk {chunk} -> {out_dir.name}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== PHASE A {tag} [start={args.start_rung}] on {device}, "
          f"{iterations} iters in chunks of {chunk} ===", flush=True)

    # ---- setup: identical to the gate's run_one.py -----------------------------
    # THE GOVERNING PLAN uses --start route_b: a Route B starting model (wave-
    # equation cross-correlation, no picking) is what we will have at FORGE, so
    # it is what the validation runs from. The smoothed-truth rungs remain
    # available (--start rung) but they leak the true model and are only the
    # controlled proof-of-concept, not the deployable validation.
    vp_true, vp_rung = load_models(args.start_rung)
    if args.start == "route_b":
        f = _starter_file(args.starter)
        if not f.is_file():
            raise SystemExit(
                f"--start route_b but no starter at {f}; run "
                "hpc/marmousi_full_das/run_traveltime_starter.py first")
        vp_init = np.asarray(np.load(f)["vp_start"], np.float64)
        if vp_init.shape != vp_true.shape:
            raise SystemExit(f"starter is {vp_init.shape}, grid is "
                             f"{vp_true.shape} -- rebuild it on this grid")
        print(f"    Route B starter loaded from {f}", flush=True)
    else:
        vp_init = vp_rung
    geometry = build_geometry()
    survey = build_survey(geometry)
    layer = DASObservationLayer(geometry, output="strain_rate").to(dtype).to(device)
    obs_data = SeismicData(survey)
    obs_data.load(str(OUT_ROOT / OBS_FILE))
    obs_arr = torch.as_tensor(obs_data.data["strain_rate"])
    f90 = ricker_f90(F0, DT, NT, integrated=True)
    # The band SELECTS THE REGIME: skipping is |dt| > T/2 = 1/(2 f_eff), so a low
    # band is the NON-SKIP test and a high band the SKIP test, from one and the
    # same starting model. Skip must be measured at the band actually being
    # inverted, not at f90, or the controller sees the wrong regime.
    f_eff = f90 if args.band is None else min(args.band, f90)

    model = build_model(vp_init,
                        vp_bound=[float(vp_true.min()), float(vp_true.max())],
                        vp_grad=True, device=device, dtype=dtype)
    prop = AcousticPropagator(model, survey, device=device, dtype=dtype)
    optimizer = OPTIMIZERS[args.optimizer](model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100,
                                                gamma=0.75, last_epoch=-1)

    # All terms must be STATELESS full-gather misfits (batch_size=None,
    # normalize=True); weci is refused because it carries its own iteration
    # schedule -- see adaptive_misfit._reject_stateful.
    if args.arm == "ladder":
        thr = ([float(t) for t in args.ladder_thresholds.split(",")]
               if args.ladder_thresholds else list(LADDER_THRESHOLDS))
        loss_fn = StagedMisfit([build_misfit(m, iterations=iterations)
                                for m in LADDER_STAGES],
                               names=LADDER_STAGES, normalize=True)
        ladder = StageLadder(thr, dwell=args.dwell)
        print(f"    ladder stages {'->'.join(LADDER_STAGES)} at skip {thr}",
              flush=True)
    else:
        loss_fn = BlendedMisfit(build_misfit(args.refiner, iterations=iterations),
                                build_misfit(args.robust, iterations=iterations),
                                lam=1.0, normalize=True)
        ladder, thr = None, None
    settings = MISFIT_RUN_SETTINGS[args.refiner]   # l2/gc/envelope all match
    if args.window or args.channel_weight:
        loss_fn = ConditionedMisfit(loss_fn, dt=DT, window=args.window,
                                    weight=args.channel_weight,
                                    window_pre=args.window_pre,
                                    window_post=args.window_post)
        print(f"    conditioning: window={args.window} "
              f"({args.window_pre}s/{args.window_post}s) "
              f"channel_weight={args.channel_weight}", flush=True)

    fwi = AcousticFWI(propagator=prop, model=model, optimizer=optimizer,
                      scheduler=scheduler, loss_fn=loss_fn, obs_data=obs_data,
                      gradient_processor=build_gradient_processor(
                          grad_smooth=(wavelength_span(1500.0, f_eff, DX)
                                       if args.grad_smooth == "wavelength" else 0)),
                      waveform_normalize=settings["normalize"],
                      cache_result=True, cache_result_epoch=10,
                      save_fig_epoch=-1,
                      das_layer=layer, obs_key="strain_rate")

    def measure_skip():
        """One no-grad forward on the RAW full-band data -- the gate's exact
        calibration measurement (thresholds are only valid against this)."""
        with torch.no_grad():
            rec = prop.forward(checkpoint_segments=settings["checkpoint_segments"])
            syn = layer(rec["u"], rec["w"]).cpu()
        return skip_fraction(syn, obs_arr, DT, f_eff)["skip_fraction"]

    ctrl = SkipSwitch(on_above=args.on_above, off_below=args.off_below,
                      dwell=args.dwell,
                      max_robust=max(1, -(-iterations // chunk))
                      ) if args.arm == "switch" else None

    def lam_for(done, skip):
        if args.arm == "switch":
            return ctrl.update(skip)
        if args.arm == "fixedk":
            return 1.0 if done < args.fixed_k else 0.0
        return 0.0 if args.arm == "l2" else 1.0

    traj = []                                 # (iter, skip, lam) per chunk

    def _save(done, hours, complete):
        iter_vp = np.asarray(fwi.iter_vp)
        iter_loss = np.asarray(fwi.iter_loss)
        np.savez(out_dir / "iter_vp.npz", data=iter_vp)
        np.savez(out_dir / "iter_loss.npz", data=iter_loss)
        vp_final = model.vp.detach().cpu().numpy()
        sc = model_scores(vp_true, vp_final)
        metrics = dict(
            tag=tag, arm=args.arm, refiner=args.refiner, robust=args.robust,
            start=args.start, band=args.band, f_eff=f_eff,
            device=device, iterations=iterations,
            iterations_done=int(done), complete=bool(complete),
            runtime_h=round(hours, 3), optimizer=args.optimizer,
            start_rung=args.start_rung, chunk=chunk,
            on_above=args.on_above, off_below=args.off_below,
            fixed_k=(args.fixed_k if args.arm == "fixedk" else None),
            ladder_stages=(list(LADDER_STAGES) if args.arm == "ladder" else None),
            ladder_thresholds=(thr if args.arm == "ladder" else None),
            ssim=sc["ssim"], mape=sc["mape"],
            rms_init=float(np.sqrt(((vp_init - vp_true) ** 2).mean())),
            rms_final=float(np.sqrt(((vp_final - vp_true) ** 2).mean())),
            trajectory=[dict(iter=i, skip=s, lam=l) for i, s, l in traj],
            handbacks=(ctrl.handbacks if ctrl else None),
            reentries=(ctrl.reentries if ctrl else None),
            final_stage=(ladder.stage if ladder else None),
        )
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        return vp_final, iter_loss, metrics

    # ---- the chunked loop: measure -> set lambda -> invert -> checkpoint -------
    t0 = time.time()
    done = 0
    hours = 0.0
    while done < iterations:
        skip = measure_skip()
        if ladder is not None:                        # N-stage ladder arm
            level = ladder.update(skip)
            loss_fn.set_stage(level)
            mode = loss_fn.active_name.upper()
            label = f"stage={level}"
        else:                                         # 2-term blend arms
            level = lam_for(done, skip)
            loss_fn.set_lambda(level)
            mode = args.robust.upper() if level >= 0.5 else args.refiner.upper()
            label = f"lambda={level:.0f}"
        traj.append((done, float(skip), float(level)))
        print(f"  iter {done:3d}: skip={skip:.3f} -> {label} ({mode})", flush=True)
        n = min(chunk, iterations - done)
        fwi.forward(iteration=n, start_iter=done,
                    batch_size=settings["batch_size"],
                    checkpoint_segments=settings["checkpoint_segments"],
                    cutoff_freq=args.band)
        done += n
        hours = (time.time() - t0) / 3600.0
        _save(done, hours, complete=(done >= iterations))

    skip_end = measure_skip()
    traj.append((done, float(skip_end),
                 float(ladder.stage if ladder is not None else loss_fn.lam)))
    vp_final, iter_loss, metrics = _save(done, hours, complete=True)
    print(f"skip@final: {skip_end:.3f}", flush=True)
    if ctrl:
        print(f"controller: handbacks={ctrl.handbacks} reentries={ctrl.reentries}"
              + ("  <-- REENTRIES>0: hand-over criterion suspect"
                 if ctrl.reentries else ""), flush=True)
    if ladder is not None:
        print(f"ladder: reached stage {ladder.stage}/{len(LADDER_STAGES) - 1} "
              f"({loss_fn.active_name})"
              + ("  <-- never left the robust stage" if ladder.stage == 0 else ""),
              flush=True)
    print(json.dumps({k: v for k, v in metrics.items() if k != "trajectory"},
                     indent=2), flush=True)

    # ---- figure: models + loss + the skip/lambda trajectory --------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    ext = [0, (NX - 1) * DX / 1000, (NZ - 1) * DZ / 1000, 0]
    for ax, (d, ttl) in zip(axes.flat[:2], [(vp_true, "true"),
                                            (vp_final, f"inverted ({tag})")]):
        im = ax.imshow(d, extent=ext, cmap="jet",
                       vmin=vp_true.min(), vmax=vp_true.max())
        ax.set(title=f"vp {ttl} [m/s]", xlabel="x [km]", ylabel="z [km]")
        fig.colorbar(im, ax=ax, shrink=0.8)
    axes.flat[2].plot(iter_loss, "k.-")
    axes.flat[2].set(title="loss (per-mode scale)", xlabel="iteration")
    it_, sk_, lm_ = zip(*traj)
    ax = axes.flat[3]
    ax.plot(it_, sk_, "b.-", label="skip fraction")
    if ladder is not None:
        for t in thr:
            ax.axhline(t, color="g", ls="--", lw=0.8)
        ax.step(it_, lm_, "k-", where="post", alpha=0.6, label="stage")
    else:
        ax.axhline(args.on_above, color="r", ls="--", lw=0.8, label="on_above")
        ax.axhline(args.off_below, color="g", ls="--", lw=0.8, label="off_below")
        ax.step(it_, lm_, "k-", where="post", alpha=0.6, label="lambda")
    ax.set(title="controller trajectory", xlabel="iteration",
           ylim=(-0.05, max(1.0, max(lm_)) + 0.05))
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "final.png", dpi=150)
    print("saved results to", out_dir, flush=True)


if __name__ == "__main__":
    main()
