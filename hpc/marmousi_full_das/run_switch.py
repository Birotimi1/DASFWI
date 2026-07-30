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
from inversion.adaptive_misfit import (BlendedMisfit, SkipSwitch,
                                       StagedMisfit, StageLadder,
                                       SKIP_ON_ABOVE, SKIP_OFF_BELOW,
                                       LADDER_THRESHOLDS)

ARMS = ("switch", "fixedk", "ladder", "l2", "envelope")
LADDER_STAGES = ("envelope", "gc", "l2")     # robust -> gentle -> sharp refiner


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
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true", help="2 chunks of 2 iters")
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = torch.float32
    iterations = 4 if args.smoke else args.iterations
    chunk = 2 if args.smoke else max(1, args.chunk)
    # encode the pair in the tag unless it is the default envelope->l2, so the
    # envelope->gc variant cannot overwrite the proposal's results
    pair = "" if args.refiner == "l2" else f"-{args.refiner}"
    tag = f"{args.arm}{pair}_{args.optimizer}"
    if args.smoke:
        tag = "smoke_" + tag
    out_dir = OUT_ROOT / f"switch_{args.start_rung}" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== PHASE A {tag} [start={args.start_rung}] on {device}, "
          f"{iterations} iters in chunks of {chunk} ===", flush=True)

    # ---- setup: identical to the gate's run_one.py -----------------------------
    vp_true, vp_init = load_models(args.start_rung)
    geometry = build_geometry()
    survey = build_survey(geometry)
    layer = DASObservationLayer(geometry, output="strain_rate").to(dtype).to(device)
    obs_data = SeismicData(survey)
    obs_data.load(str(OUT_ROOT / OBS_FILE))
    obs_arr = torch.as_tensor(obs_data.data["strain_rate"])
    f90 = ricker_f90(F0, DT, NT, integrated=True)

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

    fwi = AcousticFWI(propagator=prop, model=model, optimizer=optimizer,
                      scheduler=scheduler, loss_fn=loss_fn, obs_data=obs_data,
                      gradient_processor=build_gradient_processor(),
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
        return skip_fraction(syn, obs_arr, DT, f90)["skip_fraction"]

    ctrl = SkipSwitch(on_above=args.on_above, off_below=args.off_below,
                      dwell=args.dwell) if args.arm == "switch" else None

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
                    checkpoint_segments=settings["checkpoint_segments"])
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
