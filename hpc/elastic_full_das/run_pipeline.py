"""PHASE 4: the integrated elastic pipeline - Route B start -> adaptive Vp/Vs FWI.

Runs the full proposal end to end on the elastic Marmousi2 section:

  1. START     a data-independent 1-D linear v(z), or the Route B starter
               produced by run_traveltime_starter.py (--start route_b), with
               Vs = Vp/sqrt(3) (Poisson-solid physics prior, NOT Castagna).
  2. CASCADE   low -> high frequency bands (zero-phase low-pass applied
               IDENTICALLY to synthetic and observed).
  3. OBJECTIVE lambda(f, stage) ramps L2 -> Wasserstein-Sinkhorn, with the two
               terms balanced by adjoint-source norm (see adaptive_misfit.py).
  4. STAGING   2-D schedule: band 1 inverts Vp ONLY; Vs is released from band 2
               with lambda forced high initially and annealed - because the
               sqrt(3) Vs seed can be cycle-skipped at the starting frequency in
               a sedimentary cover (~200 ms one-way S error vs T/2 = 167 ms).
               Vp-lead/Vs-follow with overlap stops Vp absorbing S kinematics.
  5. GUARDS    vs <= vp/1.5 Poisson clamp; water rows masked; illumination
               (diagonal-Hessian) preconditioning optional per Phase 0.

    python hpc/elastic_full_das/run_pipeline.py --smoke
    python hpc/elastic_full_das/run_pipeline.py --start route_b \
        --bands 2.0,3.0,4.5,full --iters 40 --flip-lo 3 --flip-hi 8

Density is held CONSTANT (a joint rho inversion diverged and dragged Vp/Vs down).
Set the lambda schedule's (f_lo, f_hi) from the PHASE 1 flip curve.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import (OUT_ROOT, OBS_FILE, NZ, NX, DX, DZ, DT, F0, NT, WATER_ROWS,
                    MIN_VP_VS, FD_ORDER, CHECKPOINT_SEGMENTS, CACHE_EVERY,
                    SCHEDULER, OPTIMIZERS, MISFIT_RUN_SETTINGS,
                    pick_device, load_models, build_model, build_acquisition,
                    build_misfit, normalize_traces, apply_misfit,
                    ElasticPropagator)

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ADFWI.fwi.multiScaleProcessing import lpass
from inversion.adaptive_misfit import (BlendedMisfit, LambdaSchedule, SkipSwitch,
                                       stage_plan, SKIP_ON_ABOVE, SKIP_OFF_BELOW)
from inversion.metrics import model_scores
from inversion.preconditioner import illumination_weight
from inversion.skip_diagnostic import skip_fraction, ricker_f90
from inversion.starting_model import linear_vz, vs_from_vp, poisson_clamp, SQRT3

DEFAULT_BANDS = "2.0,3.0,4.5,full"


def parse_bands(spec):
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip().lower()
        if tok:
            out.append(None if tok in ("full", "none", "0") else float(tok))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="linear", choices=("linear", "route_b", "smooth"),
                    help="linear = 1-D v(z); route_b = starter/vp_start.npz; "
                         "smooth = the 180 m smoothed truth (reference only)")
    ap.add_argument("--bands", default=DEFAULT_BANDS)
    ap.add_argument("--iters", type=int, default=50, help="iterations PER BAND")
    ap.add_argument("--optimizer", default="adam", choices=sorted(OPTIMIZERS))
    ap.add_argument("--lo", default="l2")
    # envelope, NOT sinkhorn: the Phase-1 gate refuted OT (sinkhorn never beats
    # L2 and craters with it under skip). weci is stateful and is rejected as a
    # blend term -- see adaptive_misfit._reject_stateful.
    ap.add_argument("--hi", default="envelope")
    ap.add_argument("--flip-lo", type=float, default=3.0, dest="flip_lo")
    ap.add_argument("--flip-hi", type=float, default=8.0, dest="flip_hi")
    ap.add_argument("--vs-release-band", type=int, default=2, dest="vs_release_band",
                    help="1-based band at which Vs joins the inversion")
    ap.add_argument("--vs-lambda-start", type=float, default=1.0, dest="vs_lambda_start",
                    help="lambda forced at Vs release (sqrt(3) seed may skip)")
    ap.add_argument("--vs-anneal-bands", type=int, default=1, dest="vs_anneal_bands")
    ap.add_argument("--timing", choices=("skip", "frequency"), default="skip",
                    help="how lambda is timed: 'skip' = the MEASURED cycle-skip "
                         "fraction (Phase-A validated); 'frequency' = the original "
                         "log-linear f-schedule (comparison arm)")
    ap.add_argument("--chunk", type=int, default=25,
                    help="skip timing: iterations per controller update")
    ap.add_argument("--on-above", type=float, default=SKIP_ON_ABOVE, dest="on_above")
    ap.add_argument("--off-below", type=float, default=SKIP_OFF_BELOW, dest="off_below")
    ap.add_argument("--precond", choices=["illum", "off"], default="off",
                    help="illumination preconditioning (Phase 0: helps sgd, wash for adam)")
    ap.add_argument("--vp-vs", type=float, default=SQRT3, dest="vp_vs")
    ap.add_argument("--fixed", default=None,
                    help="control arm: use this single misfit at every band")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="validate the plan (bands, Vs release, data, overwrite) "
                         "and exit -- no GPU, no data needed. RUN THIS FIRST.")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = pick_device(args.device)
    bands = parse_bands(args.bands)
    iters = 2 if args.smoke else args.iters
    adaptive = args.fixed is None
    f90 = ricker_f90(F0, DT, NT, integrated=True)
    # The tag MUST encode every knob that changes the experiment, or two runs
    # collide in one directory and corrupt each other. Learned twice: --timing
    # (skip vs frequency), then the schedule itself -- `--bands full
    # --vs-release-band 1 --iters 300` and `--bands full,full --iters 100` are
    # different experiments that previously produced the SAME tag.
    tag = args.tag or (f"pipeline_{args.start}_"
                       + (f"{args.timing}_{args.lo}-{args.hi}" if adaptive
                          else f"fixed_{args.fixed}")
                       + f"_{args.optimizer}"
                       + f"_b{len(bands)}x{iters}vs{args.vs_release_band}"
                       + ("_smoke" if args.smoke else ""))
    out_dir = OUT_ROOT / "pipeline" / tag
    print(f"=== ELASTIC PIPELINE {tag} on {device} ===\n"
          f"    bands={bands} x {iters} iters | source f90={f90:.2f} Hz | "
          f"Vs released at band {args.vs_release_band}", flush=True)

    # ---- PREFLIGHT: validate the plan BEFORE spending any GPU time ----------
    # These checks run on every launch (and alone under --dry-run, which needs
    # no data and no GPU, so the exact command can be verified locally first).
    problems, notes = [], []
    plan_preview = stage_plan(bands, None, f90,
                              vs_release_band=args.vs_release_band)
    if not any(p["vs_live"] for p in plan_preview):
        problems.append(
            f"Vs IS NEVER INVERTED: {len(bands)} band(s) but --vs-release-band="
            f"{args.vs_release_band}. This is an ELASTIC run that would only "
            f"update Vp, at full elastic cost. Use --vs-release-band <= "
            f"{len(bands)}, or add a stage (e.g. --bands full,full for a "
            f"single-scale Vp-lead/Vs-follow run).")
    f_effs = [p["f_eff"] for p in plan_preview]
    if len(set(f_effs)) != len(f_effs):
        notes.append(f"duplicate effective bands {f_effs} (cut-offs above the "
                     f"source f90={f90:.2f} Hz all clamp to it) -- those stages "
                     "invert identical data")
    obs_path = OUT_ROOT / OBS_FILE
    if not obs_path.is_file():
        if args.dry_run:
            notes.append(f"no observed data at {obs_path} (fine for --dry-run; "
                         "run genobs_elastic before the real job)")
        else:
            problems.append(f"no observed data at {obs_path} -- run generate_obs "
                            "(kind=genobs_elastic) first")
    if args.start == "route_b":
        f = OUT_ROOT / "starter" / "vp_start.npz"
        if not f.is_file():
            problems.append(f"--start route_b but no starter at {f} -- run "
                            "hpc/elastic_full_das/run_traveltime_starter.py")
        else:
            # The Route B starter is built on the ACOUSTIC grid (88x200 @ 40 m);
            # this driver runs the ELASTIC grid (78x200 @ 45 m). Loading one into
            # the other is a shape error -- never caught before because route_b
            # had never been run. Fail here, not 4 hours into a GPU job.
            try:
                shp = tuple(np.load(f)["vp_start"].shape)
            except Exception as e:                        # noqa: BLE001
                shp = None
                problems.append(f"cannot read {f}: {type(e).__name__}: {e}")
            if shp is not None and shp != (NZ, NX):
                problems.append(
                    f"route_b starter is {shp} but this grid is {(NZ, NX)} -- "
                    "rebuild it with hpc/elastic_full_das/run_traveltime_starter.py "
                    "(the ACOUSTIC starter is 88x200 @ 40 m and cannot be used here).")
    if out_dir.exists() and (out_dir / "metrics.json").is_file():
        notes.append(f"{out_dir} already has results -- this run OVERWRITES them")
    for n in notes:
        print(f"    NOTE: {n}", flush=True)
    for p in problems:
        print(f"    *** {p}", flush=True)
    print("    plan: " + " | ".join(
        f"band {p['band']} {'full' if p['cutoff'] is None else str(p['cutoff'])+'Hz'}"
        f" f_eff={p['f_eff']:.2f} {'Vp+Vs' if p['vs_live'] else 'Vp only'}"
        f" x{iters}it" for p in plan_preview), flush=True)
    if problems:
        raise SystemExit("preflight FAILED (see *** above) -- nothing was run")
    if args.dry_run:
        print("    dry-run OK: plan is valid, exiting before any computation")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- models ------------------------------------------------------------
    vp_true, vs_true, vp_smooth, vs_smooth = load_models()
    if args.start == "smooth":
        vp_init, vs_init = vp_smooth, vs_smooth
    elif args.start == "route_b":
        f = OUT_ROOT / "starter" / "vp_start.npz"
        if not f.is_file():
            raise SystemExit(f"no Route B starter at {f}; run "
                             "hpc/elastic_full_das/run_traveltime_starter.py first")
        d = np.load(f)
        vp_init = np.asarray(d["vp_start"], np.float64)
        vs_init = vs_from_vp(vp_init, ratio=args.vp_vs)
        print(f"    Route B starter loaded from {f}", flush=True)
    else:
        vp_init = linear_vz(NZ, NX, 1500.0, 4000.0, water_rows=WATER_ROWS,
                            v_water=1500.0)
        vs_init = vs_from_vp(vp_init, ratio=args.vp_vs)
    vs_init = poisson_clamp(vp_init, vs_init, MIN_VP_VS)
    vp_init[:WATER_ROWS] = vp_true[:WATER_ROWS]          # water pinned, as in the campaign
    vs_init[:WATER_ROWS] = vs_true[:WATER_ROWS]

    survey, layer, _geom = build_acquisition(device)
    n_shots = survey.source.num
    obs = torch.from_numpy(np.load(OUT_ROOT / OBS_FILE)["strain_rate"]).float()

    bounds = ([float(vp_true.min()), float(vp_true.max())],
              [float(vs_true.min()), float(vs_true.max())])
    model = build_model(vp_init, vs_init, bounds, grad=True, device=device)
    prop = ElasticPropagator(model, survey, device=device, dtype=torch.float32)

    total_iters = iters * len(bands)
    if adaptive:
        loss_fn = BlendedMisfit(build_misfit(args.lo, iterations=total_iters),
                                build_misfit(args.hi, iterations=total_iters), lam=0.0)
        settings = MISFIT_RUN_SETTINGS[args.hi]
        schedule = (None if args.timing == "skip" else
                    LambdaSchedule(args.flip_lo, args.flip_hi,
                                   stage_overrides={"vs": args.vs_lambda_start},
                                   stage_anneal=args.vs_anneal_bands))
    else:
        loss_fn = build_misfit(args.fixed, iterations=total_iters)
        settings = MISFIT_RUN_SETTINGS[args.fixed]
        schedule = None
    batch = settings["batch_size"] or n_shots

    grad_mask = torch.ones((NZ, NX), device=device)
    grad_mask[:WATER_ROWS, :] = 0

    def measure_skip(f_eff):
        with torch.no_grad():
            rec = prop.forward(model=model, fd_order=FD_ORDER,
                               checkpoint_segments=CHECKPOINT_SEGMENTS)
            syn = layer(rec["vx"], rec["vz"]).cpu()
        return skip_fraction(syn, obs, DT, f_eff)

    # ---- the 2-D cascade: bands x parameter stages -------------------------
    plan = stage_plan(bands, schedule, f90, vs_release_band=args.vs_release_band)
    losses, iter_vp, iter_vs, band_log = [], [], [], []
    t0 = time.time()
    it_global = 0
    total_iters = iters * len(plan)
    CKPT_EVERY = 25

    def _save(done, hours, complete):
        """Write the full result set. Called every CKPT_EVERY iterations and at
        each band boundary, so a walltime kill still leaves the latest Vp/Vs,
        curves and a metrics.json flagged complete=false."""
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "iter_vp.npz", data=np.asarray(iter_vp))
        np.savez(out_dir / "iter_vs.npz", data=np.asarray(iter_vs))
        np.savez(out_dir / "iter_loss.npz", data=np.asarray(losses))
        vp_f = model.vp.detach().cpu().numpy()
        vs_f = model.vs.detach().cpu().numpy()
        m = dict(tag=tag, start=args.start, adaptive=adaptive,
                 fixed=args.fixed, optimizer=args.optimizer, timing=args.timing,
                 bands=[("full" if b is None else b) for b in bands],
                 iters_per_band=iters, vs_release_band=args.vs_release_band,
                 iterations=total_iters, iterations_done=int(done),
                 complete=bool(complete),
                 precond=args.precond, device=device, runtime_h=round(hours, 3),
                 losses_finite=bool(np.isfinite(losses).all()) if losses else None,
                 band_log=band_log)
        for nm, tru, ini, fin in (("vp", vp_true, vp_init, vp_f),
                                  ("vs", vs_true, vs_init, vs_f)):
            sc = model_scores(tru, fin)
            sc0 = model_scores(tru, ini)
            m[f"ssim_{nm}"], m[f"mape_{nm}"] = sc["ssim"], sc["mape"]
            m[f"ssim_init_{nm}"] = sc0["ssim"]
            m[f"rms_init_{nm}"] = float(np.sqrt(((ini - tru) ** 2).mean()))
            m[f"rms_final_{nm}"] = float(np.sqrt(((fin - tru) ** 2).mean()))
        (out_dir / "metrics.json").write_text(json.dumps(m, indent=2, default=str))
        return vp_f, vs_f, m
    for step in plan:
        bi, f_band, f_eff = step["band"], step["cutoff"], step["f_eff"]
        vs_live, stage, lam = step["vs_live"], step["stage"], step["lam"]
        # lam is None under --timing skip (stage_plan gets schedule=None, because
        # the SkipSwitch sets lambda per chunk inside the loop below). Only the
        # frequency-scheduled path has a band-level lambda to apply here.
        if adaptive and lam is not None:
            loss_fn.set_lambda(lam)

        params = [model.vp, model.vs] if vs_live else [model.vp]
        model.vs.requires_grad_(bool(vs_live))
        optimizer = OPTIMIZERS[args.optimizer](params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **SCHEDULER)

        def _skip():
            try:
                return measure_skip(f_eff)["skip_fraction"]
            except Exception:                                 # noqa: BLE001
                return float("nan")

        sk = _skip()
        # FRESH controller per band (a higher band raises f_max, so skip jumps at
        # the boundary and the controller must be free to re-enter robust mode).
        skip_timing = adaptive and args.timing == "skip"
        # max_robust = the band's controller-update budget, so the stall guard
        # can tell whether robust can still reach off_below in the time left.
        _updates = max(1, -(-iters // max(1, args.chunk)))
        ctrl = SkipSwitch(on_above=args.on_above, off_below=args.off_below,
                          max_robust=_updates) if skip_timing else None
        chunk = args.chunk if skip_timing else iters
        traj = []
        print(f"--- band {bi}/{len(bands)} cutoff="
              f"{'full' if f_band is None else f'{f_band} Hz'} | stage="
              f"{'Vp+Vs' if vs_live else 'Vp only'} | lambda="
              f"{'skip-driven' if skip_timing else ('-' if lam is None else f'{lam:.3f}')}"
              f" | skip@start={sk:.3f} ---", flush=True)

        for it in range(iters):
            if skip_timing and it % chunk == 0:
                sk_now = sk if it == 0 else _skip()
                lam = ctrl.update(sk_now)
                # Physics floor at Vs release: the sqrt(3) Vs seed can be
                # cycle-skipped in a sedimentary cover even at 3 Hz (S delay
                # ~200 ms > T/2 = 167 ms), so start that band robust regardless
                # of what the Vp-dominated skip measurement says.
                if vs_live and bi == args.vs_release_band and it == 0:
                    lam = max(lam, args.vs_lambda_start)
                loss_fn.set_lambda(lam)
                traj.append(dict(iter=it, skip=float(sk_now), lam=float(lam)))
                mode = args.hi.upper() if lam >= 0.5 else args.lo.upper()
                print(f"    iter {it:3d}: skip={sk_now:.3f} -> lambda={lam:.0f}"
                      f" ({mode})", flush=True)
            optimizer.zero_grad()
            loss_iter, illum = 0.0, None
            for b0 in range(0, n_shots, batch):
                shot_index = np.arange(b0, min(b0 + batch, n_shots))
                rec = prop.forward(model=model, shot_index=shot_index,
                                   fd_order=FD_ORDER,
                                   checkpoint_segments=CHECKPOINT_SEGMENTS)
                syn = layer(rec["vx"], rec["vz"]).cpu()
                o = obs[shot_index]
                if f_band is not None:                    # multiscale: SAME filter
                    syn, o = lpass(syn, o, f_band, int(round(1.0 / DT)))
                if settings["normalize"]:
                    syn, o = normalize_traces(syn), normalize_traces(o)
                loss = apply_misfit(loss_fn, syn, o)
                loss.backward()
                loss_iter += float(loss)
                if args.precond == "illum":
                    fw = (rec["forward_wavefield_vx"] + rec["forward_wavefield_vz"]).detach()
                    illum = fw if illum is None else illum + fw
            with torch.no_grad():
                w = (illumination_weight(illum) if args.precond == "illum"
                     and illum is not None else None)
                for par in params:
                    if par.grad is None:
                        continue
                    par.grad *= grad_mask
                    if w is not None:
                        par.grad *= w
                    if args.optimizer == "sgd":
                        peak = par.grad.abs().max().clamp_min(1e-30)
                        par.grad *= float(par.detach().max()) / peak
            optimizer.step()
            scheduler.step()
            model.forward()
            with torch.no_grad():                          # Poisson stability
                model.vs.data = torch.minimum(model.vs.data, model.vp.data / MIN_VP_VS)
            losses.append(loss_iter)
            if it_global % CACHE_EVERY == 0:
                iter_vp.append(model.vp.detach().cpu().numpy().copy())
                iter_vs.append(model.vs.detach().cpu().numpy().copy())
            it_global += 1
            print(f"iter {it_global}: loss {loss_iter:.6f} "
                  f"({(time.time()-t0)/it_global:.0f}s/iter)", flush=True)
            # Elastic pipeline cells are the longest-running jobs in the project
            # (Vp+Vs, elastic propagator, several bands), so write the FULL result
            # set periodically -- otherwise a walltime kill leaves nothing at all.
            if it_global % CKPT_EVERY == 0:
                _save(it_global, (time.time() - t0) / 3600.0, complete=False)
                print(f"  checkpoint {it_global}/{total_iters}", flush=True)
        band_log.append(dict(band=bi, cutoff=f_band, f_eff=f_eff, stage=stage,
                             lam=lam, skip_at_start=sk, timing=args.timing,
                             trajectory=traj,
                             handbacks=(ctrl.handbacks if ctrl else None),
                             reentries=(ctrl.reentries if ctrl else None)))
        _save(it_global, (time.time() - t0) / 3600.0,          # end of each band
              complete=(bi >= len(plan)))

    hours = (time.time() - t0) / 3600.0
    iter_vp.append(model.vp.detach().cpu().numpy().copy())
    iter_vs.append(model.vs.detach().cpu().numpy().copy())
    vp_final, vs_final, metrics = _save(it_global, hours, complete=True)
    print(json.dumps({k: v for k, v in metrics.items() if k != "band_log"},
                     indent=2, default=str), flush=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), constrained_layout=True)
    ext = [0, (NX - 1) * DX / 1000, (NZ - 1) * DZ / 1000, 0]
    for r, (nm, tru, ini, fin) in enumerate((("vp", vp_true, vp_init, vp_final),
                                             ("vs", vs_true, vs_init, vs_final))):
        for c, (d, ttl) in enumerate([(tru, "true"), (ini, f"start ({args.start})"),
                                      (fin, "inverted")]):
            im = axes[r, c].imshow(d, extent=ext, cmap="jet",
                                   vmin=tru.min(), vmax=tru.max())
            axes[r, c].set(title=f"{nm} {ttl} [m/s]", xlabel="x [km]", ylabel="z [km]")
            fig.colorbar(im, ax=axes[r, c], shrink=0.8)
    fig.savefig(out_dir / "final.png", dpi=150)
    print("saved pipeline results to", out_dir, flush=True)


if __name__ == "__main__":
    main()
