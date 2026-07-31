"""ROUTE B on the ELASTIC grid: a picking-free, data-driven starting model.

Why this exists: the elastic runs have until now started from the 180 m SMOOTHED
TRUTH, which leaks the answer and is not deployable, or from a 1-D linear ramp,
which is outside the basin of attraction (measured: SSIM 0.361 -> 0.276 with the
switch, 0.197 with plain L2 -- BOTH degrade the start). A traveltime starting
model is in-basin AND transferable, and it is what the field workflow uses.

Why a SEPARATE script from the acoustic one: the two campaigns are different
grids -- acoustic 88x200 @ 40 m, elastic 78x200 @ 45 m. The acoustic Route B
starter cannot be loaded into the elastic model (shape mismatch, which the
run_pipeline preflight now rejects). This builds the starter on the ELASTIC grid
with the ELASTIC acquisition, so run_pipeline --start route_b can use it.

METHOD (no picking anywhere):
  1-D linear v(z)  ->  FWI with the cross-correlation TRAVELTIME misfit at a low
  band, Vp only, with Tikhonov-2 smoothing  ->  smooth  ->  Vs = Vp/sqrt(3).

Contrast with the literature: Park et al. (2025) at Utah FORGE MANUALLY PICK
first arrivals and invert them with an eikonal solver. `forge/traveltime_
tomography.py` auto-picks with STA/LTA. This uses no picks at all -- the
traveltime misfit measures arrival-time error by cross-correlating the full
waveforms, so nothing has to be detected or labelled. That is the property the
project mandate needs: deployable at a new site with no manual intervention.

    python hpc/elastic_full_das/run_traveltime_starter.py --iters 60 --band 2.0
    python hpc/elastic_full_das/run_traveltime_starter.py --dry-run
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import (OUT_ROOT, OBS_FILE, NZ, NX, DX, DZ, DT, F0, NT, WATER_ROWS,
                    MIN_VP_VS, FD_ORDER, CHECKPOINT_SEGMENTS, SCHEDULER,
                    OPTIMIZERS, MISFIT_RUN_SETTINGS,
                    pick_device, load_models, build_model, build_acquisition,
                    build_misfit, normalize_traces, apply_misfit,
                    ElasticPropagator)

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ADFWI.fwi.multiScaleProcessing import lpass
from inversion.metrics import model_scores
from inversion.skip_diagnostic import skip_fraction, skip_vs_band, ricker_f90
from inversion.starting_model import (linear_vz, vs_from_vp, smooth_model,
                                      poisson_clamp, SQRT3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--band", type=float, default=2.0,
                    help="low-pass cut-off for the traveltime leg (Hz)")
    ap.add_argument("--optimizer", default="adam", choices=sorted(OPTIMIZERS))
    ap.add_argument("--misfit", default="traveltime",
                    help="kinematic misfit; 'traveltime' is picking-free")
    ap.add_argument("--smooth-sigma", type=float, default=4.0, dest="smooth_sigma",
                    help="Gaussian smoothing of the delivered starter (cells)")
    ap.add_argument("--v-top", type=float, default=1500.0, dest="v_top")
    ap.add_argument("--v-bottom", type=float, default=4000.0, dest="v_bottom")
    ap.add_argument("--vp-vs", type=float, default=SQRT3, dest="vp_vs")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = pick_device(args.device)
    iters = 2 if args.smoke else args.iters
    out_dir = OUT_ROOT / "starter"
    f90 = ricker_f90(F0, DT, NT, integrated=True)
    band = min(args.band, f90)
    print(f"=== ELASTIC Route B starter on {device} | {iters} iters @ "
          f"{band:.2f} Hz (source f90={f90:.2f}) | grid {NZ}x{NX} @ {DX:g} m ===",
          flush=True)

    problems = []
    if not (OUT_ROOT / OBS_FILE).is_file():
        msg = f"no observed data at {OUT_ROOT / OBS_FILE} (run genobs_elastic)"
        (print(f"    NOTE: {msg} (fine for --dry-run)") if args.dry_run
         else problems.append(msg))
    for p in problems:
        print(f"    *** {p}", flush=True)
    if problems:
        raise SystemExit("preflight FAILED -- nothing was run")
    if args.dry_run:
        print(f"    dry-run OK: would write {out_dir}/vp_start.npz "
              f"shaped {(NZ, NX)} for run_pipeline --start route_b")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- truth is used ONLY to score the starter, never to build it ----------
    vp_true, vs_true, _vp_smooth, _vs_smooth = load_models()
    survey, layer, _geom = build_acquisition(device)
    n_shots = survey.source.num
    obs = torch.from_numpy(np.load(OUT_ROOT / OBS_FILE)["strain_rate"]).float()

    # ---- the ONLY input: a blind 1-D ramp (no truth, no logs, no picks) ------
    vp_1d = linear_vz(NZ, NX, args.v_top, args.v_bottom,
                      water_rows=WATER_ROWS, v_water=args.v_top)
    vs_1d = poisson_clamp(vp_1d, vs_from_vp(vp_1d, ratio=args.vp_vs), MIN_VP_VS)

    bounds = ([float(vp_true.min()), float(vp_true.max())],
              [float(vs_true.min()), float(vs_true.max())])
    model = build_model(vp_1d, vs_1d, bounds, grad=True, device=device)
    model.vs.requires_grad_(False)              # Vp only: traveltime is P-driven
    prop = ElasticPropagator(model, survey, device=device, dtype=torch.float32)
    optimizer = OPTIMIZERS[args.optimizer]([model.vp])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **SCHEDULER)
    misfit = build_misfit(args.misfit, iters)
    settings = MISFIT_RUN_SETTINGS[args.misfit]
    batch = settings["batch_size"] or n_shots
    grad_mask = torch.ones((NZ, NX), device=device)
    grad_mask[:WATER_ROWS, :] = 0

    def measure_skip(vp_np):
        with torch.no_grad():
            rec = prop.forward(model=model, fd_order=FD_ORDER,
                               checkpoint_segments=CHECKPOINT_SEGMENTS)
            syn = layer(rec["vx"], rec["vz"]).cpu()
        return skip_fraction(syn, obs, DT, f90)["skip_fraction"]

    try:
        skip_1d = measure_skip(vp_1d)
        print(f"skip@1-D start: {skip_1d:.3f}", flush=True)
    except Exception as e:                                   # noqa: BLE001
        print(f"skip@1-D failed: {type(e).__name__}: {e}", flush=True)
        skip_1d = None

    # ---- the traveltime leg --------------------------------------------------
    losses, t0 = [], time.time()
    for it in range(iters):
        optimizer.zero_grad()
        loss_iter = 0.0
        for b0 in range(0, n_shots, batch):
            shot_index = np.arange(b0, min(b0 + batch, n_shots))
            rec = prop.forward(model=model, shot_index=shot_index,
                               fd_order=FD_ORDER,
                               checkpoint_segments=CHECKPOINT_SEGMENTS)
            syn = layer(rec["vx"], rec["vz"]).cpu()
            o = obs[shot_index]
            syn, o = lpass(syn, o, band, int(round(1.0 / DT)))   # SAME filter
            if settings["normalize"]:
                syn, o = normalize_traces(syn), normalize_traces(o)
            loss = apply_misfit(misfit, syn, o)
            loss.backward()
            loss_iter += float(loss)
        with torch.no_grad():
            model.vp.grad *= grad_mask
        optimizer.step()
        scheduler.step()
        model.forward()
        losses.append(loss_iter)
        print(f"iter {it}: loss {loss_iter:.6f} "
              f"({(time.time()-t0)/(it+1):.0f}s/iter)", flush=True)
        if (it + 1) % 10 == 0:                               # checkpoint
            np.savez(out_dir / "iter_loss.npz", data=np.asarray(losses))
            np.savez(out_dir / "vp_partial.npz",
                     vp=model.vp.detach().cpu().numpy(), iterations_done=it + 1)
    hours = (time.time() - t0) / 3600.0

    # ---- deliver: smooth, pin the water layer, seed Vs -----------------------
    vp_raw = model.vp.detach().cpu().numpy()
    vp_start = (smooth_model(vp_raw, args.smooth_sigma)
                if args.smooth_sigma > 0 else vp_raw)
    vp_start[:WATER_ROWS] = vp_1d[:WATER_ROWS]
    vs_start = poisson_clamp(vp_start, vs_from_vp(vp_start, ratio=args.vp_vs),
                             MIN_VP_VS)

    with torch.no_grad():                       # score the DELIVERED starter
        model.vp.data = torch.as_tensor(vp_start, dtype=model.vp.dtype,
                                        device=model.vp.device)
    try:
        skip_starter = measure_skip(vp_start)
        print(f"skip@starter: {skip_starter:.3f}", flush=True)
    except Exception as e:                                   # noqa: BLE001
        print(f"skip@starter failed: {type(e).__name__}: {e}", flush=True)
        skip_starter = None

    np.savez(out_dir / "vp_start.npz", vp_start=vp_start, vs_start=vs_start,
             nz=NZ, nx=NX, dx=DX, dz=DZ)
    np.savez(out_dir / "iter_loss.npz", data=np.asarray(losses))
    meta = dict(grid=[NZ, NX], dx=DX, dz=DZ, band=band, iterations=iters,
                misfit=args.misfit, optimizer=args.optimizer,
                smooth_sigma=args.smooth_sigma, vp_vs=args.vp_vs,
                runtime_h=round(hours, 3), complete=True,
                skip_1d=skip_1d, skip_starter=skip_starter,
                ssim_1d_vp=model_scores(vp_true, vp_1d)["ssim"],
                ssim_starter_vp=model_scores(vp_true, vp_start)["ssim"],
                ssim_starter_vs=model_scores(vs_true, vs_start)["ssim"],
                loss_first=float(losses[0]), loss_last=float(losses[-1]))
    (out_dir / "starter_metrics.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    print("ACCEPTANCE: the starter must beat the 1-D ramp on BOTH skip and SSIM;"
          " a starter that is no better than the ramp is not worth using.",
          flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), constrained_layout=True)
    ext = [0, (NX - 1) * DX / 1000, (NZ - 1) * DZ / 1000, 0]
    for ax, (d, t) in zip(axes, [(vp_true, "true vp"), (vp_1d, "1-D ramp"),
                                 (vp_start, "Route B starter")]):
        im = ax.imshow(d, extent=ext, cmap="jet",
                       vmin=vp_true.min(), vmax=vp_true.max())
        ax.set(title=t, xlabel="x [km]", ylabel="z [km]")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(out_dir / "starter.png", dpi=150)
    print("saved starter to", out_dir, flush=True)


if __name__ == "__main__":
    main()
