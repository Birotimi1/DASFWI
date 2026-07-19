"""One misfit x optimizer cell of the full-Marmousi2 ELASTIC 3-parameter DAS
campaign (joint Vp + Vs + density).

    python hpc/elastic_full_das/run_one.py --misfit gc --optimizer adam
        [--iterations 300] [--device cuda|mps|cpu] [--smoke]

Requires generate_obs.py to have produced $DASFWI_RESULTS/obs_data_das.npz.
ADFWI's ElasticFWI has no DAS path, so the inversion loop lives here (Liu's
structure + the vs<=vp/1.5 Poisson clamp). Outputs per combo into
$DASFWI_RESULTS/<misfit>_<optimizer>/:
    iter_vp.npz, iter_vs.npz, iter_rho.npz, iter_loss.npz
    metrics.json   (rms_init/final + update_corr for vp, vs, rho)
    final.png      (3x3: vp/vs/rho x true/init/inverted)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import (OUT_ROOT, OBS_FILE, ITERATIONS, NZ, NX, DX, DZ, WATER_ROWS,
                    MIN_VP_VS, FD_ORDER, CHECKPOINT_SEGMENTS, CACHE_EVERY,
                    SCHEDULER, MISFITS, OPTIMIZERS, MISFIT_RUN_SETTINGS,
                    pick_device, load_models, build_model, build_acquisition,
                    build_misfit, normalize_traces, apply_misfit,
                    ElasticPropagator)

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--misfit", required=True, choices=MISFITS)
    ap.add_argument("--optimizer", required=True, choices=sorted(OPTIMIZERS))
    ap.add_argument("--iterations", type=int, default=ITERATIONS)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="2-iteration wiring check")
    args = ap.parse_args()

    device = pick_device(args.device)
    iterations = 2 if args.smoke else args.iterations
    tag = f"{args.misfit}_{args.optimizer}"
    if args.smoke:
        tag = "smoke_" + tag
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== elastic {tag} on {device}, {iterations} iterations ===",
          flush=True)

    vp_true, vs_true, rho_true, vp_init, vs_init, rho_init = load_models()
    survey, layer, _geom = build_acquisition(device)
    n_shots = survey.source.num
    settings = MISFIT_RUN_SETTINGS[args.misfit]
    batch = settings["batch_size"] or n_shots

    # shared observed data (generate_obs.py wrote it once)
    obs = torch.from_numpy(np.load(OUT_ROOT / OBS_FILE)["strain_rate"]).float()
    print(f"observed {tuple(obs.shape)}, max|.| {float(obs.abs().max()):.3e}",
          flush=True)

    bounds = ([float(vp_true.min()), float(vp_true.max())],
              [float(vs_true.min()), float(vs_true.max())],
              [float(rho_true.min()), float(rho_true.max())])
    model = build_model(vp_init, vs_init, rho_init, bounds, grad=True,
                        device=device)
    prop = ElasticPropagator(model, survey, device=device, dtype=torch.float32)
    optimizer = OPTIMIZERS[args.optimizer]([model.vp, model.vs, model.rho])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **SCHEDULER)
    misfit = build_misfit(args.misfit, iterations)

    grad_mask = torch.ones((NZ, NX), device=device)
    grad_mask[:WATER_ROWS, :] = 0

    losses, iter_vp, iter_vs, iter_rho = [], [], [], []
    t0 = time.time()
    for it in range(iterations):
        optimizer.zero_grad()
        loss_iter = 0.0
        for b0 in range(0, n_shots, batch):
            shot_index = np.arange(b0, min(b0 + batch, n_shots))
            rec = prop.forward(model=model, shot_index=shot_index,
                               fd_order=FD_ORDER,
                               checkpoint_segments=CHECKPOINT_SEGMENTS)
            syn = layer(rec["vx"], rec["vz"]).cpu()
            o = obs[shot_index]
            if settings["normalize"]:
                syn, o = normalize_traces(syn), normalize_traces(o)
            loss = apply_misfit(misfit, syn, o)
            loss.backward()
            loss_iter += float(loss)
        with torch.no_grad():
            for par in (model.vp, model.vs, model.rho):
                par.grad *= grad_mask                          # Liu's mask
                if args.optimizer == "sgd":                    # norm_grad
                    peak = par.grad.abs().max().clamp_min(1e-30)
                    par.grad *= float(par.detach().max()) / peak
        optimizer.step()
        scheduler.step()
        model.forward()                                        # clip to bounds
        with torch.no_grad():                                  # Poisson clamp
            model.vs.data = torch.minimum(model.vs.data,
                                          model.vp.data / MIN_VP_VS)
        losses.append(loss_iter)
        if it % CACHE_EVERY == 0 or it == iterations - 1:
            iter_vp.append(model.vp.detach().cpu().numpy().copy())
            iter_vs.append(model.vs.detach().cpu().numpy().copy())
            iter_rho.append(model.rho.detach().cpu().numpy().copy())
        print(f"iter {it}: loss {loss_iter:.6f} "
              f"({(time.time()-t0)/(it+1):.0f}s/iter)", flush=True)

    hours = (time.time() - t0) / 3600.0

    np.savez(out_dir / "iter_vp.npz", data=np.asarray(iter_vp))
    np.savez(out_dir / "iter_vs.npz", data=np.asarray(iter_vs))
    np.savez(out_dir / "iter_rho.npz", data=np.asarray(iter_rho))
    np.savez(out_dir / "iter_loss.npz", data=np.asarray(losses))
    vp_final = model.vp.detach().cpu().numpy()
    vs_final = model.vs.detach().cpu().numpy()
    rho_final = model.rho.detach().cpu().numpy()

    metrics = dict(tag=tag, device=device, iterations=iterations,
                   runtime_h=round(hours, 3),
                   loss_first=float(losses[0]), loss_last=float(losses[-1]),
                   losses_finite=bool(np.isfinite(losses).all()))
    triplet = (("vp", vp_true, vp_init, vp_final),
               ("vs", vs_true, vs_init, vs_final),
               ("rho", rho_true, rho_init, rho_final))
    for nm, tru, ini, fin in triplet:
        dt_, di = tru - ini, fin - ini
        denom = np.sqrt((dt_ ** 2).sum() * (di ** 2).sum())
        metrics[f"rms_init_{nm}"] = float(np.sqrt(((ini - tru) ** 2).mean()))
        metrics[f"rms_final_{nm}"] = float(np.sqrt(((fin - tru) ** 2).mean()))
        metrics[f"update_corr_{nm}"] = (float((dt_ * di).sum() / denom)
                                        if denom else 0.0)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)

    fig, axes = plt.subplots(3, 3, figsize=(18, 12), constrained_layout=True)
    ext = [0, (NX - 1) * DX / 1000, (NZ - 1) * DZ / 1000, 0]
    units = {"vp": "m/s", "vs": "m/s", "rho": "kg/m^3"}
    for r, (nm, tru, ini, fin) in enumerate(triplet):
        for c, (d, ttl) in enumerate([(tru, "true"), (ini, "initial"),
                                      (fin, f"inverted ({tag})")]):
            im = axes[r, c].imshow(d, extent=ext, cmap="jet",
                                   vmin=tru.min(), vmax=tru.max())
            axes[r, c].set(title=f"{nm} {ttl} [{units[nm]}]", xlabel="x [km]",
                           ylabel="z [km]")
            fig.colorbar(im, ax=axes[r, c], shrink=0.8)
    fig.savefig(out_dir / "final.png", dpi=150)
    print("saved results to", out_dir, flush=True)


if __name__ == "__main__":
    main()
