"""One misfit x optimizer cell of the full-Marmousi2 DAS campaign.

    python hpc/marmousi_full_das/run_one.py --misfit gc --optimizer adam
        [--iterations 300] [--regularization none|tikhonov1|tikhonov2|tv1|tv2]
        [--device cuda|mps|cpu] [--smoke]

Requires generate_obs.py to have produced $DASFWI_RESULTS/obs_data_das.npz.
--smoke runs 2 iterations for wiring verification (results marked smoke_).

Everything follows Liu's ADFWI Marmousi2 examples verbatim except the
receiver side (DAS strain rate on vertical fibers); see common.py. Outputs
per combo into $DASFWI_RESULTS/<misfit>_<optimizer>[_<reg>]/:
    iter_vp.npz, iter_loss.npz     (Liu's convention)
    metrics.json                   (RMS/update-corr per iteration cache)
    final.png                      (true / init / inverted + loss curve)
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import (OUT_ROOT, OBS_FILE, ITERATIONS, NZ, NX, DX, DZ,
                    MISFITS, OPTIMIZERS, MISFIT_RUN_SETTINGS,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--misfit", required=True, choices=MISFITS)
    ap.add_argument("--optimizer", required=True, choices=sorted(OPTIMIZERS))
    ap.add_argument("--iterations", type=int, default=ITERATIONS)
    ap.add_argument("--regularization", default="none")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="2-iteration wiring check")
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = torch.float32
    iterations = 2 if args.smoke else args.iterations
    tag = f"{args.misfit}_{args.optimizer}"
    if args.regularization != "none":
        tag += f"_{args.regularization}"
    if args.smoke:
        tag = "smoke_" + tag
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {tag} on {device}, {iterations} iterations ===", flush=True)

    # setup (identical across combos)
    vp_true, vp_init = load_models()
    geometry = build_geometry()
    survey = build_survey(geometry)
    layer = DASObservationLayer(geometry,
                                output="strain_rate").to(dtype).to(device)

    obs_data = SeismicData(survey)
    obs_data.load(str(OUT_ROOT / OBS_FILE))

    model = build_model(vp_init,
                        vp_bound=[float(vp_true.min()), float(vp_true.max())],
                        vp_grad=True, device=device, dtype=dtype)
    prop = AcousticPropagator(model, survey, device=device, dtype=dtype)

    optimizer = OPTIMIZERS[args.optimizer](model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100,
                                                gamma=0.75, last_epoch=-1)
    loss_fn = build_misfit(args.misfit, iterations=iterations)
    reg_fn, wx, wz = build_regularization(args.regularization, device, dtype)
    settings = MISFIT_RUN_SETTINGS[args.misfit]

    fwi = AcousticFWI(propagator=prop, model=model,
                      optimizer=optimizer, scheduler=scheduler,
                      loss_fn=loss_fn, obs_data=obs_data,
                      gradient_processor=build_gradient_processor(),
                      regularization_fn=reg_fn,
                      regularization_weights_x=wx,
                      regularization_weights_z=wz,
                      waveform_normalize=settings["normalize"],
                      cache_result=True, cache_result_epoch=10,
                      save_fig_epoch=-1,
                      das_layer=layer, obs_key="strain_rate")

    t0 = time.time()
    fwi.forward(iteration=iterations,
                batch_size=settings["batch_size"],
                checkpoint_segments=settings["checkpoint_segments"])
    hours = (time.time() - t0) / 3600.0

    # save Liu-style outputs
    iter_vp = np.asarray(fwi.iter_vp)
    iter_loss = np.asarray(fwi.iter_loss)
    np.savez(out_dir / "iter_vp.npz", data=iter_vp)
    np.savez(out_dir / "iter_loss.npz", data=iter_loss)

    # metrics
    vp_final = model.vp.detach().cpu().numpy()
    d_true = vp_true - vp_init
    d_inv = vp_final - vp_init
    denom = np.sqrt((d_true ** 2).sum() * (d_inv ** 2).sum())
    from inversion.metrics import model_scores       # SSIM + MAPE (Liu's metrics)
    sc = model_scores(vp_true, vp_final)
    metrics = dict(
        tag=tag, device=device, iterations=iterations, runtime_h=round(hours, 3),
        rms_init=float(np.sqrt(((vp_init - vp_true) ** 2).mean())),
        rms_final=float(np.sqrt(((vp_final - vp_true) ** 2).mean())),
        update_corr=float((d_true * d_inv).sum() / denom) if denom > 0 else 0.0,
        ssim=sc["ssim"], mape=sc["mape"],
        loss_first=float(iter_loss[0]), loss_last=float(iter_loss[-1]),
        losses_finite=bool(np.isfinite(iter_loss).all()),
    )
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)

    # figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    ext = [0, (NX - 1) * DX / 1000, (NZ - 1) * DZ / 1000, 0]
    for ax, (d, ttl) in zip(axes.flat[:3], [(vp_true, "true"),
                                            (vp_init, "initial"),
                                            (vp_final, f"inverted ({tag})")]):
        im = ax.imshow(d, extent=ext, cmap="jet",
                       vmin=vp_true.min(), vmax=vp_true.max())
        ax.set(title=f"vp {ttl} [m/s]", xlabel="x [km]", ylabel="z [km]")
        fig.colorbar(im, ax=ax, shrink=0.8)
    axes.flat[3].plot(iter_loss, "k.-")
    axes.flat[3].set(title="loss", xlabel="iteration")
    fig.savefig(out_dir / "final.png", dpi=150)
    print("saved results to", out_dir, flush=True)


if __name__ == "__main__":
    main()
