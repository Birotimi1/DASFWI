"""PHASE 3: ROUTE B - build a transferable starting model from the data alone.

Starts from a DATA-INDEPENDENT 1-D linear v(z) and runs the cross-correlation
traveltime misfit at the lowest frequency band with heavy smoothing, producing a
long-wavelength Vp that is kinematically consistent with the observed DAS strain
rate. No first-break picking anywhere (the fragile step on DAS), no velocity
conversion, and no region-calibrated prior - so the same procedure transfers to
any survey. See inversion/starting_model.py for the physics rationale.

    python hpc/marmousi_full_das/run_traveltime_starter.py
    python hpc/marmousi_full_das/run_traveltime_starter.py --iters 80 --band 3.0

Writes $DASFWI_RESULTS/starter/vp_start.npz  (vp_start, vs_start, metadata)
for the Phase-4 pipeline to consume.

ACCEPTANCE (the number that matters): the cycle-skip fraction of `vp_start` at
the first band must sit BELOW the empirical flip threshold measured in Phase 1 -
that is what certifies the model as safe to start L2 on. The script prints the
skip fraction of the 1-D start and of the result so the improvement is explicit.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import (OUT_ROOT, OBS_FILE, NZ, NX, DX, DZ, DT, F0, NT,
                    GRAD_MASK_TOP, OPTIMIZERS, MISFIT_RUN_SETTINGS,
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
from inversion.metrics import model_scores
from inversion.skip_diagnostic import skip_fraction, skip_vs_band, ricker_f90
from inversion.starting_model import linear_vz, vs_from_vp, smooth_model, SQRT3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--band", type=float, default=3.0,
                    help="low-pass cut-off (Hz) for the kinematic stage")
    ap.add_argument("--optimizer", default="adam", choices=sorted(OPTIMIZERS))
    ap.add_argument("--misfit", default="traveltime",
                    help="kinematic misfit (traveltime; gc/sinkhorn also valid)")
    ap.add_argument("--regularization", default="tikhonov2",
                    help="smoothness prior - a starter must stay long-wavelength")
    ap.add_argument("--smooth-sigma", type=float, default=4.0, dest="smooth_sigma",
                    help="post-hoc Gaussian smoothing of the result (nodes)")
    ap.add_argument("--v-top", type=float, default=1500.0, dest="v_top")
    ap.add_argument("--v-bottom", type=float, default=4000.0, dest="v_bottom")
    ap.add_argument("--vp-vs", type=float, default=SQRT3, dest="vp_vs",
                    help="Vp/Vs for the Vs seed (sqrt(3) = Poisson solid)")
    ap.add_argument("--tag", default=None,
                    help="starter name (default i<iters>); each convergence "
                         "level needs its own so they do not overwrite")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="validate and exit (no GPU, no data). RUN THIS FIRST.")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = torch.float32
    iters = 2 if args.smoke else args.iters
    # PLAN (b): the starter's CONVERGENCE is the skip axis -- a partly converged
    # tomography is a realistic field outcome and leaves more skipping than a
    # converged one. So each level gets its OWN directory; a fixed path would
    # silently overwrite the previous level.
    out_dir = OUT_ROOT / "starter" / (args.tag or f"i{iters}")
    if not (OUT_ROOT / OBS_FILE).is_file():
        msg = f"no observed data at {OUT_ROOT / OBS_FILE} (run genobs first)"
        if args.dry_run:
            print(f"    NOTE: {msg} (fine for --dry-run)")
        else:
            raise SystemExit(f"preflight FAILED: {msg}")
    if args.dry_run:
        print(f"    dry-run OK: {iters} iters @ {min(args.band, ricker_f90(F0, DT, NT)):.2f} Hz"
              f" -> {out_dir}/vp_start.npz shaped {(NZ, NX)}; then run_switch "
              "--start route_b (PLAN STEPS 2-3)")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    f90 = ricker_f90(F0, DT, NT, integrated=True)
    f_eff = min(args.band, f90)
    print(f"=== ROUTE B traveltime starter on {device} | misfit={args.misfit} "
          f"band={args.band} Hz | {iters} iters ===", flush=True)

    # truth is loaded ONLY to score the result - it never enters the inversion
    vp_true, _ = load_models()

    # data-independent 1-D linear start (pin the water layer, as the campaign does)
    vp_1d = linear_vz(NZ, NX, args.v_top, args.v_bottom,
                      water_rows=GRAD_MASK_TOP + 2, v_water=args.v_top)
    print(f"    1-D start: {vp_1d.min():.0f}-{vp_1d.max():.0f} m/s "
          f"(v_top={args.v_top}, v_bottom={args.v_bottom}) - uses NO true-model info",
          flush=True)

    geometry = build_geometry()
    survey = build_survey(geometry)
    layer = DASObservationLayer(geometry, output="strain_rate").to(dtype).to(device)
    obs_data = SeismicData(survey)
    obs_data.load(str(OUT_ROOT / OBS_FILE))
    obs_arr = torch.as_tensor(obs_data.data["strain_rate"])

    settings = MISFIT_RUN_SETTINGS[args.misfit]
    model = build_model(vp_1d, vp_bound=[args.v_top * 0.8, args.v_bottom * 1.25],
                        vp_grad=True, device=device, dtype=dtype)
    prop = AcousticPropagator(model, survey, device=device, dtype=dtype)

    TEST_BANDS = (2.0, 3.0, 4.5, f90)

    def _syn():
        with torch.no_grad():
            rec = prop.forward(checkpoint_segments=settings["checkpoint_segments"])
            return layer(rec["u"], rec["w"]).cpu()

    def measure_skip():
        return skip_fraction(_syn(), obs_arr, DT, f_eff)

    def band_table():
        """Skip fraction at every candidate band, from ONE forward: the lags do
        not depend on the band, only the T/2 threshold does. This is what picks
        the NON-SKIP band (skip below off_below -> L2 is safe) and the SKIP band
        (skip above on_above -> L2 should fail) for the follow-up runs."""
        return skip_vs_band(_syn(), obs_arr, DT, TEST_BANDS)

    try:
        sk0 = measure_skip()
        print(f"    skip@1-D start: {sk0['skip_fraction']:.3f} "
              f"(mean|lag| {1000*sk0['mean_abs_lag_s']:.0f} ms, "
              f"threshold {1000*sk0['threshold_s']:.0f} ms)", flush=True)
    except Exception as e:                                   # noqa: BLE001
        print(f"    skip@start failed: {e}", flush=True); sk0 = None

    optimizer = OPTIMIZERS[args.optimizer](model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.75)
    reg_fn, wx, wz = build_regularization(args.regularization, device, dtype)
    fwi = AcousticFWI(propagator=prop, model=model, optimizer=optimizer,
                      scheduler=scheduler,
                      loss_fn=build_misfit(args.misfit, iterations=iters),
                      obs_data=obs_data,
                      gradient_processor=build_gradient_processor(),
                      regularization_fn=reg_fn, regularization_weights_x=wx,
                      regularization_weights_z=wz,
                      waveform_normalize=settings["normalize"],
                      cache_result=True, cache_result_epoch=10, save_fig_epoch=-1,
                      das_layer=layer, obs_key="strain_rate")

    # Chunked so a walltime kill leaves the partial model + loss curve rather
    # than nothing (trajectory-identical to one call: forward() accumulates and
    # start_iter only offsets the range).
    t0 = time.time()
    done = 0
    while done < iters:
        n = min(25, iters - done)
        fwi.forward(iteration=n, start_iter=done, batch_size=settings["batch_size"],
                    checkpoint_segments=settings["checkpoint_segments"],
                    cutoff_freq=args.band)
        done += n
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "iter_loss.npz", data=np.asarray(fwi.iter_loss))
        np.savez(out_dir / "vp_partial.npz",
                 vp=model.vp.detach().cpu().numpy(), iterations_done=done)
        print(f"  checkpoint {done}/{iters}", flush=True)
    hours = (time.time() - t0) / 3600.0

    vp_raw = model.vp.detach().cpu().numpy()
    vp_start = smooth_model(vp_raw, args.smooth_sigma) if args.smooth_sigma > 0 else vp_raw
    vp_start[:GRAD_MASK_TOP + 2] = vp_1d[:GRAD_MASK_TOP + 2]     # keep the water layer
    vs_start = vs_from_vp(vp_start, ratio=args.vp_vs)

    # skip fraction of the DELIVERED starter (the acceptance number)
    with torch.no_grad():
        model.vp.data = torch.as_tensor(vp_start, dtype=model.vp.dtype,
                                        device=model.vp.device)
    try:
        sk1 = measure_skip()
        print(f"    skip@starter : {sk1['skip_fraction']:.3f} "
              f"(mean|lag| {1000*sk1['mean_abs_lag_s']:.0f} ms)", flush=True)
    except Exception as e:                                   # noqa: BLE001
        print(f"    skip@starter failed: {e}", flush=True); sk1 = None

    sc0 = model_scores(vp_true, vp_1d)
    sc1 = model_scores(vp_true, vp_start)
    meta = dict(
        misfit=args.misfit, optimizer=args.optimizer, band=args.band,
        iterations=iters, regularization=args.regularization,
        smooth_sigma=args.smooth_sigma, vp_vs=args.vp_vs, device=device,
        runtime_h=round(hours, 3),
        skip_1d=(sk0 or {}).get("skip_fraction"),
        skip_starter=(sk1 or {}).get("skip_fraction"),
        skip_threshold_s=(sk0 or {}).get("threshold_s"),
        ssim_1d=sc0["ssim"], ssim_starter=sc1["ssim"],
        mape_1d=sc0["mape"], mape_starter=sc1["mape"],
        rms_1d=float(np.sqrt(((vp_1d - vp_true) ** 2).mean())),
        rms_starter=float(np.sqrt(((vp_start - vp_true) ** 2).mean())),
        losses_finite=bool(np.isfinite(np.asarray(fwi.iter_loss)).all()),
    )
    np.savez(out_dir / "vp_start.npz", vp_start=vp_start, vs_start=vs_start,
             vp_1d=vp_1d, **{k: v for k, v in meta.items() if v is not None})
    (out_dir / "starter_metrics.json").write_text(json.dumps(meta, indent=2))
    np.savez(out_dir / "iter_loss.npz", data=np.asarray(fwi.iter_loss))
    print(json.dumps(meta, indent=2), flush=True)

    try:
        rows = band_table()
        print("\n  SKIP vs BAND for the delivered starter "
              "(picks the non-skip / skip test bands):", flush=True)
        for r in rows:
            f, sf = r["f_max"], r["skip_fraction"]
            regime = ("SKIP  (L2 should fail -> the skip test)" if sf >= 0.58 else
                      "transition" if sf > 0.45 else
                      "NO-SKIP (L2 safe -> the non-skip test)")
            print(f"    {f:5.2f} Hz  T/2={1000*r['threshold_s']:3.0f} ms  "
                  f"skip={sf:.3f}   {regime}", flush=True)
    except Exception as e:                                   # noqa: BLE001
        print(f"  band table failed: {type(e).__name__}: {e}", flush=True)
    print("\n  ACCEPTANCE: compare skip_starter with the empirical flip threshold "
          "from Phase 1;\n  it must sit BELOW it for L2 to be safe at band 1.",
          flush=True)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4), constrained_layout=True)
    ext = [0, (NX - 1) * DX / 1000, (NZ - 1) * DZ / 1000, 0]
    for ax, (d, ttl) in zip(axes, [(vp_true, "true (reference only)"),
                                   (vp_1d, "1-D linear start"),
                                   (vp_start, "Route B starter (Vp)"),
                                   (vs_start, f"Vs seed = Vp/{args.vp_vs:.2f}")]):
        im = ax.imshow(d, extent=ext, cmap="jet",
                       vmin=(vp_true.min() if "Vs" not in ttl else vs_start.min()),
                       vmax=(vp_true.max() if "Vs" not in ttl else vs_start.max()))
        ax.set(title=ttl, xlabel="x [km]", ylabel="z [km]")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(out_dir / "starter.png", dpi=150)
    print("saved starter to", out_dir, flush=True)


if __name__ == "__main__":
    main()
