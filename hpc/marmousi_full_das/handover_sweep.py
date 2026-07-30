"""OFFLINE HAND-OVER SWEEP (verification F4.2): simulate the staged
envelope->L2 switch using compute the gate already paid for. ONE GPU job, ~2 SU.

The gate cached the model every 10 iterations for every cell. This script takes
a robust cell's trajectory (default weci_adam@s16), and

  PROBE    for each cached snapshot: one no-grad forward -> measured skip + SSIM
           (the robust run's skip trajectory, for free), then
  RESTART  from selected snapshots: a short pure-L2 leg -> does L2 help yet?

The lowest snapshot skip at which the L2 restart clearly gains is the EMPIRICAL
hand-over point: set SkipSwitch's off_below just above the highest skip where L2
still helped (biased conservative). This previews the Phase-A headline result
before any switch code runs in anger.

    python hpc/marmousi_full_das/handover_sweep.py                 # defaults
    python hpc/marmousi_full_das/handover_sweep.py --combo weci_adam \
        --rung s16 --restart-from 50,100,150,200 --restart-iters 100

Requires the gate results (iter_vp.npz) and obs_data_das.npz.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import (OUT_ROOT, OBS_FILE, NZ, NX, DT, F0, NT,
                    MISFIT_RUN_SETTINGS, OPTIMIZERS,
                    pick_device, load_models, build_model, build_geometry,
                    build_survey, build_misfit, build_gradient_processor,
                    DASObservationLayer, SeismicData, AcousticPropagator)

import numpy as np
import torch

from ADFWI.fwi import AcousticFWI
from inversion.metrics import model_scores
from inversion.skip_diagnostic import skip_fraction, ricker_f90
from inversion.adaptive_misfit import SKIP_ON_ABOVE, SKIP_OFF_BELOW

CACHE_EVERY = 10          # ADFWI cache_result_epoch used throughout the gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combo", default="weci_adam",
                    help="gate cell whose trajectory to probe")
    ap.add_argument("--rung", default="s16")
    ap.add_argument("--restart-from", default="50,100,150,200",
                    dest="restart_from",
                    help="iterations whose snapshots get an L2 restart leg")
    ap.add_argument("--restart-iters", type=int, default=100, dest="restart_iters")
    ap.add_argument("--optimizer", default="adam")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = torch.float32
    cell = (OUT_ROOT / args.combo if args.rung == "s6"
            else OUT_ROOT / f"ladder_{args.rung}" / args.combo)
    snaps = np.load(cell / "iter_vp.npz")["data"]        # (n_snap, NZ, NX)
    if snaps.ndim == 2:                                  # single snapshot edge
        snaps = snaps[None]
    n_snap = snaps.shape[0]
    print(f"=== hand-over sweep on {args.combo}@{args.rung} ({device}) ===\n"
          f"    {n_snap} cached snapshots (every {CACHE_EVERY} iters); "
          f"thresholds on_above={SKIP_ON_ABOVE} off_below={SKIP_OFF_BELOW}",
          flush=True)

    vp_true, vp_init = load_models(args.rung)
    geometry = build_geometry()
    survey = build_survey(geometry)
    layer = DASObservationLayer(geometry, output="strain_rate").to(dtype).to(device)
    obs_data = SeismicData(survey)
    obs_data.load(str(OUT_ROOT / OBS_FILE))
    obs_arr = torch.as_tensor(obs_data.data["strain_rate"])
    f90 = ricker_f90(F0, DT, NT, integrated=True)
    bounds = [float(vp_true.min()), float(vp_true.max())]

    def probe(vp):
        """One no-grad forward on model vp -> (skip_fraction, ssim)."""
        model = build_model(vp.astype(np.float64), vp_bound=bounds,
                            vp_grad=False, device=device, dtype=dtype)
        prop = AcousticPropagator(model, survey, device=device, dtype=dtype)
        with torch.no_grad():
            rec = prop.forward(checkpoint_segments=1)
            syn = layer(rec["u"], rec["w"]).cpu()
        sk = skip_fraction(syn, obs_arr, DT, f90)["skip_fraction"]
        return sk, model_scores(vp_true, vp)["ssim"]

    # ---- PROBE: the robust run's skip trajectory ------------------------------
    print(f"\n--- probe: skip/SSIM at each cached snapshot ---")
    print(f"{'iter':>6s} {'skip':>7s} {'ssim':>7s}")
    skips = {}
    for k in range(n_snap):
        it = k * CACHE_EVERY
        sk, ss = probe(snaps[k])
        skips[it] = sk
        mark = "  <- below off_below" if sk <= SKIP_OFF_BELOW else ""
        print(f"{it:6d} {sk:7.3f} {ss:7.3f}{mark}", flush=True)

    # ---- RESTART: short pure-L2 legs from selected snapshots ------------------
    want = [int(t) for t in args.restart_from.split(",") if t.strip()]
    rows = []
    for it in want:
        k = min(it // CACHE_EVERY, n_snap - 1)
        vp0 = snaps[k].astype(np.float64)
        sk0 = skips[k * CACHE_EVERY]
        ss0 = model_scores(vp_true, vp0)["ssim"]
        print(f"\n--- L2 restart from iter {k * CACHE_EVERY} "
              f"(skip {sk0:.3f}, ssim {ss0:.3f}), {args.restart_iters} iters ---",
              flush=True)
        model = build_model(vp0, vp_bound=bounds, vp_grad=True,
                            device=device, dtype=dtype)
        prop = AcousticPropagator(model, survey, device=device, dtype=dtype)
        optimizer = OPTIMIZERS[args.optimizer](model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100,
                                                    gamma=0.75, last_epoch=-1)
        settings = MISFIT_RUN_SETTINGS["l2"]
        fwi = AcousticFWI(propagator=prop, model=model, optimizer=optimizer,
                          scheduler=scheduler,
                          loss_fn=build_misfit("l2", iterations=args.restart_iters),
                          obs_data=obs_data,
                          gradient_processor=build_gradient_processor(),
                          waveform_normalize=settings["normalize"],
                          cache_result=True, cache_result_epoch=CACHE_EVERY,
                          save_fig_epoch=-1,
                          das_layer=layer, obs_key="strain_rate")
        t0 = time.time()
        fwi.forward(iteration=args.restart_iters,
                    batch_size=settings["batch_size"],
                    checkpoint_segments=settings["checkpoint_segments"])
        vp1 = model.vp.detach().cpu().numpy()
        sk1, ss1 = probe(vp1)
        gain = ss1 - ss0
        rows.append(dict(from_iter=k * CACHE_EVERY, skip_at_start=sk0,
                         ssim_before=ss0, ssim_after=ss1, gain=gain,
                         skip_after=sk1,
                         minutes=round((time.time() - t0) / 60, 1)))
        print(f"    ssim {ss0:.3f} -> {ss1:.3f}  (gain {gain:+.3f}); "
              f"skip {sk0:.3f} -> {sk1:.3f}", flush=True)

    # ---- verdict ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"{'from_iter':>9s} {'skip@start':>10s} {'ssim before->after':>19s} {'gain':>7s}")
    helped = [r for r in rows if r["gain"] >= 0.02]
    for r in rows:
        print(f"{r['from_iter']:9d} {r['skip_at_start']:10.3f} "
              f"{r['ssim_before']:8.3f} -> {r['ssim_after']:.3f} {r['gain']:+7.3f}")
    if helped:
        hi = max(r["skip_at_start"] for r in helped)
        print(f"\nEMPIRICAL hand-over point: L2 already helps at skip <= {hi:.3f}.")
        print(f"  -> set SkipSwitch off_below ~ {min(hi, SKIP_ON_ABOVE - 0.05):.2f} "
              f"(current default {SKIP_OFF_BELOW})")
    else:
        print("\nL2 never helped at the probed snapshots -> hand-over premature "
              "everywhere probed; keep off_below low (or probe later snapshots).")
    out = cell / "handover_sweep.json"
    with open(out, "w") as f:
        json.dump(dict(combo=args.combo, rung=args.rung,
                       probe={str(k * CACHE_EVERY): skips[k * CACHE_EVERY]
                              for k in range(n_snap)},
                       restarts=rows), f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
