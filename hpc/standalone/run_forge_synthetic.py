"""FORGE SYNTHETIC: elastic data, acoustic inversion, KNOWN TRUTH.

The validation step between Marmousi and the FORGE field, and the only place we
can measure the things the field will hide. Park's own parameters (10 m grid,
dt 1 ms, 2 s, Ricker 10 Hz peak) on the MEASURED FORGE geometry.

WHY IT IS NOT AN INVERSE CRIME, which is the entire point:
  * data are generated ELASTICALLY and inverted ACOUSTICALLY, so Rayleigh and
    converted waves exist and are UNMODELLABLE by the inverter -- Park report
    strong surface waves in the near-offset gathers, and this is their situation
  * the inversion may assume a DIFFERENT source wavelet from the true one
    (Park assume a 10 Hz Ricker; the real source is unknown)
  * noise at a controllable SNR
  * the surface is the MEASURED 161.6 m ramp with an air layer, not a flat datum
`proxy_model.generate_observed` is an inverse crime by its own docstring, and
our four "settled negatives" (windowing, channel weighting, lambda/4 smoothing,
multiscale) were all measured on it. This driver is what can falsify them.

Unlike the field, TRUTH IS KNOWN here, so SSIM/MAPE are meaningful and every
choice can be decided by measurement rather than argument.

    python hpc/standalone/run_forge_synthetic.py --dry-run
    python hpc/standalone/run_forge_synthetic.py --arm switch --refiner convsi \
        --f0-true 14 --f0-assumed 10 --smoke
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ADFWI_local"))

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ADFWI.fwi import AcousticFWI
from ADFWI.model import AcousticModel
from ADFWI.propagator import AcousticPropagator, GradProcessor

from das.geometry import merge_fibers
from forge.proxy_model import (forge_proxy_vp, forge_fibers, vibroseis_line,
                               V_AIR)
from forge.realistic_synthetic import elastic_observed, mismatched_wavelet
from inversion import config, near_surface as ns
from inversion.adaptive_misfit import BlendedMisfit, SkipSwitch
from inversion.metrics import model_scores
from inversion.skip_diagnostic import skip_fraction

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_field_das import parse_bands, allocate_iters                # noqa: E402

OUT_ROOT = Path(os.environ.get("DASFWI_RESULTS", "results")) / "forge_synthetic"

# ---- Park's INV3 geometry, and the MEASURED FORGE surface ------------------
DX = DZ = 10.0                 # Park: "the spatial grid interval is 10 m"
NT, DT = 2000, 1e-3            # Park: "0.001 s over a total recording time of 2 s"
#: measured from 318 shots: 161.6 m of relief over ~2960 m, corr(x,z)=+0.994
RELIEF_M, SECTION_M = 161.6, 2960.0
DEPTH_M = 2000.0               # DAS-VSP constrains the upper ~1 km; model deeper
ARMS = ("switch", "fixedk", "l2", "gc", "convsi", "tfphase", "envelope")
SOLO_ARMS = ("l2", "gc", "convsi", "tfphase")
OPTIMIZERS = config.LIU_OPTIMIZERS


def build_truth(nz, nx, ground):
    """True Vp: the FORGE proxy (zones I/II/III) under the measured ramp."""
    vp = forge_proxy_vp(nz, nx, dz=DZ, z_air=0.0)      # zones, no air yet
    return ns.with_air_layer_topo(vp, ground, DZ)      # air ABOVE the topography


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", default="switch", choices=ARMS)
    ap.add_argument("--refiner", default="gc",
                    choices=("l2", "gc", "convsi", "tfphase"))
    ap.add_argument("--robust", default="envelope", choices=("envelope", "tfphase"))
    ap.add_argument("--optimizer", default="adam", choices=sorted(OPTIMIZERS))
    ap.add_argument("--iterations", type=int, default=150)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--fixed-k", type=int, default=50, dest="fixed_k")
    ap.add_argument("--bands", default=None,
                    help="e.g. 5,8,12,full -- FORGE spans ~2.7 octaves so a "
                         "cascade is real here, unlike Marmousi's 1.06")
    ap.add_argument("--iter-alloc", default="final-heavy", dest="iter_alloc",
                    choices=("equal", "final-heavy"))
    # ---- the non-crime knobs ----------------------------------------------
    ap.add_argument("--f0-true", type=float, default=10.0, dest="f0_true",
                    help="TRUE source peak frequency")
    ap.add_argument("--f0-assumed", type=float, default=None, dest="f0_assumed",
                    help="what the INVERSION assumes. Park assume 10 Hz and the "
                         "real source is unknown; differing from --f0-true "
                         "measures what that assumption costs (#50). "
                         "Default: same as true (the control).")
    ap.add_argument("--snr", type=float, default=20.0,
                    help="dB. Conditioning exists for noise; a noiseless "
                         "synthetic cannot test it.")
    ap.add_argument("--acoustic-data", action="store_true", dest="acoustic_data",
                    help="generate ACOUSTICALLY = inverse crime. Only as a "
                         "CONTROL, to separate physics mismatch from everything "
                         "else -- never as the headline result.")
    ap.add_argument("--flat-datum", action="store_true", dest="flat_datum",
                    help="no air layer (control): the measured 162 m ramp then "
                         "fabricates a free-surface ghost 215 ms late")
    ap.add_argument("--n-shots", type=int, default=12, dest="n_shots")
    ap.add_argument("--grad-smooth", default="none", dest="grad_smooth",
                    choices=("none", "wavelength"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    iterations = 4 if args.smoke else args.iterations
    f0_asm = args.f0_assumed if args.f0_assumed is not None else args.f0_true
    arm = args.arm
    if arm in SOLO_ARMS:
        args.refiner = arm
    bands = parse_bands(args.bands) if args.bands else [None]
    iters_by_band = allocate_iters(max(1, iterations // len(bands)), len(bands),
                                   args.iter_alloc)

    # EVERY knob in the tag: this bug class has struck five times.
    pair = "" if args.refiner in ("l2", arm) else f"-{args.refiner}"
    rb = "" if args.robust == "envelope" or arm in SOLO_ARMS else f"+{args.robust}"
    tag = ("fsyn_" + arm + pair + rb + "_" + args.optimizer
           + (f"_w{args.f0_true:g}-{f0_asm:g}" if f0_asm != args.f0_true
              else f"_w{args.f0_true:g}")
           + f"_snr{args.snr:g}"
           + ("_ac" if args.acoustic_data else "_el")
           + ("_flat" if args.flat_datum else "_topoair")
           + ("_b" + args.bands.replace(",", "-") if args.bands else "")
           + ("_fh" if args.bands and args.iter_alloc == "final-heavy" else "")
           + ("_g" if args.grad_smooth != "none" else "")
           + ("_smoke" if args.smoke else ""))
    out_dir = OUT_ROOT / tag

    nz, nx = int(DEPTH_M / DZ), int(SECTION_M / DX)
    ground = np.linspace(0.0, RELIEF_M, nx)            # the MEASURED ramp
    if args.flat_datum:
        ground = np.zeros(nx)
    n_air = int(ns.air_mask_topo(nz, nx, ground, DZ).sum(axis=0).max())

    print(f"=== {tag} on {dev} ===", flush=True)
    print(f"    grid {nz}x{nx} @ {DZ:g} m, nt={NT}, dt={DT}  (Park's INV3)",
          flush=True)
    print(f"    data: {'ACOUSTIC (INVERSE CRIME)' if args.acoustic_data else 'ELASTIC'}"
          f", SNR {args.snr:g} dB, true f0 {args.f0_true:g} Hz, "
          f"inversion assumes {f0_asm:g} Hz"
          + ("  <-- MISMATCHED" if f0_asm != args.f0_true else "  (matched)"),
          flush=True)
    print("    " + ns.describe(nz, nx, DZ, n_air * DZ, ns.VP_BOUND_FIELD[0],
                               2 * args.f0_true, DX,
                               src_z=ground), flush=True)
    print(f"    arm={arm} refiner={args.refiner} robust={args.robust} "
          f"bands={bands} iters/band={iters_by_band}", flush=True)
    if args.dry_run:
        print("    dry-run OK -- nothing was run", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    vp_true = build_truth(nz, nx, ground)
    geom = merge_fibers(list(forge_fibers(
        nz, x_well_a=0.45 * SECTION_M, x_well_b=0.62 * SECTION_M,
        z_top=RELIEF_M + 100.0, n_channels=int(0.7 * nz), dz=DZ, dx=DX)))
    sx = np.linspace(0.05 * nx, 0.95 * nx, args.n_shots).astype(int)
    # Sources sit ON THE GROUND, per source. The ramp spans 16 grid rows, so a
    # single median row buries 5 of 12 sources IN THE AIR -- they radiate into
    # 340 m/s, the gather is empty, and skip_fraction returns NaN.
    src_z_idx = np.clip((ground[sx] / DZ).astype(int), 0, nz - 1)
    src = vibroseis_line(NT, DT, args.f0_true, list(sx), list(src_z_idx))
    print(f"    {len(sx)} sources on the ground, rows "
          f"{src_z_idx.min()}-{src_z_idx.max()}", flush=True)

    t_gen = time.time()
    if args.acoustic_data:
        from forge.proxy_model import make_acoustic_model, generate_observed
        m_true = make_acoustic_model(vp_true, dx=DX, dz=DZ, nabc=20,
                                     device=dev, dtype=torch.float32)
        obs, survey, layer = generate_observed(m_true, geom, src, device=dev,
                                               dtype=torch.float32)
    else:
        obs, survey, layer = elastic_observed(vp_true, geom, src, DX, DZ,
                                              nabc=20, snr_db=args.snr,
                                              device=dev, dtype=torch.float32)
    obs_arr = np.asarray(obs.data["strain_rate"])
    print(f"    observed {obs_arr.shape} in {time.time()-t_gen:.0f}s, "
          f"finite={np.isfinite(obs_arr).all()}", flush=True)

    # ---- starting model: smoothed truth is NOT allowed here; use a 1-D ramp -
    zc = (np.arange(nz) + 0.5) * DZ
    vp0 = np.tile(np.clip(1500.0 + 1.8 * zc, *ns.VP_BOUND_FIELD)[:, None],
                  (1, nx))
    vp0 = ns.with_air_layer_topo(vp0, ground, DZ)
    water_mask = ns.air_mask_topo(nz, nx, ground, DZ)

    model = AcousticModel(0, 0, nx, nz, DX, DZ, vp0,
                          np.power(vp0, 0.25) * 310.0,
                          vp_bound=list(ns.VP_BOUND_FIELD), vp_grad=True,
                          water_layer_mask=water_mask, free_surface=True,
                          abc_type="PML", nabc=20, device=dev,
                          dtype=torch.float32)
    prop = AcousticPropagator(model, survey, device=dev, dtype=torch.float32)
    opt = OPTIMIZERS[args.optimizer](model.parameters())
    # AcousticFWI.forward calls scheduler.step() unconditionally -> never None
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.75)
    f_eff = 2.0 * args.f0_true
    loss_fn = BlendedMisfit(
        config.build_misfit(args.refiner, dt=DT, iterations=iterations),
        config.build_misfit(args.robust, dt=DT, iterations=iterations),
        lam=1.0, normalize=True)
    st = config.MISFIT_SETTINGS[args.refiner]
    gp = GradProcessor(grad_mute=n_air,
                       marine_or_land=("marine" if n_air else "land"),
                       grad_smooth=(ns.anisotropic_span(ns.VP_BOUND_FIELD[0],
                                                        f_eff, DX, DZ)[1]
                                    if args.grad_smooth == "wavelength" else 0))
    fwi = AcousticFWI(propagator=prop, model=model, optimizer=opt, scheduler=sch,
                      loss_fn=loss_fn, obs_data=obs, gradient_processor=gp,
                      waveform_normalize=st["normalize"], cache_result=True,
                      save_fig_epoch=-1, das_layer=layer,
                      obs_key="strain_rate")

    def _save(done, hours, complete):
        vp = model.vp.detach().cpu().numpy()
        L = np.asarray(fwi.iter_loss)
        below = ~water_mask
        sc = model_scores(vp_true[below.any(axis=1)], vp[below.any(axis=1)])
        m = dict(tag=tag, arm=arm, refiner=args.refiner, robust=args.robust,
                 optimizer=args.optimizer, iterations=int(sum(iters_by_band)),
                 iterations_done=int(done), complete=bool(complete),
                 runtime_h=round(hours, 3),
                 f0_true=args.f0_true, f0_assumed=f0_asm,
                 wavelet_mismatched=bool(f0_asm != args.f0_true),
                 snr_db=args.snr, elastic_data=not args.acoustic_data,
                 flat_datum=bool(args.flat_datum), n_air_rows=int(n_air),
                 bands=[("full" if b is None else b) for b in bands],
                 iters_per_band=list(iters_by_band),
                 grad_smooth=args.grad_smooth,
                 ssim=float(sc["ssim"]), mape=float(sc["mape"]),
                 loss_first=float(L[0]) if len(L) else None,
                 loss_last=float(L[-1]) if len(L) else None,
                 losses_finite=bool(np.isfinite(L).all()),
                 model_finite=bool(np.isfinite(vp).all()),
                 diverged=bool(not np.isfinite(sc["ssim"])
                               or not np.isfinite(L).all()),
                 trajectory=traj)
        np.savez(out_dir / "iter_vp.npz", data=np.asarray(fwi.iter_vp))
        np.savez(out_dir / "vp.npz", vp=vp, vp_true=vp_true, vp_init=vp0)
        (out_dir / "metrics.json").write_text(json.dumps(m, indent=2))
        return vp, m

    traj, t0, done = [], time.time(), 0
    ctrl = SkipSwitch() if arm == "switch" else None
    for bi, f_band in enumerate(bands):
        fb = f_eff if f_band is None else min(f_band, f_eff)
        if arm == "switch":
            ctrl = SkipSwitch()        # fresh per band: skip jumps at the edge
        in_band = 0
        while in_band < iters_by_band[bi]:
            with torch.no_grad():
                rec = prop.forward(checkpoint_segments=st["checkpoint_segments"])
                syn = layer(rec["u"], rec["w"]).cpu()
            sk = float(skip_fraction(syn, obs_arr, DT, fb)["skip_fraction"])
            if not np.isfinite(sk):
                # A NaN measurement must NOT read as "no skip" -- that would
                # hand straight to the refiner at a badly skipped start, which
                # is the worst possible move. Fail SAFE: assume skipped.
                print("  *** skip is NaN (empty synthetic?) -- assuming SKIPPED",
                      flush=True)
                sk = 1.0
            lam = (ctrl.update(sk) if arm == "switch" else
                   (1.0 if done < args.fixed_k else 0.0) if arm == "fixedk" else
                   (0.0 if arm in SOLO_ARMS else 1.0))
            loss_fn.set_lambda(lam)
            traj.append(dict(iter=done, band=(None if f_band is None else f_band),
                             skip=sk, lam=float(lam)))
            print(f"  iter {done:3d} band {bi+1}/{len(bands)}: skip={sk:.3f} "
                  f"-> lambda={lam:.0f} "
                  f"({args.robust.upper() if lam >= .5 else args.refiner.upper()})",
                  flush=True)
            n = min(args.chunk, iters_by_band[bi] - in_band)
            fwi.forward(iteration=n, start_iter=done,
                        batch_size=st["batch_size"],
                        checkpoint_segments=st["checkpoint_segments"],
                        cutoff_freq=f_band)
            done += n; in_band += n
            _save(done, (time.time() - t0) / 3600.0,
                  complete=(done >= sum(iters_by_band)))
            print(f"  checkpoint {done}/{sum(iters_by_band)}", flush=True)

    vp, metrics = _save(done, (time.time() - t0) / 3600.0, complete=True)
    print(json.dumps({k: v for k, v in metrics.items() if k != "trajectory"},
                     indent=2), flush=True)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    ext = [0, nx * DX / 1000, nz * DZ / 1000, 0]
    for a, d, t in ((ax[0], vp_true, "true"), (ax[1], vp0, "initial"),
                    (ax[2], vp, f"inverted (SSIM {metrics['ssim']:.3f})")):
        im = a.imshow(d, extent=ext, aspect="auto", cmap="jet",
                      vmin=ns.VP_BOUND_FIELD[0], vmax=vp_true.max())
        a.set_title(t); a.set_xlabel("x (km)"); a.set_ylabel("z (km)")
    fig.colorbar(im, ax=ax, label="Vp (m/s)")
    fig.savefig(out_dir / "result.png", dpi=110); plt.close(fig)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
