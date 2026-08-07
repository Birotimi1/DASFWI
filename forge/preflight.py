"""ONE COMMAND that validates the WHOLE field chain on the REAL data.

    python forge/preflight.py --well 78A-32 --shots 20 --dz 10 --f0 10

Written because the failures on this project have all had the same shape: each
LAYER passed its own test and the SEAM between layers was never checked until a
job ran. That cost queue time we no longer have. Every check below is one that
has ALREADY failed at least once here, silently:

  DATA        .sgy present                  -- the first field smoke died here
  SHOT SPREAD subset spans the line         -- files[:20] took 6% of a 2960 m
                                               line, clustered at one end
  RELIEF      topography reaches the model  -- reported 0 m on data measured
                                               at 161.6 m, because of the above
  CFL         dt is stable for THIS vmax    -- Park's 1 ms blew up at 2 s, but
                                               only at full record length
  AMPLITUDE   no denormal traces            -- DAS strain rate ~1e-16 underflows
                                               float32 and normalisation -> NaN
  AIR LAYER   built, and survives clipping  -- vp_bound clamped 340 -> 1500
  FORWARD     one propagation is finite
  SKIP        the diagnostic is measurable  -- NaN silently disabled the switch
  MISFIT      finite loss AND finite, non-zero gradient

SITE-AGNOSTIC BY CONSTRUCTION: nothing is hardcoded to FORGE. Wells are
discovered from the data directory, relief and geometry are MEASURED from the
SEG-Y headers, and the stability limit is computed from the velocity bounds you
pass. Point it at another site's directory and it re-derives everything.

Exit code 0 = safe to submit. Non-zero = do not spend SUs.
"""

# --- login-node CPU budget -------------------------------------------------- #
# THIS RAN ON A LOGIN NODE AND WAS KILLED: "cpu time 2105.1 seconds exceeded
# limit 1800". Not because it does much work -- because torch/BLAS default to
# one thread per core, and the limit counts CPU-seconds = wall x threads. On a
# 24-core box the same run cost 222 CPU-s at 380% CPU with 189 s of that pure
# SYSTEM time, i.e. threads fighting, not computing. Capped to one thread it
# costs 16 CPU-s and finishes FASTER in wall time (21 s vs 58 s).
# Must be set BEFORE numpy/torch are imported, so it sits above the bootstrap.
import os as _os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")     # setdefault: an explicit export wins
# ---------------------------------------------------------------------------- #

# --- import bootstrap: depend on NO environment ----------------------------- #
# `python forge/preflight.py` puts forge/ on sys.path, NOT the repo root, so
# ADFWI/forge/inversion are unimportable unless PYTHONPATH happens to be set.
# It was set in every shell I tested in and NOT in the user's cluster shell, so
# this died there with a bare ModuleNotFoundError. Resolve from __file__
# instead: walk up to the directory holding forge/ and inversion/, and add the
# tracked ADFWI package next to it.
import sys as _sys
from pathlib import Path as _Path

for _r in _Path(__file__).resolve().parents:
    if (_r / "forge").is_dir() and (_r / "inversion").is_dir():
        for _p in (_r, _r / "ADFWI_local"):
            if (_p / "ADFWI").is_dir() and str(_p) not in _sys.path:
                _sys.path.insert(0, str(_p))
        if str(_r) not in _sys.path:
            _sys.path.insert(0, str(_r))
        break
# ---------------------------------------------------------------------------- #
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_R = []                                   # (ok, name, detail)


def chk(ok, name, detail=""):
    _R.append((bool(ok), name, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:26s} {detail}", flush=True)
    return bool(ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--well", default=None,
                    help="well subdirectory; default = the first discovered")
    ap.add_argument("--data-dir", default=os.environ.get("FORGE_DAS_DIR"))
    ap.add_argument("--shots", type=int, default=20)
    ap.add_argument("--dz", type=float, default=10.0)
    ap.add_argument("--f0", type=float, default=10.0)
    ap.add_argument("--record-s", type=float, default=2.0, dest="record_s")
    ap.add_argument("--device", default=None,
                    help="cuda | cpu | mps. Default: auto. Run with cuda "
                         "inside a GPU job to validate the device the "
                         "inversion will actually use.")
    ap.add_argument("--vmin", type=float, default=1000.0)
    ap.add_argument("--vmax", type=float, default=6000.0)
    ap.add_argument("--spread-frac", type=float, default=0.8, dest="spread",
                    help="the shot subset must span at least this fraction of "
                         "the full acquisition line")
    args = ap.parse_args()

    print(f"=== FIELD PREFLIGHT: {args.data_dir} ===\n", flush=True)

    # ---- 1. DATA ---------------------------------------------------------- #
    if not args.data_dir:
        chk(False, "data dir", "FORGE_DAS_DIR unset and --data-dir not given")
        return 2
    root = Path(args.data_dir)
    wells = sorted(d.name for d in root.iterdir()
                   if d.is_dir() and list(d.glob("*.sgy")))
    if not chk(wells, "wells discovered", f"{wells or 'NONE'}  (site-agnostic: "
                                          f"read from the directory)"):
        return 2
    well = args.well or wells[0]
    n_sgy = len(list((root / well).glob("*.sgy")))
    chk(n_sgy > 0, "sgy files", f"{well}: {n_sgy}")

    from forge.field_loader import read_shot_geometry, project_to_2d
    from inversion import near_surface as ns

    # ---- 2. SHOT SPREAD --------------------------------------------------- #
    gall = read_shot_geometry(root / well, n_shots=None)
    pall = project_to_2d(gall["src_xyz"], gall["rcv_xyz"])
    full = float(pall["src_x"].max() - pall["src_x"].min())
    g = read_shot_geometry(root / well, n_shots=args.shots)
    p = project_to_2d(g["src_xyz"], g["rcv_xyz"])
    sub = float(p["src_x"].max() - p["src_x"].min())
    chk(sub >= args.spread * full, "shot spread",
        f"{len(g['files'])} shots span {sub:.0f} m of {full:.0f} m "
        f"({100*sub/max(full,1):.0f}%)")

    # ---- 3. RELIEF / TOPOGRAPHY ------------------------------------------- #
    relief = ns.topography_relief(p["src_z"])
    relief_all = ns.topography_relief(pall["src_z"])
    lam = args.vmin / (2.0 * args.f0)
    need_air = relief_all > 0.25 * lam
    chk(abs(relief - relief_all) < 0.15 * max(relief_all, 1.0), "relief in subset",
        f"{relief:.0f} m vs {relief_all:.0f} m full "
        f"(lambda/4 = {0.25*lam:.0f} m -> air layer "
        f"{'REQUIRED' if need_air else 'not needed'})")

    # ---- 4. CFL ----------------------------------------------------------- #
    dt_max = ns.cfl_dt(args.vmax, args.dz)
    dt, nt = ns.stable_time_axis(args.vmax, args.dz, args.record_s)
    chk(dt <= dt_max, "CFL",
        f"dz={args.dz:g} m, vmax={args.vmax:g} -> dt<={dt_max*1e3:.3f} ms; "
        f"using {dt*1e3:.3f} ms, nt={nt} for {args.record_s:g}s")

    # ---- 5. AMPLITUDE / DENORMALS ----------------------------------------- #
    from forge.field_loader import load_strain_gathers
    d = load_strain_gathers(g["files"][:min(3, len(g["files"]))],
                            len(g["rcv_xyz"]))
    pt = np.abs(d).max(axis=1)
    n_zero = int((pt == 0).sum())
    n_den = int(((pt > 0) & (pt < 1e-30)).sum())
    chk(np.isfinite(d).all(), "data finite", f"max|d| = {np.abs(d).max():.3e}")
    chk(n_zero == 0 and n_den == 0, "no dead/denormal traces",
        f"{n_zero} dead, {n_den} denormal of {pt.size} "
        f"(per-trace max {pt.min():.2e}..{pt.max():.2e})")

    # ---- 6. AIR LAYER ----------------------------------------------------- #
    nz = int(2000.0 / args.dz)
    nx = int((full + 800.0) / args.dz)
    ground = ns.surface_profile(p["src_x"], p["src_z"], nx, args.dz,
                                x0=float(p["src_x"].min()) - 400.0)
    n_air = int(ns.air_mask_topo(nz, nx, ground, args.dz).sum(axis=0).max())
    chk((n_air > 0) == need_air, "air layer",
        f"{n_air} rows at the deepest column "
        f"({'built' if n_air else 'none'}; ground spans "
        f"{ground.min():.0f}-{ground.max():.0f} m)")
    vp = ns.with_air_layer_topo(np.full((nz, nx), 3000.0), ground, args.dz)
    mask = ns.air_mask_topo(nz, nx, ground, args.dz)
    kept = np.where(mask, vp, np.clip(vp, args.vmin, args.vmax))
    chk(not n_air or abs(kept[mask].min() - ns.V_AIR) < 1e-6,
        "air survives clipping",
        f"masked min {kept[mask].min() if n_air else float('nan'):.0f} m/s "
        f"vs bound {args.vmin:.0f}")

    # ---- 7-9. FORWARD / SKIP / MISFIT ------------------------------------- #
    from inversion.skip_diagnostic import skip_fraction
    # TWO-SIDED. A diagnostic that always returns 0 -- or always NaN, which is
    # what silently disabled the switch on the first field smoke -- passes any
    # one-sided check. So probe BOTH regimes: a shift well inside T/2 must read
    # ~0, and one well beyond it must read ~1.
    half_T = 1.0 / (2.0 * args.f0)
    n_small = max(1, int(0.1 * half_T / g["dt"]))
    n_big = int(1.8 * half_T / g["dt"])
    sk_lo = skip_fraction(np.roll(d, n_small, axis=1), d, g["dt"],
                          args.f0)["skip_fraction"]
    sk_hi = skip_fraction(np.roll(d, n_big, axis=1), d, g["dt"],
                          args.f0)["skip_fraction"]
    chk(np.isfinite(sk_lo) and np.isfinite(sk_hi) and sk_lo < 0.2 < 0.8 < sk_hi,
        "skip discriminates",
        f"{n_small*g['dt']*1e3:.0f} ms -> {sk_lo:.3f} (want ~0);  "
        f"{n_big*g['dt']*1e3:.0f} ms -> {sk_hi:.3f} (want ~1);  "
        f"T/2 = {half_T*1e3:.0f} ms")

    # ---- 5b. FIBRE ORIENTATION: does the moveout agree with the headers? -- #
    # THE CHECK THAT WOULD HAVE SAVED THE 13-CELL CAMPAIGN. Those runs completed
    # with a channel->depth map that was reversed, so receivers sat where no
    # velocity model could explain the arrivals. Every symptom pointed
    # elsewhere -- skip stuck at 1.000, convsi collapsing to 1e-7, gc diverging,
    # the model railing to both bounds -- and none of them said "geometry".
    # Site-agnostic: nothing is assumed about the well, only that a direct
    # arrival exists and that rock carries P waves at 0.5-9 km/s.
    from inversion.fibre_geometry import check_orientation
    ordr = np.argsort(p["chan_z"])                  # the loader's depth belief
    spacing = float(np.median(np.diff(np.sort(p["chan_z"]))))
    # NEAR-OFFSET shots only. The amplitude block above reads the first three
    # files, which after the shot-spread fix are spread ACROSS the line -- far
    # offsets, where the ray does not run along the fibre and the lags are not
    # resolvable. Feeding those in gave 1 usable measurement of 9 and a
    # confident, wrong "AGREES".
    near = np.argsort(np.abs(p["src_x"]))[:3]
    d_near = load_strain_gathers([g["files"][i] for i in near],
                                 len(g["rcv_xyz"]))
    ori = check_orientation(d_near[:, :, ordr], spacing, g["dt"])
    chk(ori["agrees"] is True, "fibre orientation",
        f"{ori['verdict'][:96]}"
        + (f"  [v={ori['v_median']:+.0f} m/s, {ori['n_usable']}/{ori['n_total']}]"
           if "v_median" in ori else ""))

    # ---- 5c. FIRST-BREAK PICKS: do they follow a COHERENT arrival? -------- #
    # The picks fed the `traveltime` starting model, and on FORGE they landed in
    # the 0.05-0.25 s pre-arrival noise while the real arrival was at
    # 0.42-0.65 s. The resulting starter came out SATURATED at the 6000 m/s
    # upper bound -- a constant block with no information -- so every
    # `--starting traveltime` cell was an invalid test rather than a bad result.
    # Nothing flagged it; the run completed and the loss fell.
    # A real arrival is CONTINUOUS across a fibre, so scatter about a smooth
    # trend is the test. Site-agnostic: it assumes only continuity.
    from forge.traveltime_tomography import pick_first_breaks
    d0 = d[0] if d.ndim == 3 else d
    pk = pick_first_breaks(d0, g["dt"], min_time_s=0.02)
    fin = np.isfinite(pk)
    if fin.sum() >= 8:
        idx = np.arange(pk.size)[fin]
        trend = np.poly1d(np.polyfit(idx, pk[fin], 3))(idx)
        scat = float(np.median(np.abs(pk[fin] - trend)))
        frac = float(fin.mean())
        # 3 half-periods of scatter means the picks are not on one event
        tol = 1.5 / args.f0
        chk(scat < tol and frac > 0.5, "first-break picks",
            f"{fin.sum()}/{pk.size} picked ({frac*100:.0f}%), scatter about a "
            f"smooth trend {scat*1e3:.1f} ms (limit {tol*1e3:.0f} ms)")
    else:
        chk(False, "first-break picks",
            f"only {int(fin.sum())} of {pk.size} channels picked -- the "
            f"traveltime starter cannot be built from this")

    # a modestly shifted copy stands in for a synthetic: enough mismatch to
    # give a non-zero gradient, not so much that the misfit saturates
    syn = np.roll(d, n_small, axis=1)
    import torch
    torch.set_num_threads(1)   # see the CPU-budget note at the top
    from inversion import config
    from inversion.das_conditioning import ConditionedMisfit
    from inversion.device import pick_device

    # >>> VALIDATE ON THE DEVICE THE JOBS WILL ACTUALLY USE. <<<
    # Running every check on the login-node CPU tests a configuration that no
    # job ever runs. The misfits, the dtype and the memory footprint all behave
    # differently on an H100, and "it worked on CPU" has never been the
    # question. allow_cpu=True because a login-node run is a legitimate way to
    # use this -- the guard exists for the DRIVERS, which must never fall back.
    dev = pick_device(args.device, allow_cpu=True)
    if dev == "mps":
        # NOT mps: envelope/gc/convsi need FFT ops MPS does not implement --
        # run_technique_matrix.py already refuses it for the same reason. Auto-
        # selecting it here turned three PASSes into FAILs on a Mac, which
        # would have read as "the data is bad" rather than "wrong backend".
        dev = "cpu"
    if dev == "cuda":
        p = torch.cuda.get_device_properties(0)
        free_b, total_b = torch.cuda.mem_get_info()
        chk(True, "gpu identity",
            f"{p.name}, {total_b/2**30:.0f} GiB total, "
            f"{free_b/2**30:.0f} GiB free, sm_{p.major}{p.minor}")
        # Rough forward-wavefield footprint for ONE shot: nt x nz x nx x 4 B.
        # An ESTIMATE, labelled as one -- the point is to catch an order-of-
        # magnitude problem before 13 jobs die at hour three, not to predict
        # allocator behaviour to the megabyte.
        est = nt * nz * nx * 4
        chk(est < 0.5 * free_b, "gpu memory headroom",
            f"~{est/2**30:.2f} GiB estimated per-shot wavefield "
            f"(nt={nt} x nz={nz} x nx={nx} x 4B) "
            f"vs {free_b/2**30:.0f} GiB free")
    else:
        print(f"  [note] running on {dev}: the GPU checks are SKIPPED, so this "
              f"does NOT prove the H100 path works. Use --device cuda in a GPU "
              f"job for that.")

    ok_m = True
    for name in ("convsi", "gc", "l2"):
        try:
            m = ConditionedMisfit(config.build_misfit(name, dt=g["dt"],
                                                      iterations=10),
                                  dt=g["dt"], window=True,
                                  window_pre=0.15, window_post=0.5)
            o = torch.tensor(d, dtype=torch.float32, device=dev)
            s = torch.tensor(syn, dtype=torch.float32,
                             device=dev).requires_grad_(True)
            e = m.forward(s, o)
            gr, = torch.autograd.grad(e, s)
            good = (torch.isfinite(e).all() and torch.isfinite(gr).all()
                    and float(gr.abs().max()) > 0)
            chk(good, f"misfit {name}",
                f"loss {float(e):.3e}, |grad|max {float(gr.abs().max()):.3e}")
            ok_m &= good
        except Exception as ex:                          # noqa: BLE001
            chk(False, f"misfit {name}", f"{type(ex).__name__}: {ex}")
            ok_m = False

    bad = [n for ok, n, _ in _R if not ok]
    # Report the login-node CPU budget consumed. This was killed at 2105 CPU-s
    # against an 1800 s limit, and the only reason that was a surprise is that
    # nothing measured it -- so measure it, every run.
    import resource
    ru = resource.getrusage(resource.RUSAGE_SELF)
    cpu = ru.ru_utime + ru.ru_stime
    where = f"on {dev}" if dev != "cpu" else "on the login node"
    print(f"\n=== {len(_R)-len(bad)}/{len(_R)} PASSED ===   "
          f"[{where}, {cpu:.0f} CPU-s]")
    # The 1800 s cap is a LOGIN-NODE limit; inside a job it does not apply, so
    # warning about it there would train the reader to ignore the warning.
    if dev == "cpu" and not os.environ.get("SLURM_JOB_ID") and cpu > 600:
        print("  *** over a third of the 1800 s login-node CPU budget -- "
              "lower --shots before this gets killed ***")
    if bad:
        print("FAILED: " + ", ".join(bad))
        print("DO NOT SUBMIT until these pass.")
        return 1
    print("Safe to submit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
