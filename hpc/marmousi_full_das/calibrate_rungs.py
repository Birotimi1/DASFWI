"""Phase 1a: calibrate the starting-model ladder BEFORE spending the grid.

For each candidate rung this runs ONE forward model (no inversion) against the
shared observed data and reports the cycle-skip diagnostic. The point is to pick
the 4-5 rungs that BRACKET the skip threshold, instead of burning 45 jobs on a
rung that is 0% skipped (nothing to learn) or 100% skipped (everything fails).

Why this matters: a vertical-traveltime estimate on a Marmousi-like proxy puts
the transition between sigma=16 (~4% of channels skipped) and sigma=24 (~96%),
with sigma>=32 fully saturated. That is a SHARP transition, and it is model
dependent - so measure it on the real model here rather than trusting the proxy.

    python hpc/marmousi_full_das/calibrate_rungs.py            # all rungs
    python hpc/marmousi_full_das/calibrate_rungs.py --rungs s6,s12,s16,s20,s24

Requires generate_obs.py to have produced $DASFWI_RESULTS/obs_data_das.npz.
Cost: one forward per rung (~1-2 min each on a GPU), not 300 iterations.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import (OUT_ROOT, OBS_FILE, DT, F0, NT, START_RUNGS,
                    MISFIT_RUN_SETTINGS, pick_device, load_models, build_model,
                    build_geometry, build_survey, DASObservationLayer,
                    SeismicData, AcousticPropagator)

import numpy as np
import torch

from inversion.skip_diagnostic import skip_fraction, ricker_f90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", default=",".join(START_RUNGS),
                    help="comma-separated rungs to test")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None,
                    help="JSON output (default <results>/rung_calibration.json)")
    args = ap.parse_args()
    rungs = [r.strip() for r in args.rungs.split(",") if r.strip()]
    for r in rungs:
        if r not in START_RUNGS:
            raise SystemExit(f"unknown rung {r!r}; choices {START_RUNGS}")

    device = pick_device(args.device)
    dtype = torch.float32
    f90 = ricker_f90(F0, DT, NT, integrated=True)
    thr = 1.0 / (2 * f90)
    print(f"device={device}  source f90={f90:.2f} Hz  -> skip threshold "
          f"T/2={1000*thr:.0f} ms  (nominal f0={F0} would give "
          f"{1000/(2*F0):.0f} ms)", flush=True)

    geometry = build_geometry()
    survey = build_survey(geometry)
    layer = DASObservationLayer(geometry, output="strain_rate").to(dtype).to(device)
    obs_data = SeismicData(survey)
    obs_data.load(str(OUT_ROOT / OBS_FILE))
    obs = torch.as_tensor(obs_data.data["strain_rate"])
    print(f"observed {tuple(obs.shape)}  max|.|={float(obs.abs().max()):.3e}\n",
          flush=True)

    seg = MISFIT_RUN_SETTINGS["l2"]["checkpoint_segments"]
    rows = []
    hdr = (f"  {'rung':>7}{'RMS(vp)':>9}{'skip%':>8}{'mean|lag|':>11}"
           f"{'p90|lag|':>10}{'corr':>7}   verdict")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for rung in rungs:
        vp_true, vp_init = load_models(rung)
        model = build_model(vp_init, vp_bound=None, vp_grad=False,
                            device=device, dtype=dtype)
        prop = AcousticPropagator(model, survey, device=device, dtype=dtype)
        t0 = time.time()
        with torch.no_grad():
            rec = prop.forward(checkpoint_segments=seg)
            syn = layer(rec["u"], rec["w"]).cpu()
        st = skip_fraction(syn, obs, DT, f90)
        st["rung"] = rung
        st["rms_vp"] = float(np.sqrt(((vp_init - vp_true) ** 2).mean()))
        st["forward_s"] = round(time.time() - t0, 1)
        rows.append(st)
        sf = st["skip_fraction"]
        verdict = ("no-skip (baseline)" if sf < 0.05 else
                   "TRANSITION  <-- informative" if sf < 0.85 else
                   "saturated (all skipped)")
        print(f"  {rung:>7}{st['rms_vp']:9.0f}{100*sf:7.0f}%"
              f"{1000*st['mean_abs_lag_s']:10.0f}ms"
              f"{1000*st['p90_abs_lag_s']:9.0f}ms{st['mean_peak']:7.2f}   {verdict}",
              flush=True)

    out = args.out or str(OUT_ROOT / "rung_calibration.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(dict(f90_hz=f90, threshold_s=thr, rungs=rows), f, indent=2)
    print(f"\nwrote {out}")

    trans = [r["rung"] for r in rows if 0.05 <= r["skip_fraction"] < 0.85]
    msg = (", ".join(trans) if trans else
           "NONE - widen the ladder (add rungs between the last no-skip and "
           "the first saturated rung)")
    print("\nRECOMMENDATION: run the 45-combo grid on the rungs that bracket "
          "the threshold.\n  transition rungs found: " + msg)


if __name__ == "__main__":
    main()
