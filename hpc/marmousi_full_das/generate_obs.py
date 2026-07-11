"""Generate the shared observed DAS strain-rate data (run ONCE, before the
campaign; every misfit x optimizer run loads the same file).

    python hpc/marmousi_full_das/generate_obs.py [--device cuda|mps|cpu]

Runs Liu's true Marmousi2 section through the acoustic propagator and
records strain rate on the merged fiber geometry via DASObservationLayer
(inverse crime by construction). Output:

    $DASFWI_RESULTS/obs_data_das.npz   (SeismicData with key "strain_rate")
    $DASFWI_RESULTS/setup.npz          (vp_true, vp_init, geometry summary)
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import (OUT_ROOT, OBS_FILE, NT, NZ, NX, pick_device, load_models,
                    build_model, build_geometry, build_survey,
                    DASObservationLayer, SeismicData, AcousticPropagator)

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--checkpoint-segments", type=int, default=4)
    args = ap.parse_args()
    device = pick_device(args.device)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    vp_true, vp_init = load_models()
    print(f"vp_true {vp_true.shape} {vp_true.min():.0f}-{vp_true.max():.0f}; "
          f"device {device}", flush=True)

    geometry = build_geometry()
    survey = build_survey(geometry)
    layer = DASObservationLayer(geometry,
                                output="strain_rate").to(torch.float32).to(device)
    print(f"fibers: C = {geometry.C} channels, n_rcv = {geometry.n_rcv}",
          flush=True)

    true_model = build_model(vp_true, vp_bound=None, vp_grad=False,
                             device=device)
    prop = AcousticPropagator(true_model, survey, device=device,
                              dtype=torch.float32)
    t0 = time.time()
    with torch.no_grad():
        rec = prop.forward(checkpoint_segments=args.checkpoint_segments)
        obs_sr = layer(rec["u"], rec["w"]).cpu()
    print(f"forward {time.time()-t0:.0f}s; obs {tuple(obs_sr.shape)}, "
          f"max|.| {float(obs_sr.abs().max()):.3e}, "
          f"finite: {bool(torch.isfinite(obs_sr).all())}", flush=True)
    assert torch.isfinite(obs_sr).all()

    obs_data = SeismicData(survey)
    obs_data.record_data({"strain_rate": obs_sr})
    obs_data.save(str(OUT_ROOT / OBS_FILE))

    np.savez(OUT_ROOT / "setup.npz",
             vp_true=vp_true, vp_init=vp_init,
             channel_z=geometry.channel_z,
             fiber_of_channel=geometry.fiber_of_channel.numpy(),
             rcv_pos=np.asarray(geometry.rcv_pos))
    print("saved:", OUT_ROOT / OBS_FILE, "and setup.npz", flush=True)


if __name__ == "__main__":
    main()
