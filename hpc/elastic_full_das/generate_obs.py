"""Generate the shared observed DAS strain-rate data for the elastic campaign.

ONE full elastic forward of the TRUE Marmousi2 Vp/Vs/density through the 4
vertical DAS fibers. All 45 combos read the result, so this runs ONCE (on a
GPU node) before the campaign.

    python hpc/elastic_full_das/generate_obs.py [--device cuda|cpu]

Writes into $DASFWI_RESULTS (default results/elastic_full_das/):
    obs_data_das.npz   key "strain_rate": (n_shots, nt, n_channels)
    setup.npz          vp/vs/rho true + init, channel_z, rcv_pos
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import (OUT_ROOT, OBS_FILE, FD_ORDER, CHECKPOINT_SEGMENTS,
                    pick_device, load_models, build_model, build_acquisition,
                    ElasticPropagator)

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = pick_device(args.device)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    vp_true, vs_true, rho_true, vp_init, vs_init, rho_init = load_models()
    print(f"vp_true {vp_true.shape} {vp_true.min():.0f}-{vp_true.max():.0f}; "
          f"vs {vs_true.min():.0f}-{vs_true.max():.0f}; "
          f"rho {rho_true.min():.0f}-{rho_true.max():.0f}; device {device}",
          flush=True)

    survey, layer, geometry = build_acquisition(device)
    print(f"fibers: C = {geometry.C} channels, n_rcv = {geometry.n_rcv}, "
          f"shots = {survey.source.num}", flush=True)

    true_model = build_model(vp_true, vs_true, rho_true, (None, None, None),
                             grad=False, device=device)
    prop = ElasticPropagator(true_model, survey, device=device,
                             dtype=torch.float32)
    t0 = time.time()
    with torch.no_grad():
        rec = prop.forward(model=true_model, fd_order=FD_ORDER,
                           checkpoint_segments=CHECKPOINT_SEGMENTS)
        obs_sr = layer(rec["vx"], rec["vz"]).cpu()
    print(f"forward {time.time()-t0:.0f}s; obs {tuple(obs_sr.shape)}, "
          f"max|.| {float(obs_sr.abs().max()):.3e}, "
          f"finite: {bool(torch.isfinite(obs_sr).all())}", flush=True)
    assert torch.isfinite(obs_sr).all()

    np.savez(OUT_ROOT / OBS_FILE, strain_rate=obs_sr.numpy())
    np.savez(OUT_ROOT / "setup.npz",
             vp_true=vp_true, vs_true=vs_true, rho_true=rho_true,
             vp_init=vp_init, vs_init=vs_init, rho_init=rho_init,
             channel_z=geometry.channel_z,
             fiber_of_channel=geometry.fiber_of_channel.numpy(),
             rcv_pos=np.asarray(geometry.rcv_pos))
    print("saved:", OUT_ROOT / OBS_FILE, "and setup.npz", flush=True)


if __name__ == "__main__":
    main()
