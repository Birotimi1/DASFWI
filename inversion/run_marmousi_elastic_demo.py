"""Downscaled LOCAL Marmousi2 ELASTIC (Vp + Vs) DAS inverse-crime demo.

User-directed extension beyond the v6b local scope (elastic DAS is Phase F
there): invert BOTH Vp and Vs from vertical-fiber DAS strain rate on a
structurally rich Marmousi2 crop.

Physics note: for a STRAIGHT VERTICAL fiber the E3 endpoint-difference
operator eps_zz_bar_dot = [v_z(z+l/2) - v_z(z-l/2)] / l is exact in elastic
media too - the E5 pointwise machinery is only required for curved fibers.
ADFWI's ElasticPropagator returns "vx"/"vz" records [S, T, R], which feed
DASObservationLayer directly.

Upstream discipline: ADFWI/fwi/elastic_fwi.py is READ-ONLY (spec rule 3), so
this script carries its own compact inversion loop (forward -> DAS layer ->
low-pass -> GC misfit -> autograd -> AdamW) instead of patching ElasticFWI.

Setup:
- Crop: x = 10-11.5 km, z = 455-1455 m (5 m below the water bottom so
  Vs > 0 everywhere), decimated exactly 4x -> 5 m grid, 201 x 301.
- Initial models: TRUE Vp and Vs smoothed with a HEAVY 180 x 180 m Gaussian
  (sigma = 180 m in x and z = 36 cells, per user direction) - essentially a
  smooth regional trend with no interfaces or faults left. Rho fixed at
  TRUE rho (pure inverse crime in Vp/Vs).
- 6 surface moment-tensor (explosive) shots, Ricker f0 = 10 Hz; vertical
  fiber at x = 750 m, 100 channels, z = 200-695 m (model depths).
- Multiscale: 5 Hz band then 10 Hz; GC misfit (float64-safe subclass);
  AdamW(weight_decay=0). float32 like ADFWI's published runs.

Results -> results/marmousi_elastic_demo/.
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from ADFWI.model import IsotropicElasticModel
from ADFWI.propagator import ElasticPropagator
from ADFWI.utils.velocityDemo import load_marmousi_model
from ADFWI.fwi.multiScaleProcessing import lpass

from das.geometry import FiberGeometry
from das.das_layer import DASObservationLayer
from forge.proxy_model import vibroseis_line, build_survey
from inversion.run_inverse_crime import GCMisfit64

MARMOUSI_DIR = os.path.join(os.path.dirname(__file__), "..", "..",
                            "Data_downloads", "marmousi2")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results",
                       "marmousi_elastic_demo")

X0_M, Z0_M = 10000.0, 455.0        # 455: one node below the water bottom
NZ, NX = 201, 301
STEP = 4
DXZ = 5.0

SMOOTH_SIGMA_M = 180.0             # user-specified: sigma = 180 m in x AND z
SIGMA_CELLS = SMOOTH_SIGMA_M / DXZ           # heavy smooth: near-1D trend

# Runs on Apple-Silicon GPU (torch MPS): validated against CPU (GC-misfit
# vp/vs gradients agree to cosine 1.0000), ~4x faster. 4 shots, 0.88 s records
# (P + shallow S arrivals; deep S at vs ~ 300 m/s is out of local reach).
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
NT, DT, F0 = 2200, 4e-4, 10.0
SHOT_X = (20, 105, 190, 280)
BANDS = [(5.0, 150), (10.0, 150)]  # (cutoff_freq_Hz, iterations): 300 total
LR = 5.0
CHECKPOINT_SEGMENTS = 12
SAVE_EVERY = 10                    # checkpoint npz every N iterations


def load_crops():
    m = load_marmousi_model(MARMOUSI_DIR)
    ix0, iz0 = int(X0_M / 1.25), int(Z0_M / 1.25)
    sl = (slice(ix0, ix0 + (NX - 1) * STEP + 1, STEP),
          slice(iz0, iz0 + (NZ - 1) * STEP + 1, STEP))
    vp = np.asarray(m["vp"][sl].T, dtype=np.float64)
    vs = np.asarray(m["vs"][sl].T, dtype=np.float64)
    rho = np.asarray(m["rho"][sl].T, dtype=np.float64)
    assert vs.min() > 0, "crop touches the water layer (vs = 0)"
    return vp, vs, rho


def make_model(vp, vs, rho, grad=False):
    return IsotropicElasticModel(
        0, 0, NX, NZ, DXZ, DXZ, vp=vp, vs=vs, rho=rho,
        vp_bound=(1200.0, 4000.0) if grad else None,
        vs_bound=(150.0, 2300.0) if grad else None,
        vp_grad=grad, vs_grad=grad, rho_grad=False,
        auto_update_rho=False, free_surface=False,
        abc_type="PML", nabc=20, device=DEVICE, dtype=torch.float32)


def das_forward(prop, model, layer):
    rec = prop.forward(model=model, checkpoint_segments=CHECKPOINT_SEGMENTS)
    # .cpu() is differentiable; lpass/misfit run on CPU (lpass needs numpy)
    return layer(rec["vx"], rec["vz"]).cpu()


class GCSafe(GCMisfit64):
    """GC misfit with trace-norm divisions clamped: near-zero traces (e.g.
    quiet channels after a low cut) otherwise underflow the float32 norm and
    poison the loss with NaN."""

    def forward(self, obs, syn):
        rsd = torch.zeros((obs.shape[0], obs.shape[2]),
                          device=obs.device, dtype=obs.dtype)
        for itr in range(obs.shape[2]):
            o = obs[:, :, itr]
            s = syn[:, :, itr]
            o = o / o.norm(dim=1, keepdim=True).clamp_min(1e-20)
            s = s / s.norm(dim=1, keepdim=True).clamp_min(1e-20)
            cov = torch.mean(o * s, dim=1)
            corr = cov / (torch.sqrt(torch.var(o, dim=1)
                                     * torch.var(s, dim=1)) + 1e-8)
            rsd[:, itr] = -corr
        return torch.sum(rsd * self.dt)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    vp_true, vs_true, rho_true = load_crops()
    vp_init = gaussian_filter(vp_true, sigma=SIGMA_CELLS)
    vs_init = gaussian_filter(vs_true, sigma=SIGMA_CELLS)
    print(f"vp {vp_true.min():.0f}-{vp_true.max():.0f}, "
          f"vs {vs_true.min():.0f}-{vs_true.max():.0f}, "
          f"smoothing sigma = {SIGMA_CELLS:.0f} cells "
          f"({SMOOTH_SIGMA_M:.0f} m)", flush=True)

    geometry = FiberGeometry(x_well=750.0, z_top=200.0, n_channels=100,
                             snap_to_nodes=False)
    layer = DASObservationLayer(geometry,
                                output="strain_rate").to(torch.float32).to(DEVICE)
    source = vibroseis_line(NT, DT, F0, SHOT_X, 2)
    survey = build_survey(source, geometry)
    print("device:", DEVICE, flush=True)

    # observed data: TRUE elastic model through the same operator
    true_model = make_model(vp_true, vs_true, rho_true)
    prop = ElasticPropagator(true_model, survey, device=DEVICE,
                             dtype=torch.float32)
    t0 = time.time()
    with torch.no_grad():
        obs = das_forward(prop, true_model, layer)
    print(f"observed data generated in {time.time()-t0:.0f}s, "
          f"max|obs| = {obs.abs().max():.3e}", flush=True)
    assert torch.isfinite(obs).all()

    # inversion loop (custom: ElasticFWI is read-only upstream)
    model = make_model(vp_init, vs_init, rho_true, grad=True)
    optimizer = torch.optim.AdamW(
        [p for p in (model.vp, model.vs)], lr=LR, weight_decay=0.0)
    misfit = GCSafe(dt=1)
    fs = int(1.0 / DT)

    losses = []
    for cutoff, n_iter in BANDS:
        for it in range(n_iter):
            t_it = time.time()
            optimizer.zero_grad()
            syn = das_forward(prop, model, layer)
            if cutoff is not None:
                syn_f, obs_f = lpass(syn, obs, cutoff, fs)
            else:
                syn_f, obs_f = syn, obs
            loss = misfit.forward(obs_f, syn_f)
            loss.backward()
            optimizer.step()
            model.forward()          # clip vp/vs to bounds
            losses.append(float(loss))
            print(f"band {cutoff} Hz iter {it}: loss {loss:.5f} "
                  f"({time.time()-t_it:.0f}s)", flush=True)
            if len(losses) % SAVE_EVERY == 0:
                np.savez(os.path.join(OUT_DIR, "checkpoint.npz"),
                         vp=model.vp.detach().cpu().numpy(),
                         vs=model.vs.detach().cpu().numpy(),
                         losses=np.asarray(losses))

    vp_final = model.vp.detach().cpu().numpy().copy()
    vs_final = model.vs.detach().cpu().numpy().copy()

    for name, true, init, final in (("vp", vp_true, vp_init, vp_final),
                                    ("vs", vs_true, vs_init, vs_final)):
        e0 = np.sqrt(np.mean((init - true) ** 2))
        e1 = np.sqrt(np.mean((final - true) ** 2))
        c = np.corrcoef((true - init).ravel(), (final - init).ravel())[0, 1]
        print(f"{name}: RMS error {e0:.1f} -> {e1:.1f} m/s, "
              f"update correlation {c:.3f}", flush=True)

    np.savez(os.path.join(OUT_DIR, "result.npz"),
             vp_true=vp_true, vs_true=vs_true, vp_init=vp_init,
             vs_init=vs_init, vp_final=vp_final, vs_final=vs_final,
             losses=np.asarray(losses))

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), constrained_layout=True)
    ext = [0, (NX - 1) * DXZ / 1000, (NZ - 1) * DXZ / 1000, 0]
    rows = [("vp", vp_true, vp_init, vp_final),
            ("vs", vs_true, vs_init, vs_final)]
    for r, (name, true, init, final) in enumerate(rows):
        for c, (data, ttl) in enumerate([(true, "true"), (init, "initial"),
                                         (final, "inverted")]):
            im = axes[r, c].imshow(data, extent=ext, cmap="jet",
                                   vmin=true.min(), vmax=true.max())
            axes[r, c].set(title=f"{name} {ttl} [m/s]", xlabel="x [km]",
                           ylabel="z [km]")
            fig.colorbar(im, ax=axes[r, c], shrink=0.8)
    fig.savefig(os.path.join(OUT_DIR, "marmousi_elastic_demo.png"), dpi=150)

    fig2, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(losses, "k.-")
    ax.axvline(BANDS[0][1] - 0.5, color="r", ls="--", label="5 -> 10 Hz")
    ax.set(title="GC loss", xlabel="iteration")
    ax.legend()
    fig2.savefig(os.path.join(OUT_DIR, "loss.png"), dpi=150)
    print("saved figures to", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
