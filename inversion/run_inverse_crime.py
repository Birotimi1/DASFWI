"""Config-driven inverse-crime DAS FWI (T7).

Wires the stock (T5-patched) AcousticFWI with das_layer + obs_key =
"strain_rate", AdamW(weight_decay=0.0), a GC or scheduled misfit, and
multiscale bands via successive fwi.forward(cutoff_freq=f) calls with
start_iter bookkeeping.

Usage (programmatic):
    from inversion.run_inverse_crime import run_inverse_crime
    result = run_inverse_crime(config)

Config keys (all plain Python; see DEFAULT_CONFIG for a miniature example):
    vp_true, vp_init : [nz, nx] float arrays (true and starting models)
    fiber            : dict(x_well, z_top, n_channels)  (synthetic mode)
    shots            : dict(x_indices, z_index, f0)
    nt, dt           : time sampling
    lr               : AdamW learning rate (weight_decay is always 0.0)
    misfit           : "gc"  -> global correlation (float64-safe)
                       or a ready ADFWI-Misfit-like instance (e.g. a
                       dasfwi ScheduledMisfit)
    bands            : list of (cutoff_freq_or_None, n_iterations) tuples,
                       run in order with start_iter bookkeeping
    batch_size       : shots per batch (None = all)
    checkpoint_segments, device, dtype, nabc

Full-scale campaigns (Marmousi2 reproduction, FORGE proxy + degradation
ladder) are HPC work - do NOT run them locally (spec section 3).
"""

import numpy as np
import torch

from ADFWI.propagator import AcousticPropagator, GradProcessor
from ADFWI.fwi import AcousticFWI
from ADFWI.fwi.misfit import Misfit_global_correlation

from das.geometry import FiberGeometry
from forge.proxy_model import (make_acoustic_model, vibroseis_line,
                               generate_observed, DX, DZ, GAUGE_L, DCH)


class GCMisfit64(Misfit_global_correlation):
    """Global-correlation misfit with the accumulator dtype following the
    inputs. Upstream hard-codes float32 (GlobalCorrelation.py: torch.zeros
    without dtype) and crashes on float64 gathers; math identical.
    """

    def forward(self, obs, syn):
        mask1 = torch.sum(torch.abs(obs), axis=1) == 0
        mask2 = torch.sum(torch.abs(syn), axis=1) == 0
        mask = ~(mask1 * mask2)

        rsd = torch.zeros((obs.shape[0], obs.shape[2]),
                          device=obs.device, dtype=obs.dtype)
        for itrace in range(obs.shape[2]):
            shot_idx = torch.argwhere(mask[:, itrace])
            obs_trace = obs[shot_idx, :, itrace].squeeze(axis=1)
            syn_trace = syn[shot_idx, :, itrace].squeeze(axis=1)

            obs_trace = obs_trace / obs_trace.norm(dim=1, keepdim=True)
            syn_trace = syn_trace / syn_trace.norm(dim=1, keepdim=True)

            cov = torch.mean(obs_trace * syn_trace, dim=1)
            var_obs = torch.var(obs_trace, dim=1)
            var_syn = torch.var(syn_trace, dim=1)

            corr = cov / (torch.sqrt(var_obs * var_syn) + 1e-8)
            corr[torch.isnan(corr)] = 0
            rsd[shot_idx, itrace] = -corr.reshape(-1, 1)

        return torch.sum(rsd * self.dt)


DEFAULT_CONFIG = dict(
    # miniature: 101 x 101, 4 shots, 20 iterations (local CPU scale)
    nt=1200, dt=4e-4,
    fiber=dict(x_well=300.0, z_top=100.0, n_channels=20),
    shots=dict(x_indices=(15, 35, 55, 75), z_index=2, f0=10.0),
    lr=2.0,
    misfit="gc",
    bands=[(None, 20)],
    batch_size=None,
    checkpoint_segments=10,
    device="cpu",
    dtype=torch.float64,
    nabc=20,
    vp_bound=None,          # (min, max) clip for the inverted vp, or None
    rho=None,               # fixed density [nz, nx]; None -> Gardner(vp_true)
    optimizer="adamw",      # "adamw" (spec default) or "sgd".
    # "sgd" + GradProcessor(norm_grad=True) is the classic FWI update: the
    # processed gradient is vmax * g/|g|max, so the peak cell moves lr*vmax
    # m/s per iteration and weakly-illuminated cells move proportionally
    # less. AdamW's per-parameter normalization instead moves EVERY cell at
    # ~lr per iteration, amplifying gradient noise in unilluminated regions
    # into full-size updates (diagnosed on the Marmousi single-fiber demo:
    # uniform ~100 m/s |update| everywhere, RMS worse far from the fiber),
    # and silently defeats the illumination preconditioner.
)


def run_inverse_crime(config):
    """Run a full inverse-crime DAS inversion per the config; returns a dict
    with iter_loss, vp_init, vp_final, vp_true, delta_vp and the fwi object."""
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(config)

    vp_true = np.asarray(cfg["vp_true"], dtype=np.float64)
    vp_init = np.asarray(cfg["vp_init"], dtype=np.float64)
    device, dtype = cfg["device"], cfg["dtype"]

    geometry = FiberGeometry(x_well=cfg["fiber"]["x_well"],
                             z_top=cfg["fiber"]["z_top"],
                             n_channels=cfg["fiber"]["n_channels"],
                             dch=DCH, l=GAUGE_L, dx=DX, dz=DZ,
                             snap_to_nodes=False)
    source = vibroseis_line(cfg["nt"], cfg["dt"], cfg["shots"]["f0"],
                            cfg["shots"]["x_indices"], cfg["shots"]["z_index"])

    # ONE fixed density for BOTH models: deriving obs-rho from vp_true but
    # inversion-rho from vp_init injects a systematic amplitude error that vp
    # must absorb (documented failure of the first acoustic Marmousi demo)
    from forge.proxy_model import gardner_rho
    rho_fixed = (gardner_rho(vp_true) if cfg["rho"] is None
                 else np.asarray(cfg["rho"], dtype=np.float64))

    # observed data: TRUE model through the same operator (inverse crime)
    true_model = make_acoustic_model(vp_true, nabc=cfg["nabc"],
                                     device=device, dtype=dtype,
                                     rho=rho_fixed)
    obs_data, survey, layer = generate_observed(
        true_model, geometry, source,
        checkpoint_segments=cfg["checkpoint_segments"],
        device=device, dtype=dtype)

    # inversion
    model = make_acoustic_model(vp_init, vp_grad=True, nabc=cfg["nabc"],
                                device=device, dtype=dtype,
                                vp_bound=cfg["vp_bound"], rho=rho_fixed)
    prop = AcousticPropagator(model, survey, device=device, dtype=dtype)
    if cfg["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                      weight_decay=0.0)
    elif cfg["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"])
    else:
        raise ValueError(f'unknown optimizer {cfg["optimizer"]!r}')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10 ** 6,
                                                gamma=1.0)
    loss_fn = GCMisfit64(dt=1) if cfg["misfit"] == "gc" else cfg["misfit"]

    fwi = AcousticFWI(prop, model, optimizer, scheduler,
                      loss_fn=loss_fn, obs_data=obs_data,
                      gradient_processor=GradProcessor(),
                      waveform_normalize=True,
                      cache_result=True, save_fig_epoch=-1,
                      das_layer=layer, obs_key="strain_rate")

    # multiscale bands with start_iter bookkeeping
    start_iter = 0
    for cutoff_freq, n_iter in cfg["bands"]:
        fwi.forward(iteration=n_iter, batch_size=cfg["batch_size"],
                    checkpoint_segments=cfg["checkpoint_segments"],
                    start_iter=start_iter, cutoff_freq=cutoff_freq)
        start_iter += n_iter

    vp_final = model.vp.detach().cpu().numpy().copy()
    return dict(iter_loss=list(fwi.iter_loss), vp_init=vp_init,
                vp_final=vp_final, vp_true=vp_true,
                delta_vp=vp_final - vp_init, fwi=fwi)


if __name__ == "__main__":
    # miniature demonstration run (local CPU scale)
    nz = nx = 101
    zz, xx = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
    bump = 150.0 * np.exp(-(((zz - 30) ** 2 + (xx - 40) ** 2) / (2 * 8.0 ** 2)))
    result = run_inverse_crime(dict(
        vp_true=np.full((nz, nx), 2000.0) + bump,
        vp_init=np.full((nz, nx), 2000.0)))
    print("losses:", [f"{l:.6f}" for l in result["iter_loss"]])
    print("max |delta vp|:", np.abs(result["delta_vp"]).max())
