"""Portable, numerically-hardened versions of ADFWI misfits.

Each class is MATHEMATICALLY IDENTICAL to its upstream parent; the changes
fix bugs that Liu's CUDA-only usage masks, plus numerics that only downhole
DAS gathers exercise. Use these everywhere (they behave exactly like the
stock classes on CUDA):

GCMisfit64     - global correlation. Upstream hard-codes a float32 residual
                 accumulator (torch.zeros without dtype) and crashes on
                 float64 input on CPU; here the accumulator follows the
                 input dtype.
SinkhornSafe   - Wasserstein sinkhorn divergence. Upstream (a) mixes a
                 float64 time-coordinate tensor into input-dtype clouds and
                 crashes in every precision on CPU, and (b) presumes every
                 trace carries signal. This version follows the input dtype
                 throughout, applies ONE detached global scale per shot
                 (run it with waveform_normalize=False: per-trace
                 max-normalization backward overflows float32 on
                 numerically-dead fiber traces), and drops traces below
                 1e-3 of the gather peak on EITHER side (dead traces carry
                 no information and poison the transport problem).
                 Parameters: use Liu's working configuration
                 (dt=0.01, sparse_sampling=2, p=1, blur=1e-2).
SdtwSafe       - soft-DTW divergence. Upstream decides use_cuda by
                 comparing a torch.device OBJECT to the string "cpu"
                 (always unequal), demanding CUDA on any build; here
                 use_cuda derives from device.type. pysdtw supports cuda
                 and cpu only (no Apple MPS).
"""

import torch

import pysdtw
from geomloss import SamplesLoss

from ADFWI.fwi.misfit import (Misfit_global_correlation,
                              Misfit_wasserstein_sinkhorn, Misfit_sdtw)


class GCMisfit64(Misfit_global_correlation):
    """Global-correlation misfit; accumulator dtype follows the inputs."""

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


class SinkhornSafe(Misfit_wasserstein_sinkhorn):
    """Sinkhorn divergence with portable dtypes and DAS-safe scaling/masking.
    RUN WITH waveform_normalize=False (see module docstring)."""

    DEAD_TRACE_REL = 1e-3   # ~-60 dB of the gather peak

    def forward(self, obs, syn):
        scale = obs.abs().amax(dim=(1, 2), keepdim=True).detach().clamp_min(1e-300)
        obs = obs / scale
        syn = syn / scale

        o_amp = obs.abs().amax(dim=1)                       # [S, C]
        s_amp = syn.abs().amax(dim=1)
        mask = ((o_amp > self.DEAD_TRACE_REL * o_amp.amax(dim=1, keepdim=True))
                & (s_amp > self.DEAD_TRACE_REL * s_amp.amax(dim=1, keepdim=True)))

        rsd = torch.zeros((obs.shape[0], obs.shape[2]),
                          device=obs.device, dtype=obs.dtype)
        loss_fn = SamplesLoss(loss=self.loss_method, p=self.p,
                              blur=self.blur, scaling=self.scaling)
        for ishot in range(obs.shape[0]):
            idx = torch.argwhere(mask[ishot]).reshape(-1)
            o = obs[ishot, ::self.sparse_sampling, idx].T   # [trace, samples]
            s = syn[ishot, ::self.sparse_sampling, idx].T
            t = (torch.arange(o.shape[1], device=obs.device, dtype=obs.dtype)
                 * self.dt).reshape(1, -1).expand_as(o)
            o2 = torch.stack((t, o), dim=-1)                # [trace, n, 2]
            s2 = torch.stack((t, s), dim=-1)
            rsd[ishot, idx] = loss_fn(o2, s2).reshape(-1).to(obs.dtype)
        return torch.sum(rsd * rsd * self.dt)


class SdtwSafe(Misfit_sdtw):
    """Soft-DTW divergence with a correct device test (cuda or cpu)."""

    def forward(self, obs, syn):
        device = obs.device
        mask = ~((torch.sum(torch.abs(obs), axis=1) == 0)
                 * (torch.sum(torch.abs(syn), axis=1) == 0))
        fun = pysdtw.distance.pairwise_l2_squared_exact
        rsd = torch.zeros((obs.shape[0], obs.shape[2]),
                          device=device, dtype=obs.dtype)
        sdtw = pysdtw.SoftDTW(gamma=self.gamma, dist_func=fun,
                              use_cuda=(device.type == "cuda"))
        for ishot in range(obs.shape[0]):
            trace_idx = torch.argwhere(mask[ishot]).reshape(-1)
            obs_shot = obs[ishot, ::self.sparse_sampling,
                           trace_idx].squeeze().T.unsqueeze(2)
            syn_shot = syn[ishot, ::self.sparse_sampling,
                           trace_idx].squeeze().T.unsqueeze(2)
            sdtw_obs = sdtw(obs_shot, obs_shot)
            sdtw_syn = sdtw(syn_shot, syn_shot)
            sdtw_obs_syn = sdtw(obs_shot, syn_shot)
            std = sdtw_obs_syn - 0.5 * (sdtw_obs + sdtw_syn)
            rsd[ishot, trace_idx] = std.reshape(1, -1).to(rsd.dtype)
        return torch.sum(rsd * self.dt)
