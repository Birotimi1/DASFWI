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
TravelTimeSafe - cross-correlation traveltime misfit (cycle-skipping
                 robust). Upstream hard-codes a float32 residual
                 accumulator AND a float32 lag axis, crashing on float64
                 CPU input; here both follow the input dtype. NOTE: it is
                 O(n_shots * n_receivers) conv1d calls per evaluation and
                 is genuinely slow on dense channel counts -- Liu marks it
                 experimental; decimate channels / shots for it.
make_nim       - Normalized Integration Method (= Wasserstein-1 for p=1),
                 also cycle-skipping robust. Misfit_NIM is a
                 torch.autograd.Function (hand-coded backward), invoked via
                 .apply(syn, obs, p, trans_type, theta); the stock
                 AcousticFWI.calculate_loss already dispatches it, but
                 custom loops must use apply_misfit() below.
apply_misfit   - dispatcher: calls .apply(...) for Misfit_NIM, else
                 .forward(syn, obs). Use it in any custom (non-AcousticFWI)
                 inversion loop so NIM and the plain misfits are uniform.
ConvolvedWavefieldMisfit - SOURCE-INDEPENDENT convolved-wavefields misfit
                 (Choi & Alkhalifah 2011): cross-convolution cancels the
                 unknown source wavelet (and its amplitude) exactly. Carries
                 over to DAS because the operator is linear/time-invariant.
                 Run with waveform_normalize=False; CUDA/CPU (FFT, no MPS).
"""

import torch

import pysdtw
from geomloss import SamplesLoss

from ADFWI.fwi.misfit import (Misfit, Misfit_global_correlation,
                              Misfit_wasserstein_sinkhorn, Misfit_sdtw,
                              Misfit_traveltime, Misfit_NIM)


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


class TravelTimeSafe(Misfit_traveltime):
    """Cross-correlation traveltime misfit; accumulator and lag axis follow
    the input dtype. Same softmax-based differentiable time shift as upstream.
    O(n_shots * n_receivers) conv1d calls -- slow on dense channels."""

    def forward(self, obs, syn):
        import torch.nn.functional as F
        srcn, nt, rcvn = obs.shape
        device, dtype = obs.device, obs.dtype
        obs = obs / torch.max(torch.abs(obs), dim=1, keepdim=True)[0]
        syn = syn / torch.max(torch.abs(syn), dim=1, keepdim=True)[0]
        lags = torch.arange(-nt + 1, nt, device=device, dtype=dtype)
        rsd = torch.zeros((srcn, rcvn), device=device, dtype=dtype)
        for ishot in range(srcn):
            for ircv in range(rcvn):
                w1 = obs[ishot, :, ircv]
                w2 = syn[ishot, :, ircv]
                cc = F.conv1d(w1.view(1, 1, -1), w2.view(1, 1, -1),
                              padding=nt - 1).view(-1)
                weights = F.softmax(self.beta * cc, dim=0)
                rsd[ishot, ircv] = torch.abs((weights * lags).sum() * self.dt)
        return torch.sum(rsd)


class ConvolvedWavefieldMisfit(Misfit):
    """Source-INDEPENDENT convolved-wavefields misfit (Choi & Alkhalifah, 2011,
    Geophysics 76(5) R125-R134).

        E = sum_shot sum_chan || syn_c (*) obs_ref  -  obs_c (*) syn_ref ||^2

    where (*) is convolution in time and obs_ref / syn_ref is the per-shot
    channel-AVERAGE trace (the paper found averaging beats a single reference).
    Writing syn = g_syn (*) s_syn and obs = g_obs (*) s_obs, BOTH cross-terms
    carry the factor s_syn (*) s_obs equally, so it cancels: the misfit is zero
    at the true model (g_syn = g_obs) REGARDLESS of the unknown source wavelet
    -> no source estimation needed, and it also removes the source-amplitude
    dependency. A byproduct (paper): the modeled data low-pass-filters the
    observed, giving implicit low->high frequency continuation.

    DAS reformulation: our observation operator R (endpoint difference / E5
    contraction) is LINEAR and TIME-INVARIANT, so it commutes with the temporal
    source convolution -- d_das = R(g (*) s) = R(g) (*) s = g_das (*) s -- and
    the source-cancellation carries over verbatim to strain-rate channel
    gathers. Apply this misfit to the DAS gathers exactly as to pressure.

    Notes:
    - Run with waveform_normalize=False to preserve the exact source-
      independence (per-trace max-normalization perturbs the cancellation).
    - It cancels the SOURCE WAVELET (common to all traces); a per-CHANNEL
      amplitude miscalibration is a separate, smaller effect.
    - FFT convolution -> CUDA/CPU only (torch rfft is unavailable on Apple MPS,
      like the envelope/weci misfits).
    """

    def __init__(self, dt=1):
        super().__init__()
        self.dt = dt

    def forward(self, syn, obs):
        # syn, obs: [S, T, C]. AcousticFWI passes (syn, obs) so obs (fixed
        # observed data) sets the global scale below -> a constant, clean loss.
        S, T, C = syn.shape
        # Global detached rescale to O(1): raw DAS strain rate is ~1e-13, and
        # the convolution-then-square here is quartic in amplitude (~1e-52),
        # which underflows float32 to zero -> the gradient normalization then
        # divides 0/0 -> NaN. A single constant scale (same for syn and obs)
        # is uniform, so it preserves source-independence and only fixes the
        # dynamic range.
        scale = obs.abs().amax().detach().clamp_min(1e-30)
        syn = syn / scale
        obs = obs / scale
        syn_ref = syn.mean(dim=2, keepdim=True)          # [S, T, 1] channel avg
        obs_ref = obs.mean(dim=2, keepdim=True)
        n = 2 * T - 1                                    # full linear conv length
        Fs = torch.fft.rfft(syn, n=n, dim=1)
        Fo = torch.fft.rfft(obs, n=n, dim=1)
        Fsr = torch.fft.rfft(syn_ref, n=n, dim=1)
        For = torch.fft.rfft(obs_ref, n=n, dim=1)
        term1 = torch.fft.irfft(Fs * For, n=n, dim=1)    # syn_c (*) obs_ref
        term2 = torch.fft.irfft(Fo * Fsr, n=n, dim=1)    # obs_c (*) syn_ref
        r = term1 - term2
        return (r * r).sum() * self.dt


def make_nim(p=1, trans_type="linear", theta=1.0, dt=1.0):
    """A configured Misfit_NIM (Normalized Integration Method; = Wasserstein-1
    when p=1). It is a torch.autograd.Function -- pass the returned instance as
    a loss_fn to AcousticFWI (calculate_loss dispatches it) or, in custom
    loops, evaluate it via apply_misfit()."""
    return Misfit_NIM(p=p, trans_type=trans_type, theta=theta, dt=dt)


def apply_misfit(misfit, syn, obs):
    """Evaluate any misfit uniformly in a custom loop. Misfit_NIM is an
    autograd.Function invoked via .apply(syn, obs, p, trans_type, theta);
    everything else is a Misfit with .forward(syn, obs)."""
    if isinstance(misfit, Misfit_NIM):
        return misfit.apply(syn, obs, misfit.p, misfit.trans_type, misfit.theta)
    return misfit.forward(syn, obs)
