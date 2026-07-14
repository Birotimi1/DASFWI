"""Tests for the source-independent convolved-wavefields misfit (Choi &
Alkhalifah 2011), including its behaviour on DAS strain-rate gathers."""

import numpy as np
import pytest
import torch

from das.geometry import FiberGeometry
from das.das_layer import DASObservationLayer
from inversion.safe_misfits import ConvolvedWavefieldMisfit


def _conv_time(g, s):
    """Convolve every trace g[...,T,...] along time (dim=1) with wavelet s[T]."""
    T = g.shape[1]
    n = 2 * T - 1
    G = torch.fft.rfft(g, n=n, dim=1)
    S = torch.fft.rfft(s.reshape(1, -1, 1), n=n, dim=1)
    return torch.fft.irfft(G * S, n=n, dim=1)[:, :T, :]


def ricker(T, f0, dt, t0):
    t = torch.arange(T) * dt - t0
    a = (np.pi * f0) ** 2
    return (1 - 2 * a * t ** 2) * torch.exp(-a * t ** 2)


def reflectivity(S, T, C, n_spikes=4, seed=0):
    """Compact CAUSAL earth response: sparse spikes in the first third of the
    trace, so g (*) wavelet is fully contained in [0, T) and no truncation
    artifact corrupts the convolution algebra (real traces are causal/decaying;
    full-length white noise would break it)."""
    gen = torch.Generator().manual_seed(seed)
    g = torch.zeros(S, T, C, dtype=torch.float64)
    for s_ in range(S):
        for c in range(C):
            idx = torch.randint(20, T // 3, (n_spikes,), generator=gen)
            g[s_, idx, c] = torch.randn(n_spikes, dtype=torch.float64,
                                        generator=gen)
    return g


def test_source_independence_same_model_different_wavelet():
    # SAME earth response g, DIFFERENT source wavelets -> convolved misfit ~0
    # (source cancels to machine precision), while plain L2 is large.
    T, dt = 400, 1e-3
    g = reflectivity(2, T, 12, seed=0)                     # shared "Green fn"
    s_obs = ricker(T, 25.0, dt, 0.05).double()
    s_syn = ricker(T, 15.0, dt, 0.08).double()             # different f0 AND t0
    obs = _conv_time(g, s_obs)
    syn = _conv_time(g, s_syn)                             # same g, wrong source

    conv = ConvolvedWavefieldMisfit(dt=1)
    e_si = float(conv.forward(syn, obs))
    e_si_wrongmodel = float(conv.forward(
        _conv_time(reflectivity(2, T, 12, seed=99), s_syn), obs))
    l2 = float(((syn - obs) ** 2).sum())

    assert e_si < 1e-8 * l2, f"not source independent: e_si={e_si:.3e} l2={l2:.3e}"
    assert e_si < 1e-8 * e_si_wrongmodel, (
        f"e_si {e_si:.3e} not << wrong-model {e_si_wrongmodel:.3e}")


def test_source_cancellation_either_arg_order():
    # The meaningful invariant: source cancellation holds whichever argument is
    # observed (the global scale uses the 2nd arg, so the loss magnitude is
    # arg-dependent by a benign constant, but the ZERO at the true model is not).
    T, dt = 400, 1e-3
    g = reflectivity(2, T, 10, seed=1)
    obs = _conv_time(g, ricker(T, 24.0, dt, 0.05).double())
    syn = _conv_time(g, ricker(T, 14.0, dt, 0.09).double())   # same g, wrong src
    conv = ConvolvedWavefieldMisfit(dt=1)
    ref = float(conv.forward(reflectivity(2, T, 10, seed=42), obs))  # wrong model
    for a, b in ((syn, obs), (obs, syn)):
        e = float(conv.forward(a, b))
        assert e < 1e-8 * ref, f"source not cancelled (order {a is syn}): {e:.3e}"


def test_gradient_flows():
    torch.manual_seed(2)
    syn = torch.randn(2, 200, 8, dtype=torch.float64, requires_grad=True)
    obs = torch.randn(2, 200, 8, dtype=torch.float64)
    ConvolvedWavefieldMisfit(dt=1).forward(syn, obs).backward()
    assert torch.isfinite(syn.grad).all() and syn.grad.abs().max() > 0


def test_source_independent_through_das_operator():
    # THE DAS point: apply the SI misfit to strain-rate gathers built through
    # the E3 operator. Same velocity wavefield, different source -> SI misfit
    # ~0, proving the operator's linearity/time-invariance preserves source
    # cancellation on DAS data.
    geom = FiberGeometry(x_well=250.0, z_top=100.0, n_channels=15,
                         l=10.0, dx=5.0, dz=5.0, snap_to_nodes=False)
    layer = DASObservationLayer(geom)
    S, T, R, dt = 1, 400, geom.n_rcv, 1e-3
    w = reflectivity(S, T, R, seed=3)                      # compact velocity records
    u = torch.zeros_like(w)
    s_obs = ricker(T, 22.0, dt, 0.05).double()
    s_syn = ricker(T, 13.0, dt, 0.09).double()
    # source convolution commutes with the (linear) DAS operator
    obs = layer(u, _conv_time(w, s_obs))
    syn = layer(u, _conv_time(w, s_syn))                   # same w, wrong source
    e_si = float(ConvolvedWavefieldMisfit(dt=1).forward(syn, obs))
    l2 = float(((syn - obs) ** 2).sum())
    assert e_si < 1e-6 * l2, f"DAS SI failed: e_si={e_si:.3e} l2={l2:.3e}"
