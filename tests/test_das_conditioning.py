"""DAS conditioning (Noe et al. 2025): window, channel weights, lambda/4 span."""
import numpy as np
import pytest
import torch

from ADFWI.fwi.misfit import Misfit_waveform_L2
from inversion.das_conditioning import (wavelength_span, arrival_window,
                                        channel_weights, ConditionedMisfit)

DT, NT = 0.003, 800


def _gather(shift=0.0, n_ch=10, weak=()):
    """(1, NT, n_ch) Ricker arrivals; `weak` channels are scaled down to imitate
    broadside insensitivity / poor coupling."""
    t = np.arange(NT) * DT
    g = np.zeros((1, NT, n_ch))
    for j in range(n_ch):
        tt = t - (0.5 + shift + 0.02 * j)
        amp = 0.01 if j in weak else 1.0
        g[0, :, j] = amp * (1 - 2 * (np.pi * 5 * tt) ** 2) * \
            np.exp(-(np.pi * 5 * tt) ** 2)
    return torch.from_numpy(g)


# --------------------------------------------------------------------------- #
# 1. wavelength-scaled gradient smoothing
# --------------------------------------------------------------------------- #
def test_wavelength_span_is_frequency_aware():
    """The whole point: LOW frequency must smooth HARDER than high, because the
    data cannot support fine structure there. A fixed Tikhonov weight cannot
    express this."""
    lo = wavelength_span(v_min=1500, f_max=3.0, dx=40)
    hi = wavelength_span(v_min=1500, f_max=6.25, dx=40)
    assert lo > hi
    assert lo == pytest.approx(0.25 * (1500 / 3.0) / 40, rel=1e-9)      # ~3.1 cells
    assert hi == pytest.approx(0.25 * (1500 / 6.25) / 40, rel=1e-9)     # ~1.5 cells


def test_wavelength_span_floor_and_validation():
    # a very high frequency would ask for sub-cell smoothing -> floored
    assert wavelength_span(1500, 100.0, 40) == 1.0
    for bad in ((0, 5, 40), (1500, 0, 40), (1500, 5, 0)):
        with pytest.raises(ValueError):
            wavelength_span(*bad)


# --------------------------------------------------------------------------- #
# 2. arrival windowing
# --------------------------------------------------------------------------- #
def test_window_keeps_the_arrival_and_removes_the_rest():
    obs = _gather()
    m = arrival_window(obs, DT, pre=0.1, post=0.2)
    assert m.shape == obs.shape
    peak = obs.abs().argmax(dim=1)
    for j in range(obs.shape[2]):
        p = int(peak[0, j])
        assert m[0, p, j] == pytest.approx(1.0, abs=1e-6)     # arrival kept
        far = min(p + int(1.0 / DT), NT - 1)                  # 1 s later
        assert float(m[0, far, j]) < 1e-3                     # coda removed
    assert float(m.min()) >= 0.0 and float(m.max()) <= 1.0


def test_window_follows_each_trace_and_is_tapered():
    """Different channels peak at different times, so the window must move with
    them -- and the edges must be tapered, or the cut injects a step."""
    obs = _gather()
    m = arrival_window(obs, DT, pre=0.1, post=0.2)
    centres = [float((m[0, :, j] * torch.arange(NT)).sum() / m[0, :, j].sum())
               for j in range(obs.shape[2])]
    assert centres[-1] > centres[0] + 10        # later channels, later windows
    edge = m[0, :, 0]
    assert 0.0 < float(edge[(edge > 0).nonzero()[0]]) < 1.0    # cosine ramp


# --------------------------------------------------------------------------- #
# 3. channel weighting
# --------------------------------------------------------------------------- #
def test_weak_channels_are_down_weighted():
    """DAS broadside insensitivity: a channel can record almost nothing by
    GEOMETRY, not just poor coupling -- so this matters on synthetics too."""
    obs = _gather(weak=(3, 7))
    w = channel_weights(obs)
    strong = float(w[0, 0, 0])
    assert float(w[0, 0, 3]) < 0.2 * strong
    assert float(w[0, 0, 7]) < 0.2 * strong
    assert float(w.min()) > 0                      # floored: never zero/NaN
    assert float(w.mean()) == pytest.approx(1.0, rel=1e-6)     # loss scale kept


# --------------------------------------------------------------------------- #
# 4. the wrapper
# --------------------------------------------------------------------------- #
def test_conditioning_is_derived_from_obs_and_held_fixed():
    """THE design point: the window/weights must come from the OBSERVED gather
    and never change, or the objective moves as the model does and the inversion
    chases a shifting target."""
    obs = _gather()
    syn1 = _gather(shift=0.05).clone().requires_grad_(True)
    m = ConditionedMisfit(Misfit_waveform_L2(dt=DT), dt=DT, window=True,
                          window_pre=0.1, window_post=0.2)
    m.forward(syn1, obs)
    cached = m._mask.clone()
    syn2 = _gather(shift=0.9).clone().requires_grad_(True)   # very different model
    m.forward(syn2, obs)
    assert torch.allclose(m._mask, cached)        # unchanged by the synthetic


def test_conditioning_off_is_a_no_op():
    obs = _gather()
    syn = _gather(shift=0.05).clone().requires_grad_(True)
    inner = Misfit_waveform_L2(dt=DT)
    plain = float(inner.forward(syn, obs))
    wrapped = ConditionedMisfit(inner, dt=DT, window=False, weight=False)
    assert float(wrapped.forward(syn, obs)) == pytest.approx(plain)


def test_conditioning_keeps_gradients_flowing():
    obs = _gather()
    syn = _gather(shift=0.05).clone().requires_grad_(True)
    m = ConditionedMisfit(Misfit_waveform_L2(dt=DT), dt=DT, window=True,
                          weight=True, window_pre=0.1, window_post=0.2)
    e = m.forward(syn, obs)
    g, = torch.autograd.grad(e, syn)
    assert torch.isfinite(g).all() and float(g.norm()) > 0


def test_windowing_reduces_the_misfit_of_a_late_mismatch():
    """Why windowing is also SKIP mitigation: a discrepancy long after the
    arrival (where phase error accumulates) is excluded from the objective."""
    obs = _gather()
    syn = _gather().clone()
    syn[0, int(0.5 / DT) + int(1.5 / DT):, :] += 0.5      # late-only corruption
    syn = syn.requires_grad_(True)
    inner = Misfit_waveform_L2(dt=DT)
    raw = float(inner.forward(syn, obs))
    win = float(ConditionedMisfit(inner, dt=DT, window=True, window_pre=0.1,
                                  window_post=0.2).forward(syn, obs))
    assert win < raw
