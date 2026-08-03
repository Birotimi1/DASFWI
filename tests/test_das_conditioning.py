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
    assert lo == round(0.25 * (1500 / 3.0) / 40)        # 3.125 -> 3 cells
    assert hi == round(0.25 * (1500 / 6.25) / 40)       # 1.5   -> 2 cells


def test_wavelength_span_floor_and_validation():
    # a very high frequency would ask for sub-cell smoothing -> floored
    assert wavelength_span(1500, 100.0, 40) == 1
    for bad in ((0, 5, 40), (1500, 0, 40), (1500, 5, 0)):
        with pytest.raises(ValueError):
            wavelength_span(*bad)


def test_wavelength_span_is_usable_by_smooth2d():
    """THE TEST THAT WAS MISSING -- and it cost 16 cluster cells.

    The old version returned a float, which is arithmetically right and
    operationally useless: smooth2d builds its kernel with
    `np.linspace(-2*span, 2*span, 2*span + 1)` and linspace's `num` must be an
    integer, so every conditioned run died at the first gradient with
    "TypeError: 'float' object cannot be interpreted as an integer".

    The formula was unit-tested in isolation and never handed to its consumer.
    So this test calls the REAL smooth2d, the way GradProcessor does."""
    from ADFWI.propagator.gradient_process import smooth2d

    grad = np.random.default_rng(0).standard_normal((60, 100))
    for f_max in (3.0, 6.25, 20.0, 100.0):
        span = wavelength_span(1500.0, f_max, 40.0)
        assert isinstance(span, int) and span >= 1
        out = smooth2d(np.copy(grad), span=span)         # must not raise
        assert out.shape == grad.shape and np.isfinite(out).all()
    # and smoothing must actually smooth: less roughness than the input
    rough = lambda a: np.abs(np.diff(a, axis=0)).mean()
    assert rough(smooth2d(np.copy(grad), span=wavelength_span(1500, 3.0, 40))) \
        < rough(grad)


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
    first = m._conditioning(obs).clone()
    syn2 = _gather(shift=0.9).clone().requires_grad_(True)   # very different model
    m.forward(syn2, obs)
    assert torch.allclose(m._conditioning(obs), first)   # obs decides, not syn
    # and a DIFFERENT batch of shots must get its OWN window, not the first one's
    other = _gather(shift=0.6)
    assert not torch.allclose(m._conditioning(other), first)


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


# --------------------------------------------------------------------------- #
# 5. the wrapper must not HIDE the controller API
# --------------------------------------------------------------------------- #
def test_conditioning_does_not_hide_the_switch_api():
    """REGRESSION -- this killed 8 Bridges-2 cells.

    run_switch.py drives the misfit THROUGH whatever wraps it: the blend arms
    call set_lambda / read .lam, and the ladder arm calls set_stage / reads
    .active_name. Wrapping in ConditionedMisfit made those invisible, so every
    conditioned cell raised AttributeError at the FIRST controller update --
    after printing its setup line, which is exactly where the cluster logs
    stopped. Conditioning is documented to compose with the switch; this test
    is what makes that claim real, and it exercises BOTH arms."""
    from inversion.adaptive_misfit import BlendedMisfit, StagedMisfit
    from ADFWI.fwi.misfit import Misfit_envelope

    # float32 like the real pipeline -- Misfit_envelope allocates its residual
    # buffer as float32, so a float64 fixture would fail inside the misfit and
    # mask what this test is actually about.
    obs = _gather().float()
    syn = _gather(shift=0.05).float().clone().requires_grad_(True)

    # --- blend arms: set_lambda / .lam  (run_switch.py:371, 387) -------------
    blend = BlendedMisfit(Misfit_waveform_L2(dt=DT), Misfit_envelope(dt=DT))
    m = ConditionedMisfit(blend, dt=DT, window=True, weight=True,
                          window_pre=0.1, window_post=0.2)
    for lam in (1.0, 0.0):
        m.set_lambda(lam)                      # would AttributeError before
        assert m.lam == lam and blend.lam == lam        # mutates the INNER obj
    assert torch.isfinite(m.forward(syn, obs)).all()

    # --- ladder arm: set_stage / .active_name  (run_switch.py:366-367) -------
    staged = StagedMisfit([Misfit_envelope(dt=DT), Misfit_waveform_L2(dt=DT)],
                          names=["envelope", "l2"])
    ms = ConditionedMisfit(staged, dt=DT, window=True,
                           window_pre=0.1, window_post=0.2)
    ms.set_stage(1)
    assert ms.active_name == "l2" == staged.active_name
    ms.set_stage(0)
    assert ms.active_name == "envelope"
    assert torch.isfinite(ms.forward(syn, obs)).all()

    # a genuinely missing attribute must still raise, not silently return None
    with pytest.raises(AttributeError):
        m.no_such_attribute


def test_channel_weighting_distorts_a_nonlinear_misfit():
    """REGRESSION for the A/B collapse (2026-08-03): switch+c fell 0.742 -> 0.26
    at the skip starter while l2+c was unharmed at 0.614.

    ConditionedMisfit weights the DATA, but the intent is to weight each
    channel's CONTRIBUTION. Those agree only for a quadratic misfit; through a
    nonlinear one the effective exponent is uncontrolled, so a weak channel is
    suppressed far harder than asked. The switch's rescue IS its envelope stage,
    so this quietly removed the rescue.

    Pinned as a measured FACT so the asymmetry cannot be reintroduced unnoticed
    and so any future per-channel-contribution fix has a target to beat."""
    from ADFWI.fwi.misfit import Misfit_envelope

    # float32 like the real pipeline: Misfit_envelope allocates a float32
    # residual buffer, so a float64 fixture fails inside the misfit itself.
    obs = _gather(weak=(3, 7)).float()
    syn = _gather(shift=0.03, weak=(3, 7)).float()
    w = channel_weights(obs)
    wk = float(w[0, 0, 3])
    assert wk < 0.2                                    # a genuinely weak channel

    def share(M, ch):
        plain = float(M.forward(syn[:, :, ch:ch + 1], obs[:, :, ch:ch + 1]))
        wtd = float(M.forward((syn * w)[:, :, ch:ch + 1],
                              (obs * w)[:, :, ch:ch + 1]))
        return wtd / max(plain, 1e-30)

    s_l2 = share(Misfit_waveform_L2(dt=DT), 3)
    s_env = share(Misfit_envelope(dt=DT, p=1.5), 3)
    # the envelope suppresses the SAME channel by orders of magnitude more
    assert s_env < 0.05 * s_l2, (
        f"expected the nonlinear misfit to over-suppress: l2 x{s_l2:.5f} vs "
        f"envelope x{s_env:.5f}")
