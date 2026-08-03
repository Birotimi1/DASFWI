"""Fichtner TF-phase misfit -- verified against ANALYTICALLY KNOWN answers.

A phase misfit with a subtly wrong adjoint produces confident nonsense: it will
still return a smooth decreasing loss curve while driving the model somewhere
meaningless, and we would be scoring our adaptive switch against a broken
baseline with no tell at all. So every property is checked against a case whose
answer is known in closed form BEFORE the misfit is allowed to score anything:

    pure time shift dt  ->  dphi(w) = -w * dt exactly (note the SIGN, which the
                            first run of this test caught: it recovered -0.00399 s
                            for a true +0.00400 s), so the measured phase slope
                            must return dt to within the TF resolution
    |dt| > 1/(2f)       ->  that row WRAPS (this is the cycle-skip condition
                            itself; the test asserts the limitation exists
                            rather than pretending it does not)
    autograd            ->  must match central finite differences
    empty / dead traces ->  no NaN in value OR gradient
"""
import numpy as np
import pytest
import torch

from inversion.tf_phase import (Misfit_TFPhase, gaussian_window, gabor,
                                wrap_limit_s)

DT, NT, F0 = 0.002, 1024, 15.0
T0 = 0.4


def _ricker(shift=0.0, n_ch=4, amp=1.0, dtype=torch.float64):
    """[1, NT, n_ch] Ricker wavelets, all shifted by `shift` seconds."""
    t = np.arange(NT) * DT
    g = np.zeros((1, NT, n_ch))
    for c in range(n_ch):
        tt = t - (T0 + shift)
        g[0, :, c] = amp * (1 - 2 * (np.pi * F0 * tt) ** 2) * \
            np.exp(-(np.pi * F0 * tt) ** 2)
    return torch.tensor(g, dtype=dtype)


def _misfit(**kw):
    kw.setdefault("win_s", 0.3)
    kw.setdefault("f_min", 5.0)
    kw.setdefault("f_max", 40.0)
    return Misfit_TFPhase(dt=DT, **kw)


# --------------------------------------------------------------------------- #
# 1. the analytic identity the whole method rests on:  dphi = w * dt
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shift", [0.004, 0.008, -0.006])
def test_phase_slope_recovers_a_known_time_shift(shift):
    """For a pure translation the phase difference is EXACTLY linear in
    frequency with slope 2*pi*shift. If this fails, the transform, the branch
    handling, or the frequency axis is wrong -- and every later number is void."""
    m = _misfit()
    obs, syn = _ricker(), _ricker(shift=shift)
    est = float(m.time_shift(syn, obs))
    assert est == pytest.approx(shift, abs=0.15 * abs(shift) + 3e-4), \
        f"recovered {est:.5f} s, expected {shift:.5f} s"
    # SIGN, pinned explicitly: reading dphi/omega without the minus returns the
    # shift NEGATED, which is what the first version of this test caught.
    dphi, w, freqs = m.phase_difference(syn, obs)
    om = torch.tensor(2 * np.pi * freqs, dtype=dphi.dtype).reshape(1, -1, 1)
    raw = ((w ** 2 * dphi * om).sum() / (w ** 2 * om ** 2).sum()).item()
    assert raw == pytest.approx(-shift, abs=0.15 * abs(shift) + 3e-4)


def test_zero_shift_is_zero_misfit():
    m = _misfit()
    obs = _ricker()
    assert float(m.forward(obs.clone(), obs)) < 1e-12


def test_misfit_grows_with_shift_below_the_wrap_limit():
    m = _misfit()
    obs = _ricker()
    prev = -1.0
    for s in (0.0, 0.002, 0.004, 0.006):          # all < 1/(2*40Hz) = 12.5 ms
        e = float(m.forward(_ricker(shift=s), obs))
        assert e > prev
        prev = e


# --------------------------------------------------------------------------- #
# 2. the LIMITATION, asserted rather than glossed over
# --------------------------------------------------------------------------- #
def test_wrap_limit_is_the_cycle_skip_threshold():
    """TF-phase does not abolish cycle skipping: each row wraps at |dt| > T/2,
    the identical condition. Low rows wrap later -- that, and only that, is
    where its skip resistance comes from."""
    f = np.array([3.0, 6.25, 20.0, 40.0])
    assert np.allclose(wrap_limit_s(f), 1.0 / (2 * f))
    assert wrap_limit_s(np.array([3.0]))[0] > wrap_limit_s(np.array([40.0]))[0]


def test_high_frequency_rows_wrap_first():
    """Direct demonstration: with a shift beyond the 40 Hz limit but inside the
    5 Hz one, the top rows report a wrapped (wrong-sign or folded) phase while
    the bottom rows still read the shift correctly."""
    shift = 0.02                       # > 1/(2*40)=12.5 ms, < 1/(2*5)=100 ms
    m = _misfit()
    dphi, w, freqs = m.phase_difference(_ricker(shift=shift), _ricker())
    est = -dphi / torch.tensor(2 * np.pi * freqs, dtype=dphi.dtype).reshape(1, -1, 1)
    lo = (freqs < 1 / (2 * shift))                       # rows that CANNOT wrap
    hi = (freqs > 1 / (2 * shift))                       # rows that MUST wrap
    if w[:, lo].sum() > 0 and w[:, hi].sum() > 0:
        e_lo = (est[:, lo] * w[:, lo] ** 2).sum() / (w[:, lo] ** 2).sum()
        e_hi = (est[:, hi] * w[:, hi] ** 2).sum() / (w[:, hi] ** 2).sum()
        assert abs(float(e_lo) - shift) < abs(float(e_hi) - shift)


# --------------------------------------------------------------------------- #
# 3. THE ADJOINT -- the check that gates everything downstream
# --------------------------------------------------------------------------- #
def test_adjoint_matches_finite_differences():
    """Autograd vs central differences on the real objective. This is the test
    that has to pass before TF-phase is allowed to produce a single score."""
    torch.manual_seed(0)
    m = _misfit()
    obs = _ricker()
    syn = _ricker(shift=0.005).clone().requires_grad_(True)
    e = m.forward(syn, obs)
    g, = torch.autograd.grad(e, syn)
    assert torch.isfinite(g).all()

    flat = syn.detach().reshape(-1)
    # probe where the wavefield actually has energy; a zero-amplitude sample has
    # a legitimately ~zero derivative and would make the comparison vacuous
    big = torch.topk(flat.abs(), 200).indices
    probes = big[torch.randperm(len(big))[:12]]
    h = 1e-6
    for i in probes:
        p = flat.clone(); p[i] += h
        ep = float(m.forward(p.reshape(syn.shape), obs))
        p = flat.clone(); p[i] -= h
        em = float(m.forward(p.reshape(syn.shape), obs))
        fd = (ep - em) / (2 * h)
        ad = float(g.reshape(-1)[i])
        assert fd == pytest.approx(ad, rel=2e-3, abs=1e-7), \
            f"index {int(i)}: autograd {ad:.6e} vs finite-diff {fd:.6e}"


def test_gradient_points_downhill():
    """A step along -grad must REDUCE the misfit. Cheap, and it catches a sign
    error that a finite-difference check on a symmetric point can miss."""
    m = _misfit()
    obs = _ricker()
    syn = _ricker(shift=0.006).clone().requires_grad_(True)
    e0 = m.forward(syn, obs)
    g, = torch.autograd.grad(e0, syn)
    stepped = (syn.detach() - 1e-3 * g / g.norm().clamp_min(1e-30))
    assert float(m.forward(stepped, obs)) < float(e0)


# --------------------------------------------------------------------------- #
# 4. numerics: the empty-cell trap that makes phase misfits explode
# --------------------------------------------------------------------------- #
def test_dead_traces_produce_no_nan():
    """DAS gathers really do contain dead / broadside-insensitive channels, and
    d(arg z)/dz ~ 1/|z| is unbounded there. Value AND gradient must stay finite."""
    m = _misfit()
    obs, syn = _ricker(n_ch=4), _ricker(shift=0.004, n_ch=4)
    obs[0, :, 1] = 0.0                       # dead channel
    syn[0, :, 1] = 0.0
    syn[0, :, 2] *= 1e-12                    # near-dead
    syn = syn.clone().requires_grad_(True)
    e = m.forward(syn, obs)
    g, = torch.autograd.grad(e, syn)
    assert torch.isfinite(e).all() and torch.isfinite(g).all()


def test_all_zero_input_is_finite():
    m = _misfit()
    z = torch.zeros(1, NT, 3, dtype=torch.float64, requires_grad=True)
    e = m.forward(z, torch.zeros(1, NT, 3, dtype=torch.float64))
    g, = torch.autograd.grad(e, z)
    assert torch.isfinite(e).all() and torch.isfinite(g).all()


def test_amplitude_insensitive():
    """Phase-only by construction: scaling a trace must not change the misfit.
    This is the property that makes it right for field DAS, where coupling
    scales amplitudes (measured at FORGE: a scalar, not a filter)."""
    m = _misfit()
    obs = _ricker()
    syn = _ricker(shift=0.004)
    e1 = float(m.forward(syn, obs))
    e2 = float(m.forward(syn * 37.0, obs * 0.11))
    assert e2 == pytest.approx(e1, rel=1e-6)


# --------------------------------------------------------------------------- #
# 5. composability with the switch
# --------------------------------------------------------------------------- #
def test_is_stateless_so_it_can_blend():
    """BlendedMisfit rejects stateful terms (adaptive_misfit._reject_stateful),
    because a term carrying its own iteration schedule silently changes meaning
    when the controller re-evaluates it. weci was caught by that guard."""
    m = _misfit()
    obs, syn = _ricker(), _ricker(shift=0.004)
    first = float(m.forward(syn, obs))
    for _ in range(3):
        assert float(m.forward(syn, obs)) == pytest.approx(first, rel=1e-12)


def test_accepts_the_shapes_the_pipeline_uses():
    m = _misfit()
    obs3, syn3 = _ricker(), _ricker(shift=0.003)
    assert torch.isfinite(m.forward(syn3, obs3)).all()          # [S, nt, C]
    assert torch.isfinite(m.forward(syn3[0], obs3[0])).all()    # [nt, C]


def test_composes_inside_the_switch():
    """TF-phase is both a BASELINE to beat and a candidate TERM for the switch,
    so it has to survive BlendedMisfit -- including _reject_stateful, which
    already refused weci for carrying its own iteration schedule."""
    from ADFWI.fwi.misfit import Misfit_waveform_L2
    from inversion.adaptive_misfit import BlendedMisfit

    obs = _ricker()
    syn = _ricker(shift=0.004).clone().requires_grad_(True)
    b = BlendedMisfit(Misfit_waveform_L2(dt=DT), _misfit(), lam=1.0)
    for lam in (1.0, 0.0, 0.5):
        b.set_lambda(lam)
        e = b.forward(syn, obs)
        g, = torch.autograd.grad(e, syn, retain_graph=True)
        assert torch.isfinite(e).all() and torch.isfinite(g).all(), f"lam={lam}"
        assert float(g.norm()) > 0


def test_registered_in_the_config_registry():
    """Drivers resolve misfits by NAME through inversion.config, so an
    unregistered misfit is unreachable from every campaign script."""
    from inversion import config
    assert "tfphase" in config.MISFITS
    assert "tfphase" in config.MISFIT_SETTINGS
    m = config.build_misfit("tfphase", dt=DT, iterations=300)
    assert isinstance(m, Misfit_TFPhase)
